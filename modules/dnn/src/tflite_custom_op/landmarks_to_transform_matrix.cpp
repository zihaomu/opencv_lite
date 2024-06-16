// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_register.h"
#include "custom_op_impl.h"

#ifdef HAVE_TFLITE
#include "flatbuffers/flexbuffers.h"
#include <vector>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::tflite::GetInput;
using ::tflite::GetOutput;
using ::tflite::GetTensorData;
using ::tflite::GetTensorShape;
using ::tflite::NumDimensions;
using ::tflite::NumInputs;
using ::tflite::NumOutputs;
using ::tflite::RuntimeShape;

namespace cv {
namespace dnn {
namespace mp_op {

//struct LandmarksToTransformMatrixV1Attributes {
//    int dimensions;
//    int landmarks_range;
//    int left_rotation_idx;
//    int right_rotation_idx;
//    float bbox_size_multiplier;
//    HW input_hw;
//    HW output_hw;
//    std::vector<int2> subset;
//};

struct LandmarksToTransformMatrixV2Attributes {
    std::vector<int2> subset_idxs;
    int left_rotation_idx;
    int right_rotation_idx;
    float target_rotation_radians;
    int output_height;
    int output_width;
    float scale_x;
    float scale_y;
    float multiplier = 1.0;
};

bool ParseLandmarksToTransformMatrixV2Attributes(
    const void* data, uint32_t data_size,
LandmarksToTransformMatrixV2Attributes* attr, std::vector<int>& output_shape)
{
    const flexbuffers::Map m =
              flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(data), data_size)
                  .AsMap();
    const auto subset_idxs = m["subset_idxs"].AsTypedVector();
    int amount = subset_idxs.size();
    for (int i = 0; i < amount / 2; i++) {
    attr->subset_idxs.emplace_back(subset_idxs[i * 2].AsInt32(),
                                   subset_idxs[i * 2 + 1].AsInt32());
    }
    if (amount % 2 != 0) {
    int previous = amount - 1;
    attr->subset_idxs.emplace_back(subset_idxs[previous].AsInt32(),
                               subset_idxs[previous].AsInt32());
    }
    attr->left_rotation_idx = m["left_rotation_idx"].AsInt32();
    attr->right_rotation_idx = m["right_rotation_idx"].AsInt32();
    attr->target_rotation_radians = m["target_rotation_radians"].AsFloat();
    attr->output_height = m["output_height"].AsInt32();
    attr->output_width = m["output_width"].AsInt32();
    attr->scale_x = m["scale_x"].AsFloat();
    attr->scale_y = m["scale_y"].AsFloat();

    output_shape = {1, 1, 4, 4};
    return true;
}

namespace {

constexpr int kDataInputTensor = 0;
constexpr int kOutputTensor = 0;
const int3 kTensformMatrixShape(1, 4, 4);

float2 Read3DLandmarkXY(const float* data, int idx) {
  float2 result;
  result.x = data[idx * 3];
  result.y = data[idx * 3 + 1];
  return result;
}

float3 Read3DLandmarkXYZ(const float* data, int idx) {
  float3 result;
  result.x = data[idx * 3];
  result.y = data[idx * 3 + 1];
  result.z = data[idx * 3 + 2];
  return result;
}

namespace v2 {

void EstimateRotationRadians(const float* input_data_0, int left_rotation_idx,
                             int right_rotation_idx,
                             float target_rotation_radians,
                             float* rotation_radians) {
  const float3 left_landmark =
      Read3DLandmarkXYZ(input_data_0, left_rotation_idx);
  const float3 right_landmark =
      Read3DLandmarkXYZ(input_data_0, right_rotation_idx);
  const float left_x = left_landmark[0];
  const float left_y = left_landmark[1];
  const float right_x = right_landmark[0];
  const float right_y = right_landmark[1];
  float rotation = std::atan2(right_y - left_y, right_x - left_x);
  rotation = target_rotation_radians - rotation;
  *rotation_radians = rotation;
}

void EstimateCenterAndSize(const float* input_data_0,
                           std::vector<int2> subset_idxs,
                           float rotation_radians, float* crop_x, float* crop_y,
                           float* crop_width, float* crop_height) {
  std::vector<float3> landmarks;
  landmarks.reserve(subset_idxs.size() * 2);
  for (int i = 0; i < subset_idxs.size(); i++) {
    landmarks.push_back(Read3DLandmarkXYZ(input_data_0, subset_idxs[i][0]));
    landmarks.push_back(Read3DLandmarkXYZ(input_data_0, subset_idxs[i][1]));
  }
  for (int i = 0; i < landmarks.size(); i++) {
    landmarks[i].z = 1.0;
  }
  const float& r = rotation_radians;
  // clang-format off
  const Mat3 t_rotation = Mat3(std::cos(r),  -std::sin(r), 0.0,
                               std::sin(r),   std::cos(r), 0.0,
                                       0.0,           0.0, 1.0);
  const Mat3 t_rotation_inverse =
                          Mat3(std::cos(-r), -std::sin(-r), 0.0,
                               std::sin(-r),  std::cos(-r), 0.0,
                                        0.0,           0.0, 1.0);
  // clang-format on
  for (int i = 0; i < landmarks.size(); i++) {
    landmarks[i] = t_rotation * landmarks[i];
  }
  float3 xy1_max = landmarks[0], xy1_min = landmarks[0];
  for (int i = 1; i < landmarks.size(); i++) {
    if (xy1_max.x < landmarks[i].x) xy1_max.x = landmarks[i].x;
    if (xy1_max.y < landmarks[i].y) xy1_max.y = landmarks[i].y;

    if (xy1_min.x > landmarks[i].x) xy1_min.x = landmarks[i].x;
    if (xy1_min.y > landmarks[i].y) xy1_min.y = landmarks[i].y;
  }
  *crop_width = xy1_max.x - xy1_min.x;
  *crop_height = xy1_max.y - xy1_min.y;
  float3 crop_xy1 = xy1_min;
  crop_xy1.x += xy1_max.x;
  crop_xy1.y += xy1_max.y;
  crop_xy1.x /= 2;
  crop_xy1.y /= 2;
  crop_xy1 = t_rotation_inverse * crop_xy1;
  *crop_x = crop_xy1.x;
  *crop_y = crop_xy1.y;
}

inline void LandmarksToTransformMatrixV2(
    const LandmarksToTransformMatrixV2Attributes& params,
    const RuntimeShape& input0_shape, const float* landmarks,
    const RuntimeShape& output_shape, float* output_data) {
  float rotation_radians = 0.0;
  EstimateRotationRadians(landmarks, params.left_rotation_idx,
                          params.right_rotation_idx,
                          params.target_rotation_radians, &rotation_radians);
  float crop_x = 0.0, crop_y = 0.0, crop_width = 0.0, crop_height = 0.0;
  EstimateCenterAndSize(landmarks, params.subset_idxs, rotation_radians,
                        &crop_x, &crop_y, &crop_width, &crop_height);
  // Turn off clang formatting to make matrices initialization more readable.
  // clang-format off
  Mat4 t = Mat4(1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0);
  const Mat4 t_shift = Mat4(1.0, 0.0, 0.0, crop_x,
                            0.0, 1.0, 0.0, crop_y,
                            0.0, 0.0, 1.0,    0.0,
                            0.0, 0.0, 0.0,    1.0);
  t *= t_shift;
  const float& r = -rotation_radians;
  const Mat4 t_rotation = Mat4(std::cos(r), -std::sin(r), 0.0, 0.0,
                               std::sin(r),  std::cos(r), 0.0, 0.0,
                                       0.0,          0.0, 1.0, 0.0,
                                       0.0,          0.0, 0.0, 1.0);
  t *= t_rotation;
  const float scale_x = params.scale_x * crop_width / params.output_width;
  const float scale_y = params.scale_y * crop_height / params.output_height;
  const Mat4 t_scale = Mat4(scale_x,     0.0, 0.0, 0.0,
                                0.0, scale_y, 0.0, 0.0,
                                0.0,     0.0, 1.0, 0.0,
                                0.0,     0.0, 0.0, 1.0);
  t *= t_scale;
  const float shift_x = -1.0 * (params.output_width / 2.0);
  const float shift_y = -1.0 * (params.output_height / 2.0);
  const Mat4 t_shift2 = Mat4(1.0, 0.0, 0.0, shift_x,
                             0.0, 1.0, 0.0, shift_y,
                             0.0, 0.0, 1.0,     0.0,
                             0.0, 0.0, 0.0,     1.0);
  t *= t_shift2;
  std::memcpy(output_data, t.data.data(), 16 * sizeof(float));
  // clang-format on
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 3);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = kTensformMatrixShape[0];
  output_size->data[1] = kTensformMatrixShape[1];
  output_size->data[2] = kTensformMatrixShape[2];

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  LandmarksToTransformMatrixV2Attributes op_params;
  std::vector<int> output_shape;
  auto status = ParseLandmarksToTransformMatrixV2Attributes(
      node->custom_initial_data, node->custom_initial_data_size, &op_params,
      output_shape);
  if (!status) {
    context->ReportError(context, "tflite fails to call ParseLandmarksToTransformMatrixV2Attributes!");
    return kTfLiteError;
  }

  if (op_params.left_rotation_idx < 0) {
    context->ReportError(context, "Incorrect left_rotation_idx: %d",
                         op_params.left_rotation_idx);
    return kTfLiteError;
  }

  if (op_params.right_rotation_idx < 0) {
    context->ReportError(context, "Incorrect right_rotation_idx: %d",
                         op_params.right_rotation_idx);
    return kTfLiteError;
  }

  if (op_params.output_height <= 0) {
    context->ReportError(context, "Incorrect output_height: %d",
                         op_params.output_height);
    return kTfLiteError;
  }

  if (op_params.output_width <= 0) {
    context->ReportError(context, "Incorrect output_width: %d",
                         op_params.output_width);
    return kTfLiteError;
  }

  if (op_params.scale_x <= 0) {
    context->ReportError(context, "Incorrect scale_x: %d", op_params.scale_x);
    return kTfLiteError;
  }

  if (op_params.scale_y <= 0) {
    context->ReportError(context, "Incorrect scale_y: %d", op_params.scale_y);
    return kTfLiteError;
  }

  int counter = 0;
  for (auto& val : op_params.subset_idxs) {
    for (int i = 0; i < 2; i++) {
      if (val[i] < 0) {
        context->ReportError(context,
                             "Incorrect subset value: index = %d, value = %d",
                             counter, val[i]);
        return kTfLiteError;
      }
      counter++;
    }
  }

  const TfLiteTensor* input0 = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input0 != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  LandmarksToTransformMatrixV2(
      op_params, GetTensorShape(input0), GetTensorData<float>(input0),
      GetTensorShape(output), GetTensorData<float>(output));
  return kTfLiteOk;
}

}  // namespace v2

}  // namespace

//TfLiteRegistration* RegisterLandmarksToTransformMatrixV1() {
//  static TfLiteRegistration reg = {
//      /*.init=*/nullptr,
//      /*.free=*/nullptr,
//      /*.prepare=*/v1::Prepare,
//      /*.invoke=*/v1::Eval,
//      /*.profiling_string=*/nullptr,
//      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
//      /*.custom_name=*/"Landmarks2TransformMatrix",
//      /*.version=*/1,
//  };
//  return &reg;
//}
TfLiteRegistration* RegisterLandmarksToTransformMatrixV2() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/v2::Prepare,
      /*.invoke=*/v2::Eval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
      /*.custom_name=*/"Landmarks2TransformMatrix",
      /*.version=*/2,
  };
  return &reg;
}


}}}

#endif