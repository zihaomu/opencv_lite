//
// Created by mzh on 2023/7/12.
//

#include "CPULandmarks2TransformMatrix.h"
#include "backend/cpu/CPUBackend.hpp"
#include "tflite_header.h"
#include "iostream"
#include <cmath>


namespace MNN {

void EstimateRotationRadians(const float* input_data_0, int left_rotation_idx,
                             int right_rotation_idx,
                             float target_rotation_radians,
                             float* rotation_radians) {
    const float3 left_landmark =
            Read3DLandmarkXYZ(input_data_0, left_rotation_idx);
    const float3 right_landmark =
            Read3DLandmarkXYZ(input_data_0, right_rotation_idx);
    const float left_x = left_landmark.x;
    const float left_y = left_landmark.y;
    const float right_x = right_landmark.x;
    const float right_y = right_landmark.y;
    float rotation = std::atan2(right_y - left_y, right_x - left_x);
    rotation = target_rotation_radians - rotation;
    *rotation_radians = rotation;
}

void EstimateCenterAndSize(const float* input_data_0,
                           const std::vector<int>& subset_idxs,
                           float rotation_radians, float* crop_x, float* crop_y,
                           float* crop_width, float* crop_height) {
    std::vector<float3> landmarks;
    landmarks.reserve(subset_idxs.size());

    for (int i = 0; i < subset_idxs.size(); i++) {
        landmarks.push_back(Read3DLandmarkXYZ(input_data_0, subset_idxs[i]));
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
    for (int i = 0; i < landmarks.size(); i++)
    {
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

CPULandmarks2TransformMatrix::CPULandmarks2TransformMatrix(const MNN::Op* op, Backend *b) : Execution(b)
{
    auto param = op->main_as_Landmarks2TransformMatrixParam();

    left_rotation_idx = param->left_rotation_idx();
    output_height = param->output_height();
    output_width = param->output_width();
    right_rotation_idx = param->right_rotation_idx();
    scale_x = param->scale_x();
    scale_y = param->scale_y();
    target_rotation_radians = param->target_rotation_radians();

    int pSize = param->subset_idxs()->size();
    subset_idxs.resize(pSize);
    auto p = param->subset_idxs()->data();

    for (int i = 0; i < pSize; i++)
    {
        subset_idxs[i] = p[i];
    }
}

ErrorCode CPULandmarks2TransformMatrix::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
{
    auto input  = inputs[0];
    auto output = outputs[0];

    auto inpData = input->host<float>();
    auto outData = output->host<float>();

    float rotation_radians = 0.0;
    EstimateRotationRadians(inpData, left_rotation_idx,
                            right_rotation_idx,
                            target_rotation_radians, &rotation_radians);

    float crop_x = 0.0, crop_y = 0.0, crop_width = 0.0, crop_height = 0.0;
    EstimateCenterAndSize(inpData, subset_idxs, rotation_radians,
                          &crop_x, &crop_y, &crop_width, &crop_height);

#if MNN_DEBUG
    std::cout<<"Landmarks2TransformMatrix = rotation_radians "<<rotation_radians<<", crop_x = "<<crop_x<<"crop_y = "<<crop_y<<
    ", crop_width = "<<crop_width<<", crop_height = "<<crop_height<<std::endl;
#endif

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

#if MNN_DEBUG
    std::cout<<"step 1: mat t = "<<std::endl;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            std::cout<<","<<t.data[i * 4 +j];
        }
        std::cout<<std::endl;
    }
#endif

    t *= t_rotation;
    const float scale_x = this->scale_x * crop_width / output_width;
    const float scale_y = this->scale_y * crop_height / output_height;
    const Mat4 t_scale = Mat4(scale_x,     0.0, 0.0, 0.0,
                              0.0, scale_y, 0.0, 0.0,
                              0.0,     0.0, 1.0, 0.0,
                              0.0,     0.0, 0.0, 1.0);

#if MNN_DEBUG
    std::cout<<"scale_x = "<<scale_x<<", scale_y = "<<scale_y<<std::endl;
    std::cout<<"step 2: mat t = "<<std::endl;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            std::cout<<","<<t.data[i * 4 +j];
        }
        std::cout<<std::endl;
    }
#endif

    t *= t_scale;
    const float shift_x = -1.0 * (output_width / 2.0);
    const float shift_y = -1.0 * (output_height / 2.0);
    const Mat4 t_shift2 = Mat4(1.0, 0.0, 0.0, shift_x,
                               0.0, 1.0, 0.0, shift_y,
                               0.0, 0.0, 1.0,     0.0,
                               0.0, 0.0, 0.0,     1.0);
#if MNN_DEBUG
    std::cout<<"step 3: mat t = "<<std::endl;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            std::cout<<","<<t.data[i * 4 +j];
        }
        std::cout<<std::endl;
    }
#endif

    t *= t_shift2;
    std::memcpy(outData, t.data.data(), 16 * sizeof(float));

#if MNN_DEBUG
    std::cout<<"print output on Landmarks2TransformMatrixV2"<<std::endl;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            std::cout<<","<<outData[i * 4 +j];
        }
        std::cout<<std::endl;
    }
#endif
    return NO_ERROR;
}

class CPULandmarks2TransformMatrixCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs,
                                const MNN::Op *op,
                                Backend *backend) const override {

        return new CPULandmarks2TransformMatrix(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPULandmarks2TransformMatrixCreator, OpType_Landmarks2TransformMatrix);
}
