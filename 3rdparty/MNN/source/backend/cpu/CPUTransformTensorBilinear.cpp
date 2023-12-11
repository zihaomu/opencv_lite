//
// Created by mzh on 2023/7/12.
//

#include "CPUTransformTensorBilinear.h"
#include "backend/cpu/CPUBackend.hpp"
#include "tflite_header.h"
#include "cmath"
#include "iostream"

namespace MNN {

inline int Offset(const std::vector<int>& shape, int i0, int i1, int i2, int i3) {
    MNN_ASSERT(shape.size() == 4);
    const int* dims_data = shape.data();
    MNN_ASSERT((dims_data[0] == 0 && i0 == 0) ||
               (i0 >= 0 && i0 < dims_data[0]));
    MNN_ASSERT((dims_data[1] == 0 && i1 == 0) ||
               (i1 >= 0 && i1 < dims_data[1]));
    MNN_ASSERT((dims_data[2] == 0 && i2 == 0) ||
               (i2 >= 0 && i2 < dims_data[2]));
    MNN_ASSERT((dims_data[3] == 0 && i3 == 0) ||
               (i3 >= 0 && i3 < dims_data[3]));
    return ((i0 * dims_data[1] + i1) * dims_data[2] + i2) * dims_data[3] + i3;
}

ErrorCode CPUTransformTensorBilinear::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
{
    auto input0  = inputs[0]; // 1xNxNx32
    auto input1  = inputs[1]; // 1x4x4
    auto output = outputs[0];

    auto input_data_0 = input0->host<float>();
    auto input_data_1 = input1->host<float>();
    auto output_data = output->host<float>();

    MNN_ASSERT(input0->dimensions() == 4);
    MNN_ASSERT(output->dimensions() == 4);

    const int output_height = output->length(1);
    const int output_width = output->length(2);
    const int output_channels = output->length(3);

    const int input_height = input0->length(1);
    const int input_width = input0->length(2);
    const int input_channels = input0->length(3);

    std::vector<int> input_shape_with_batch{/*batch=*/1, input_height,
                                                          input_width, input_channels};
    std::vector<int> output_shape_with_batch{/*batch=*/1, output_height,
                                                           output_width, output_channels};

    // Read first two rows of transformation matrix
    float4 x_transform(input_data_1[0], input_data_1[1],
                                    input_data_1[2], input_data_1[3]);
    float4 y_transform(input_data_1[4], input_data_1[5],
                                    input_data_1[6], input_data_1[7]);

    // Align corners correction: T -> S * ( T * A ), where T is a
    // transformation matrix, and subtruction and addition matrices are:
    // S            A
    // 1 0 0 -0.5   1 0 0 0.5
    // 0 1 0 -0.5   0 1 0 0.5
    // 0 0 1 0      0 0 1 0
    // 0 0 0 1      0 0 0 1
    // Transformation matrix column 3 and rows 3, 4 are identity, which makes
    // the final formula pretty simple and easy to get if doing a manual
    // multiuplication.
    x_transform.w += x_transform.x * 0.5 + x_transform.y * 0.5 - 0.5;
    y_transform.w += y_transform.x * 0.5 + y_transform.y * 0.5 - 0.5;

    for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
            float4 coord(
                    static_cast<float>(out_x), static_cast<float>(out_y),
                    static_cast<float>(0.0), static_cast<float>(1.0));

            // Transformed coordinates.
            float2 tc(DotProduct(x_transform, coord),
                                   DotProduct(y_transform, coord));

            bool out_of_bound = tc.x < 0.0 || tc.x > input_width - 1 || tc.y < 0.0 ||
                                tc.y > input_height - 1;

            for (int out_z = 0; out_z < output_channels; ++out_z) {
                float result = 0;
                if (!out_of_bound) {
                    // Corners position:
                    // q_11 --- q_21
                    // ----     ----
                    // q_12 --- q_22

                    auto ReadValue = [&](int h, int w) -> float {
                        return h < 0 || w < 0 || h >= input_height || w >= input_width
                               ? 0
                               : input_data_0[Offset(input_shape_with_batch, 0, h, w,
                                                     out_z)];
                    };

                    float q_11 = ReadValue(floor(tc.y), floor(tc.x));
                    float q_21 = ReadValue(floor(tc.y), floor(tc.x) + 1);
                    float q_12 = ReadValue(floor(tc.y) + 1, floor(tc.x));
                    float q_22 = ReadValue(floor(tc.y) + 1, floor(tc.x) + 1);

                    float right_contrib = tc.x - floor(tc.x);
                    float lower_contrib = tc.y - floor(tc.y);

                    float upper = (1.0 - right_contrib) * q_11 + right_contrib * q_21;
                    float lower = (1.0 - right_contrib) * q_12 + right_contrib * q_22;

                    result = lower_contrib * lower + (1.0 - lower_contrib) * upper;
                }

                const int out_offset =
                        Offset(output_shape_with_batch, 0, out_y, out_x, out_z);

                output_data[out_offset] = result;
            }
        }
    }
    return NO_ERROR;
}

class CPUTransformTensorBilinearCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs,
                                const MNN::Op *op,
                                Backend *backend) const override {
        return new CPUTransformTensorBilinear(backend);
    }
};
REGISTER_CPU_OP_CREATOR(CPUTransformTensorBilinearCreator, OpType_TransformTensorBilinear);
}
