//
// Created by mzh on 2023/7/12.
//

#include "CPUTransformLandmarks.h"
#include "backend/cpu/CPUBackend.hpp"
#include "tflite_header.h"
#include "iostream"

namespace MNN {
//inline const int32_t* DimsDataUpTo5D() const { return dims_; }

// Since tensors with '0' in their shape are valid in TF, these offset functions
// allow that as long as the corresponding index is also 0. It is upto the
// calling ops to ensure that they perform verification checks on tensor shapes
// if they don't support a particular behavior.

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

// Version2
ErrorCode CPUTransformLandmarks::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
{
    MNN_ASSERT(inputs.size() == 2);

    auto input0  = inputs[0]; // 1xNx2
    auto input1  = inputs[1]; // 1x4x4
    auto output = outputs[0];

    auto landmarks = input0->host<float>();
    auto transform_matrix = input1->host<float>();
    auto output_data = output->host<float>();


    std::vector<int> input0Shape(3, 0);
    for (int i = 0; i < 3; i++)
    {
        input0Shape[i] = input0->length(i);
    }

    const int output_width = input0->length(1);

    // Read first two rows of transformation matrix
    float4 x_transform(transform_matrix[0], transform_matrix[1],
                                    transform_matrix[2], transform_matrix[3]);
    float4 y_transform(transform_matrix[4], transform_matrix[5],
                                    transform_matrix[6], transform_matrix[7]);

    std::vector<int> input_shape_with_batch = {1, input0Shape[0], input0Shape[1], input0Shape[2]};

#if MNN_DEBUG
    std::cout<<"print intput on TransformLandmarks, input shape "<<std::endl;
    for (int i = 0; i < output_width; i++)
    {
        std::cout<<","<<landmarks[i * 2]<<" x "<<landmarks[i * 2 + 1]<<std::endl;
    }

//    std::cout<<"print intput on TransformLandmarks, input shape "<<std::endl;
//    for (int i = 0; i < output_width; i++)
//    {
//        std::cout<<","<<landmarks[i * 2]<<" x "<<landmarks[i * 2 + 1]<<std::endl;
//    }
#endif

    const int params_dimensions = 2;

    for (int landmark = 0; landmark < output_width; ++landmark)
    {
        const int offset = Offset(input_shape_with_batch, 0, 0, landmark, 0);

        if (params_dimensions == 2) {
            float4 lv(landmarks[offset], landmarks[offset + 1], static_cast<float>(0.0), static_cast<float>(1.0));
            float2 transformed(DotProduct(x_transform, lv),
                                            DotProduct(y_transform, lv));
            output_data[offset] = transformed.x;
            output_data[offset + 1] = transformed.y;
        }
        // Not used right now.
//        if (params.dimensions == 3) {
//            tflite::gpu::float4 lv(landmarks[offset], landmarks[offset + 1],
//                                   static_cast<float>(0.0), static_cast<float>(1.0));
//            tflite::gpu::float3 transformed(DotProduct(x_transform, lv),
//                                            DotProduct(y_transform, lv), lv.z);
//            output_data[offset] = transformed.x;
//            output_data[offset + 1] = transformed.y;
//            output_data[offset + 2] = landmarks[offset + 2];
//        }
    }
#if MNN_DEBUG
    std::cout<<"print output on TransformLandmarks, input shape "<<input_shape_with_batch[1]<<"x"<<input_shape_with_batch[2]<<"x"<<input_shape_with_batch[3]<<std::endl;
    for (int i = 0; i < output_width; i++)
    {
        std::cout<<","<<output_data[i * 2]<<" x "<<output_data[i * 2 + 1]<<std::endl;
    }
#endif
    return NO_ERROR;
}

class CPUTransformLandmarksCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs,
                                const MNN::Op *op,
                                Backend *backend) const override {
        return new CPUTransformLandmarks(backend);
    }
};
REGISTER_CPU_OP_CREATOR(CPUTransformLandmarksCreator, OpType_TransformLandmarks);
}
