//
//  ShapeInnerProduct.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class TransformTensorBilinearComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto output    = outputs[0];
        auto input0     = inputs[0];
        auto input1     = inputs[1];

        output->buffer().dimensions    = 4;
        MNN_ASSERT(4 == input0->buffer().dimensions);
        MNN_ASSERT(3 == input1->buffer().dimensions);

        output->buffer().dim[0].extent = input0->buffer().dim[0].extent;
        output->buffer().dim[3].extent = input0->buffer().dim[3].extent;


        output->buffer().dim[1].extent = 16;
        output->buffer().dim[2].extent = 16;

        output->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(TransformTensorBilinearComputer, OpType_TransformTensorBilinear);
} // namespace MNN
