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
class TransformLandmarksComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto output    = outputs[0];
        auto input0     = inputs[0];
        auto input1     = inputs[1];

        output->buffer().dimensions    = input0->buffer().dimensions;
        MNN_ASSERT(3 == input0->buffer().dimensions);

        output->buffer().type = input0->buffer().type;
        TensorUtils::copyShape(input0, output, true);

        output->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(TransformLandmarksComputer, OpType_TransformLandmarks);
} // namespace MNN
