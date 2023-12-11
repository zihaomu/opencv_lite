//
//  ShapeCast.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"

namespace MNN {

class Landmarks2TransformMatrixSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto output = outputs[0];
        auto input  = inputs[0];

        output->buffer().dimensions    = input->buffer().dimensions;
        MNN_ASSERT(3 == input->buffer().dimensions);

        output->buffer().dim[0].extent = input->buffer().dim[0].extent;

        output->buffer().dim[1].extent = 4;
        output->buffer().dim[2].extent = 4;

        output->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};
REGISTER_SHAPE(Landmarks2TransformMatrixSizeComputer, OpType_Landmarks2TransformMatrix);


} // namespace MNN
