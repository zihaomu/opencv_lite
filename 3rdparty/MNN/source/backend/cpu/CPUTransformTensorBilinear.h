//
// Created by mzh on 2023/7/12.
//

#ifndef MNN_CPUTransformTensorBilinear_H
#define MNN_CPUTransformTensorBilinear_H

#include "core/Execution.hpp"

namespace MNN {

class CPUTransformTensorBilinear : public Execution {
public:
    CPUTransformTensorBilinear(Backend *b) : Execution(b) {
            // nothing to do
    }

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) override;

};

} // namespace MNN

#endif //MNN_CPUTransformTensorBilinear_H
