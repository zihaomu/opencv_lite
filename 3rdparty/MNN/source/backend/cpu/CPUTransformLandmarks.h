//
// Created by mzh on 2023/7/12.
//

#ifndef MNN_CPUTransformLandmarks_H
#define MNN_CPUTransformLandmarks_H

#include "core/Execution.hpp"

namespace MNN {

class CPUTransformLandmarks : public Execution {
public:
    CPUTransformLandmarks(Backend *b) : Execution(b) {
            // nothing to do
    }

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) override;

};

} // namespace MNN

#endif //MNN_CPUTransformLandmarks_H
