//
// Created by mzh on 2023/7/12.
//

#ifndef MNN_CPULANDMARKS2TRANSFORMMATRIX_H
#define MNN_CPULANDMARKS2TRANSFORMMATRIX_H

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {

class CPULandmarks2TransformMatrix : public Execution {
public:
    CPULandmarks2TransformMatrix(const MNN::Op* op, Backend *b);

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) override;

private:
    int32_t left_rotation_idx;
    int32_t output_height;
    int32_t output_width;
    int32_t right_rotation_idx;
    float scale_x;
    float scale_y;
    int target_rotation_radians;
    std::vector<int32_t> subset_idxs;
};

} // namespace MNN

#endif //MNN_CPULANDMARKS2TRANSFORMMATRIX_H
