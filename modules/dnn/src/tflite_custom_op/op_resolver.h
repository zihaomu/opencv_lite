//
// Created by mzh on 2024/3/5.
//

#ifndef OPENCV_OP_RESOLVER_H
#define OPENCV_OP_RESOLVER_H

#include "../precomp.hpp"

#ifdef HAVE_TFLITE
#include "tensorflow/lite/kernels/register.h"

namespace cv {
namespace dnn {
namespace mp_op {

class MediaPipeBuiltinOpResolver
        : public tflite::ops::builtin::BuiltinOpResolver {
public:
    MediaPipeBuiltinOpResolver();
    MediaPipeBuiltinOpResolver(const MediaPipeBuiltinOpResolver& r) = delete;
};

}}}

#endif
#endif //OPENCV_OP_RESOLVER_H
