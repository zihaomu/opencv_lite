//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "op_resolver.h"
#include "op_register.h"
#ifdef HAVE_TFLITE
#include "tensorflow/lite/kernels/register.h"

namespace cv {
namespace dnn {
namespace mp_op {

MediaPipeBuiltinOpResolver::MediaPipeBuiltinOpResolver()
{
    AddCustom("MaxPoolingWithArgmax2D", RegisterMaxPoolingWithArgmax2D());
    AddCustom("MaxUnpooling2D", RegisterMaxUnpooling2D());
    AddCustom("Convolution2DTransposeBias", RegisterConvolution2DTransposeBias());
    AddCustom("TransformTensorBilinear", RegisterTransformTensorBilinearV2(),
            /*version=*/2);
    AddCustom("TransformLandmarks", RegisterTransformLandmarksV2(),
            /*version=*/2);
    AddCustom(
            "Landmarks2TransformMatrix", RegisterLandmarksToTransformMatrixV2(),
            /*version=*/2);
}

}}}

#endif