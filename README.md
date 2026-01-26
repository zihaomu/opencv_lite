# OpenCV-lite

这个一个定制版本的opencv，用于适配目前的edge环境的开发，完整的图像处理功能+敏感的ai推理耗时。

# 为什么还需要OpenCV-lite？
在我的众多基于ai的edge场景开发中，只要涉及到图像识别，opencv库是必不可少的，但是完整的opencv包含众多模块，而他们大多数模块是没有被使用到的。另一个需求是没有被满足的dnn模块，而在edge这种场景中，dnn的需求是性能和功耗极度敏感,原始opencv dnn目前没有充分满足这种需求，我的解决方案是保留opencv dnn的api，而适配多种不同的推理引擎来满足不同场景的需求。让熟悉opencv的用户能够在没有学习难度的情况下，获得更好的dnn推理效果。

# 本仓库做了哪些工作：
1. 裁剪opencv，扩展底层数据类型，增加fp16， int64， bool类型,够将OpenCV的Mat完整对齐ONNXRuntime，TFlite和MNN推理框架；
2. 删除opencv dnn自有推理引擎，对接第三方推理引擎，现在已支持：
    1. onnxruntime，CPU和CUDA GPU，在互联网上存在着大量的onnx模型，只有onnxruntime对这些模型有着最好的兼容性
    2. MNN，CPU，MNN针对arm平台有着极致优化，目前edge设备主要是arm为主。
    3. tflite，OpenCL后端极致优化，特别是针对android手机平台的GPU OpenCL后端，目前是最好的方案。

如何使用这个库：


TODO:
- Fix OpenCV-Python api
- Fix mnn quant error at m1
- Fix ORT backend issue cases at '/doc/dnn_failed_test_case.txt'



## ONNXRuntime 后端
由于onnx的算子在不同版本定义不一样，导致OpenCV 原始DNN一直保受onnx模型兼容性的困扰。我们的方案是直接使用ONNXRuntime作为onnx模型的实际推理引擎，所以彻底解决这个问题，能够在不修改opencv dnn现有的代码情况下，无缝替换。下面是opencv官方版本和opencv-lite版本的onnx算子覆盖率。

## The ONNX op test coverage:
| Project                                                             | ONNX op coverage (%)                             |
|---------------------------------------------------------------------|--------------------------------------------------|
| [OpenCV Official](https://github.com/opencv/opencv/tree/4.x/modules/dnn) | 30.22%**                                         |
| OpenCV-ORT                                                          | **91.69%***                                      |
| [OpenCV Lite](https://github.com/microsoft/onnxruntime)     | [**92.22%**](http://onnx.ai/backend-scoreboard/) |


**: Statistical methods:

([All_test](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance.cpp#L33) - [all_denylist](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance_layer_filter_opencv_all_denylist.inl.hpp) - [parser_denylist](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance_layer_parser_denylist.inl.hpp))/[All_test](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance.cpp#L33)
= (867 - 56 - 549)/867 = 30.2%

*: the unsupported test case can be found [here](https://github.com/zihaomu/opencv_ort/blob/main/modules/dnn/test/test_onnx_conformance_denylist.inl.hpp).

对于不兼容的onnx模型，使用官方版本opencv运行会遇到如下错误：

It's a headache, user always encounter such error:
```angular2html
[ERROR:0@0.357] global onnx_importer.cpp:xxxx cv::dnn::xxxx::ONNXImporter::handleNode ...
```

## How to use opencv-lite with ONNXRuntime？

### Step1: Download ONNXRuntime binary package and unzip.
Please choose it by your platform.
https://github.com/microsoft/onnxruntime/releases

目前opencv-lite已经兼容从：1.14版本到1.22版本的ONNXRuntime的release版本。

### Step2: Set enviroment path
The keywork of `ORT_SDK` will be used in the OpenCV compiler.
```bash
export ORT_SDK=/opt/onnxruntime-osx-arm64-1.14.1 # Fix the ORT_SDK path.
```
### Step3: Compile OpenCV_ORT from source code.
The compilation process is same from original OpenCV project.
And only difference is that we need to set the one PATH:**`ORT`**, so that `cmake` can find `ONNXRuntime lib file` and `ONNXRuntime head file` correctly.

```bash
git clone https://github.com/zihaomu/opencv_ort.git
cd opencv_ort
mkdir build & cd build
cmake -D ORT_SDK=/opt/onnxruntime-osx-arm64-1.14.1 .. # Fix the ORT_SDK path.
```

# How to use it?
The code is the totally same as original OpenCV DNN.
```C++
#include <iostream>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<algorithm>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main()
{
    // load input
    Mat image = imread("PATH_TO_image");
    Scalar meanValue(0.485, 0.456, 0.406);
    Scalar stdValue(0.229, 0.224, 0.225);

    Mat blob = blobFromImage(image, 1.0/255.0, Size(224, 224), meanValue, true);
    blob /= stdValue;
    
    Net net = readNet("PATH_TO_MODEL/resnet50-v1-12.onnx");

    std::vector<Mat> out;
    net.setInput(blob);
    net.forward(out);
    
    double min=0, max=0;
    Point minLoc, maxLoc;
    minMaxLoc(out[0], &min, &max, &minLoc, &maxLoc);
    cout<<"class num = "<<maxLoc.x<<std::endl;
}
```

## How to use opencv-lite with MNN？

目前，我们将MNN集成到我们的opencv-lite的dnn中，作为我们的第三方库一起编译，具体路径在`./3rdparty/MNN`，所以无需手动加入，当然你也可以像加入上面加入ONNXRuntime后端一样，加入你的自己版本的MNN。我们已经针对MNN的2.8版本进行二次开发，主要是扩展了mediapipe中一些不支持算子，使得这个版本的MNN能够直接推理MNN模型，这个定制版本MNN也已经用在[mediapipe_cmake](https://github.com/zihaomu/mediapipe_cmake)中。

而mnn的使用代码和上面几乎一样，唯一的变化是，住备好mnn的模型，然后从

```C++
Net net = readNet("PATH_TO_MODEL/resnet50-v1-12.onnx");
```
变成
```C++
Net net = readNet("PATH_TO_MODEL/resnet50-v1-12.mnn");
```
因为内部使用的是文件后缀去判断具体使用哪个后端。

## How to use opencv-lite with tflite？
同样，针对tflite我们也在编译opencv-lite中加入一些编译选项。


## TODO List
1. Fix some bugs in imgproc.
2. Add Github Action.
3. video demo.
4. Add ORT-CUDA support, and compatible with `net.setPreferableBackend(DNN_BACKEND_CUDA)` API.
