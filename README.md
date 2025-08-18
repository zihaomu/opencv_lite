# OpenCV-lite

一个轻量版的opencv，专注dnn模型部署场景，并融入mediapipe主要功能，方便用户在各个平台移植。
本仓库做了哪些工作：
1. 裁剪opencv，扩展数据类型，增加fp16， int64， bool类型；
2. 删除opencv dnn自有推理引擎，对接第三方推理引擎，现在已支持：
    1. onnxruntime，CPU和CUDA GPU
    2. MNN，CPU
    3. tflite，CPU和安卓GPU
 3. 全套的mediapipe 移植。mediapipe是很棒的项目。

逐一回答为什么要实现以上几点：
1. 为什么要裁剪opencv？
在我从事ai模型部署以来，opencv在这个领域模块的使用频率有限，但是多余的模块会极大的增加编译时间。有些模块引入了下载链接，这也给网络有问题的用户带来了困扰。

TODO，给出opencv 完整版本编译时间

2. 为什么要删除原有dnn模块，而引入第三方推理引擎？
过去三年时间，所接触到的部署平台包括：服务端，pc端和移动端。我发现没有一个推理引擎能够在所有平台所有情况获得最优解。
MNN作为通用推理框架，基本能够做到在：服务端，pc端和移动端上均衡的性能，不过有个大问题，就是需要将模型格式进行转换，有些模型格式无法转换，就嗝屁了。开发mnn算子的工程量太大，需要了解不少mnn库的知识。
这时候，加入onnxruntime的支持，也是一个很好的补充，windows，linux，mac都有预编译好的库，用起来很方便。
为什么需要tflite？在安卓平台上，使用一些比较宽的模型时，使用gpu推理是必须的，目前测试了，只有tflite的opencl速度是最好的，tflite的cpu是被mnn按在地上摩擦的。

tflite，onnxruntime以及mnn的调用api都不算太复杂，不过没学过的同学还是需要一些学习成本的，为了统一一下调用接口，这里将他们映射到了opencv dnn的api中。

3. 为什么需要移植一套mediapipe？
mediapipe是很棒的项目，但是官方目前的没有想让这个库编程跨平台的想法，而想把这个库做成安卓独占的存在，其他平台的用户想要用mediapipe，是很困难的。mediapipe也算是比较复杂的能力了，正好作为本仓库的一个附加模块。增加一些测试样例。


## 

1. The API of opencv is easy to use, but the compatibility with ONNX model is poor.  
2. ONNXRuntime is very compatible with ONNX, but the API is hard to use and changes all the time.

> The compatibility with ONNX model is poor.

It's a headache, user always encounter such error:
```angular2html
[ERROR:0@0.357] global onnx_importer.cpp:xxxx cv::dnn::xxxx::ONNXImporter::handleNode ...
```
In view of the fact that OpenCV DNN does not fully support dynamic shape input and has low coverage for ONNX. That means user may either in `readNet()`, or in `net.forward()`, always get an error.
It is expected that after the release of OpenCV 5.0, things will improve.

**If you have a model that needs to be inferred and deployed in a C++ environment, and you encounter errors above, maybe you can try this library.**

In this project, I removed all dnn implementation codes, only kept the dnn's API. 
And connected to the C++ API of ONNXRuntime.

### The ONNX op test coverage:
| Project                                                             | ONNX op coverage (%)                             |
|---------------------------------------------------------------------|--------------------------------------------------|
| [OpenCV DNN](https://github.com/opencv/opencv/tree/4.x/modules/dnn) | 30.22%**                                         |
| OpenCV-ORT                                                          | **91.69%***                                      |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime)     | [**92.22%**](http://onnx.ai/backend-scoreboard/) |

**: Statistical methods:

([All_test](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance.cpp#L33) - [all_denylist](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance_layer_filter_opencv_all_denylist.inl.hpp) - [parser_denylist](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance_layer_parser_denylist.inl.hpp))/[All_test](https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_conformance.cpp#L33)
= (867 - 56 - 549)/867 = 30.2%

*: the unsupported test case can be found [here](https://github.com/zihaomu/opencv_ort/blob/main/modules/dnn/test/test_onnx_conformance_denylist.inl.hpp).

## TODO List
1. Fix some bugs in imgproc.
2. Add Github Action.
3. video demo.
4. Add ORT-CUDA support, and compatible with `net.setPreferableBackend(DNN_BACKEND_CUDA)` API.

# How to install?

### Step1: Download ONNXRuntime binary package and unzip.
Please choose it by your platform.
https://github.com/microsoft/onnxruntime/releases

I have tested it with ONNXRuntime version: 1.14.1, and it works well.

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
    
    Net net = readNetFromONNX("PATH_TO_MODEL/resnet50-v1-12.onnx");

    std::vector<Mat> out;
    net.setInput(blob);
    net.forward(out);
    
    double min=0, max=0;
    Point minLoc, maxLoc;
    minMaxLoc(out[0], &min, &max, &minLoc, &maxLoc);
    cout<<"class num = "<<maxLoc.x<<std::endl;
}
```
