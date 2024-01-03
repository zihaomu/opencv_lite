# OpenCV-lite
OpenCV-lite 是一个轻量化版本的opencv，专注dnn模型部署场景。主要的修改包括：
- 移除一些不太常用的模块：features2d, flann, gapi, ml, objdetect, stitching, video.
- 保留dnn模块API，直接使用MNN和ONNXRuntime去做相应的模型推理。

相关项目：

OpenCV-lite相比于原始OpenCV的优势：裁剪掉多余模块，减少包体；引入原生ONNXRuntime推理引擎，减少ONNX模型报错；引入MNN推理引擎，最大化推理效率。


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
