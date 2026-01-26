# OpenCV-lite

A customized, high-performance OpenCV distribution optimized for edge computing and AI deployment.

## Overview

**OpenCV-lite** is a tailored version of OpenCV designed for edge scenarios. It provides the full suite of essential image processing functions while significantly enhancing AI inference performance and compatibility. By pruning unused modules and integrating specialized inference engines, OpenCV-lite offers a "drop-in" replacement for standard OpenCV with superior DNN capabilities.

## Why OpenCV-lite?

In edge AI development, OpenCV is indispensable for image handling, but the standard library has two main drawbacks:
1.  **Bloat**: It includes many modules rarely used in edge applications.
2.  **DNN Limitations**: The native DNN module often lacks the performance, power efficiency, and operator coverage required for production edge devices.

**Our Solution**: Maintain the familiar OpenCV API while swapping the underlying DNN engine for specialized frameworks like **ONNXRuntime**, **MNN**, and **TFLite**. This allows developers to gain high-performance inference without learning new APIs.

## Core Features

1.  **Module Pruning**: Significantly reduced library size by removing non-essential components.
2.  **Extended Data Types**: Added support for `FP16`, `INT64`, and `BOOL` in `cv::Mat`, ensuring perfect alignment with modern inference frameworks.
3.  **Unified Backend API**:
    *   **ONNXRuntime**: Best-in-class compatibility for ONNX models (CPU/CUDA).
    *   **MNN**: Extreme optimization for ARM-based edge devices.
    *   **TFLite**: Highly optimized for mobile GPUs via OpenCL.
4.  **Automatic Backend Selection**: The library automatically selects the appropriate backend based on the model file extension (`.onnx`, `.mnn`, `.tflite`).

---

## Performance: ONNX Operator Coverage

OpenCV-lite (via ONNXRuntime) solves the "unsupported operator" headache commonly found in standard OpenCV.

| Project | ONNX Operator Coverage (%) |
| :--- | :--- |
| [OpenCV Official (4.x)](https://github.com/opencv/opencv) | ~30.22% |
| **OpenCV-lite** | **>92%** |

*Coverage calculated based on ONNX conformance tests. OpenCV-lite provides a significantly more robust experience for modern AI models.*

---

## Installation & Setup

### 1. ONNXRuntime Backend
Download the [ONNXRuntime releases](https://github.com/microsoft/onnxruntime/releases) (v1.14 to v1.22 supported) and set the path:

```bash
export ORT_SDK=/path/to/onnxruntime
```

### 2. Build from Source
The build process is identical to standard OpenCV, with the addition of the `ORT_SDK` flag:

```bash
git clone https://github.com/zihaomu/opencv_ort.git
cd opencv_ort && mkdir build && cd build
cmake -D ORT_SDK=$ORT_SDK ..
make -j$(nproc)
```

---

## Usage Example

The API remains 100% compatible with standard OpenCV DNN. You only need to provide the model file; OpenCV-lite handles the rest.

```cpp
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    
    // Pre-processing
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0/255.0, cv::Size(224, 224), 
                                         cv::Scalar(0.485, 0.456, 0.406), true);
    
    // Load Model (Backend automatically selected via extension)
    // .onnx -> ONNXRuntime
    // .mnn  -> MNN
    // .tflite -> TFLite
    cv::dnn::Net net = cv::dnn::readNet("resnet50.onnx");

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs);
    
    // Post-processing...
}
```

## Special Backend Notes

*   **MNN**: Integrated in `3rdparty/MNN`. We use a customized MNN 2.8+ with extended operators for [mediapipe_cmake](https://github.com/zihaomu/mediapipe_cmake).
*   **TFLite**: Requires pre-compiled TFLite headers/libs. For Android OpenCL optimization, see [tflite_cmake](https://github.com/zihaomu/tflite_cmake).

## Roadmap / TODO
- [ ] Fix OpenCV-Python API bindings.
- [ ] Resolve MNN quantization errors on Apple M1.
- [ ] Improve ORT backend edge case handling (see `doc/dnn_failed_test_case.txt`).
- [ ] General bug fixes in `imgproc`.
- [ ] Integrate GitHub Actions for CI/CD.

---

## Related Projects
* [mediapipe_cmake](https://github.com/zihaomu/mediapipe_cmake) - Real-world deployment using OpenCV-lite backends.

## Testing Information
I have manually tested **v4.13** on the following platforms:
- Mac (M2 & Intel)
- Windows (Intel)
- Ubuntu (Intel)
- Android (NDK r26)

Test data can be found at: [opencv_extra](https://github.com/zihaomu/opencv_extra)

## Acknowledgments
- [OpenCV Official](https://github.com/opencv/opencv) - The foundation of this project.
- [ONNXRuntime](https://github.com/microsoft/onnxruntime)
- [MNN](https://github.com/alibaba/MNN)
- [TFLite](https://github.com/tensorflow/tensorflow)
