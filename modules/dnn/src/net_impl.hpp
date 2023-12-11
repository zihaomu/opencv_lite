// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_SRC_NET_IMPL_HPP__
#define __OPENCV_DNN_SRC_NET_IMPL_HPP__

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_MNN
#include "MNN/Interpreter.hpp"
#endif

#ifdef HAVE_ORT
#include "onnxruntime_cxx_api.h"
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::make_pair;
using std::string;

// DNN compute precision
enum Compute_Precision
{
    DNN_PRECISION_LOW = 0,
    DNN_PRECISION_NORMAL = 1,
    DNN_PRECISION_HIGH = 2
};

enum Compute_Device
{
    DNN_DEVICE_CPU = 0,
    DNN_DEVICE_GPU = 1,
};

class Net::Impl
{
public:
    Impl();
    virtual ~Impl();
    Impl(const Impl&) = delete;

    virtual bool empty() const;

    virtual void setNumThreads(int num);
    virtual void setPrecision(Compute_Precision precision); // setting computing accuracy
    virtual void setPreferDevice(Compute_Device device);    // 设置设备，紧支持GPU和CPU

    virtual Mat forward(const String& outputName);

    virtual void forward(const OutputArrayOfArrays &outputBlobs, const String &outputName);

    // Contain the empty implementation
    virtual void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames) = 0;

    // According to the model name to load model with specific Impl Runner.
    // Contain the empty implementation
    virtual void readNet(const String& model) = 0;
    virtual void setInput(InputArray blob_, const String& name) = 0;

    std::vector<string> getInputName();
    std::vector<MatShape> getInputShape();
    std::vector<string> getOutputName();
    std::vector<MatShape> getOutputShape();

protected:
    int thread_num = 1;
    int inputCount = 0;
    int outputCount = 0;
    std::vector<std::string> inputNamesString; // reserve model input name.
    std::vector<std::string> outputNamesString;

    std::vector<MatShape> inputMatShape;
    std::vector<MatShape> outputMatShape;
};

// The implement class for ONNXRuntime backend.
#ifdef HAVE_ORT
class ImplORT : public Net::Impl
{
public:
    ImplORT();
    virtual ~ImplORT();

    void setNumThreads(int num) override;
    void readNet(const String& model) override;
    void setInput(InputArray blob_, const String& name) override;
    void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames) override;

private:
    std::vector<const char*> inputNames; // for saving the setInput.
    std::vector<std::vector<int64_t> > inputInt64; // use for create Ort::Tensor
    std::vector<std::vector<int64_t> > outputInt64; // use for create Ort::Tensor

    String instanceName{"opencv-dnn-inference"};
    std::vector<ONNXTensorElementDataType> inDataType;
    std::vector<ONNXTensorElementDataType> outDataType;
    Ort::Env env;

    std::vector<Ort::Value> inputTensors;
    Ort::AllocatorWithDefaultOptions allocator;
    // Currently, we only support the ORT CPU backend. TODO! support ORT cuda, and other backend.
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::SessionOptions sessionOptions;// = Ort::SessionOptions{nullptr};
    Ort::Session session = Ort::Session{nullptr};
};
#endif

#ifdef HAVE_TRT
class ImplTensorRT : public Net::Impl
{
public:
    ImplTensorRT();
    virtual ~ImplTensorRT();
    void setNumThreads(int num) override;
    void readNet(const String& model) override;
    void setInput(InputArray blob_, const String& name) override;
    void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames) override;
private:
};
#endif

#ifdef HAVE_MNN
class ImplMNN : public Net::Impl
{
public:
    ImplMNN();
    ~ImplMNN();
    void setNumThreads(int num) override;
    void readNet(const String& model) override;
    void setInput(InputArray blob_, const String& name) override;
    void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames) override;

private:
    void parseTensorInfoFromSession();
    MNN::ScheduleConfig config;
    MNN::Interpreter *netPtr = nullptr;
    MNN::Session *session = nullptr;
    std::vector<MNN::Tensor*> inTensorsPtr;
    std::vector<MNN::Tensor*> outTensorsPtr;
};
#endif

// NB: Implementation is divided between of multiple .cpp files

// TODO, register backend.
// run net work for specific backend.
//struct Net::Impl2
//{
//    virtual ~Impl();
//    Impl();
//    Impl(const Impl&) = delete;
//
//    bool fusion;
//    bool useWinograd;
//
//    virtual bool empty() const;
////    virtual void setPreferableBackend(Net& net, int backendId);
////    virtual void setPreferableTarget(int targetId);
////    virtual void clear();
//
//    Mat forward(const String& outputName);
//
//    void forward(const OutputArrayOfArrays &outputBlobs, const String &outputName);
//
//    void forward(OutputArrayOfArrays outputBlobs,
//                 const std::vector<String>& outBlobNames);
//
//    void setNumThreads(int num);
//
//    void readNet(const String& model);
//    void setInput(InputArray blob_, const String& name);
//
//    std::vector<string> getInputName();
//    std::vector<MatShape> getInputShape();
//    std::vector<string> getOutputName();
//    std::vector<MatShape> getOutputShape();
//
//    int thread_num = getNumThreads();
//
//    // ONNXRuntime info
//    String instanceName{"opencv-dnn-inference"};
//    int inputCount = 0;
//    int outputCount = 0;
//
//    std::vector<std::string> inputNamesString; // reserve the session input name.
//    std::vector<std::string> outputNamesString;
//
//    std::vector<const char*> inputNames; // for saving the setInput.
//#ifdef HAVE_ORT
//    std::vector<Ort::Value> inputTensors;
//#endif
//
//    // To save the memory
//    std::vector<const char*> preOutputName; // used for check if the
//    std::vector<Mat> preOuts;
//    std::vector<AutoBuffer<char> > buffers;
//
//
//    std::vector<std::vector<int64_t> > inputInt64; // use for create Ort::Tensor
//    std::vector<MatShape> inputMatShape;
//    std::vector<std::vector<int64_t> > outputInt64; // use for create Ort::Tensor
//    std::vector<MatShape> outputMatShape;           // use for create OpenCV Mat.
//
//#ifdef HAVE_ORT
//    std::vector<ONNXTensorElementDataType> inDataType;
//    std::vector<ONNXTensorElementDataType> outDataType;
//    Ort::Env env;
//
//    Ort::AllocatorWithDefaultOptions allocator;
//    // Currently, we only support the ORT CPU backend. TODO! support ORT cuda, and other backend.
//    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
//            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//    Ort::SessionOptions sessionOptions;// = Ort::SessionOptions{nullptr};
//    Ort::Session session = Ort::Session{nullptr};
//#endif
//
//};  // Net::Impl


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // __OPENCV_DNN_SRC_NET_IMPL_HPP__
