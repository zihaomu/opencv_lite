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

#ifdef HAVE_TFLITE
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>

#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#ifdef HAVE_TRT
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::make_pair;
using std::string;

class Net::Impl
{
public:
    Impl();
    virtual ~Impl();
    Impl(const Impl&) = delete;

    virtual bool empty() const;

    virtual void setNumThreads(int num);
    virtual int getNumThreads();
    virtual void setPreferableBackend(Backend device);        // setting computing device
    virtual void setPreferablePrecision(Precision precision); // setting computing accuracy
    virtual void setConfig(NetConfig* config);              // setting model config.
    // setting input shape.
    virtual void setInputShape(const cv::String &inputName, const cv::dnn::MatShape &shape);

    virtual Mat forward(const String& outputName);

    virtual void forward(const OutputArrayOfArrays &outputBlobs, const String &outputName);

    // Contain the empty implementation
    virtual void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames) = 0;

    virtual void setType(ModelType type);

    virtual ModelType getType();

    // According to the model name to load model with specific Impl Runner.
    // Contain the empty implementation
    virtual void readNet(const String& model) = 0;
    virtual void readNet(const char* buffer, size_t sizeBuffer);
    virtual void setInput(InputArray blob_, const String& name) = 0;

    std::vector<string> getInputName();
    std::vector<MatShape> getInputShape();
    std::vector<string> getOutputName();
    std::vector<MatShape> getOutputShape();

protected:
    // this function will return the index of inputNamesString with given input name.
    virtual int getInputIndex(const String &name);

    // for output
    virtual int getOutputIndex(const String& name);

    // TODO use netConfig instead of single params
    NetConfig netConfig;
    int thread_num = 4;  // default thread number
    int inputCount = 0;
    int outputCount = 0;
    ModelType type = ModelType::DNN_TYPE_UNKNOW;
    Backend device = Backend::DNN_BACKEND_CPU;
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
namespace dnn_trt {
class ImplTensorRT : public Net::Impl
{
public:
    ImplTensorRT();
    virtual ~ImplTensorRT();
    void setNumThreads(int num) override;
    void readNet(const String& model) override;
    void setInput(InputArray blob_, const String& name) override;
    void setConfig(NetConfig* config) override;
    void forward(OutputArrayOfArrays outputBlobs,
        const std::vector<String>& outBlobNames) override;
private:

    // Allocate Host memory and binding to the engine.
    void allocMem();
    void tensors2Mats(const std::vector<int>& outputIdxs, std::vector<Mat>& outputMat);

    NetConfig_TRT configTRT;
    std::vector<int> input_idxs;
    std::vector<int> output_idxs;

    //<<<<<<<<<<<<<<<<<< TensorRT resource  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    std::vector<::nvinfer1::Dims> inputMatShapeTrt;                      //!< The dimensions of the input to the network.
    std::vector<::nvinfer1::Dims> outputMatShapeTrt;                      //!< The dimensions of the input to the network.

    // The following two buffer list contains both input/output. It orgnizes as input_idxs/output_idxs
    std::vector<std::pair<AutoBuffer<uchar>, size_t>> bufferListHost; // pointer and size (can be overwritten by user)
    std::vector<void *> bufferListDevice;                  // pointer to GPU memory

    Ptr<::nvinfer1::IRuntime> runtime_;
    Ptr<::nvinfer1::ICudaEngine> engine_;
    Ptr<::nvinfer1::IExecutionContext> context_;

    Ptr<::nvinfer1::IBuilder> builder_;
    Ptr<::nvinfer1::INetworkDefinition> network_;
    Ptr<::nvinfer1::IBuilderConfig> config_;
    const bool verboseLog = false;

    //<<<<<<<<<<<<<<<<<< CUDA resource  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    cudaStream_t stream_ = nullptr;
    int device_id_;
    std::string compute_capability_;

    cv::Mutex mutex;
};
}
#endif

#ifdef HAVE_MNN
namespace dnn_mnn {
class ImplMNN : public Net::Impl {
public:
    ImplMNN();

    ~ImplMNN();

    void setNumThreads(int num) override;

    void setPreferableBackend(Backend device) override;        // setting computing device
    void setPreferablePrecision(Precision precision) override; // setting computing accuracy
    void readNet(const String &model) override;

    void readNet(const char *buffer, size_t sizeBuffer) override;

    void setInput(InputArray blob_, const String &name) override;

    void setInputShape(const cv::String &inputName, const cv::dnn::MatShape &shape) override;

    void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String> &outBlobNames) override;
private:
    void parseTensorInfoFromSession();

    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig = {};
    MNN::Interpreter *netPtr = nullptr;
    MNN::Session *session = nullptr;
    std::vector<MNN::Tensor *> inTensorsPtr;
    std::vector<MNN::Tensor *> outTensorsPtr;
};
}
#endif

#ifdef HAVE_TFLITE
namespace dnn_tflite
{
class ImplTflite : public Net::Impl
{
public:
    ImplTflite();
    virtual ~ImplTflite();

    void setNumThreads(int num) override;
    void readNet(const String& model) override;
    void readNet(const char* buffer, size_t sizeBuffer) override;
    void setInput(InputArray blob_, const String& name) override;
    void setPreferableBackend(Backend device) override;
    void setPreferablePrecision(Precision precision) override;
    void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames) override;

    void setInputShape(const cv::String &inputName, const cv::dnn::MatShape &shape) override;

private:
    // do some model initialization, get input/output tensor info
    // prepare tf-lite resource
    void init();

    void parseInOutTensor();

    std::unique_ptr<tflite::FlatBufferModel> modelTF = nullptr;
    tflite::ops::builtin::BuiltinOpResolver resolveTF;
    std::unique_ptr<tflite::Interpreter> interpreterTF = nullptr;

    TfLiteDelegate* delegate_ = nullptr;
};

}
#endif

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // __OPENCV_DNN_SRC_NET_IMPL_HPP__
