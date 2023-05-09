// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_SRC_NET_IMPL_HPP__
#define __OPENCV_DNN_SRC_NET_IMPL_HPP__

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <opencv2/core/utils/logger.hpp>

#include "onnxruntime_cxx_api.h"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::make_pair;
using std::string;

// NB: Implementation is divided between of multiple .cpp files

// TODO, register backend.
// run net work for specific backend.
struct Net::Impl
{
    virtual ~Impl();
    Impl();
    Impl(const Impl&) = delete;

    bool fusion;
    bool useWinograd;

    virtual bool empty() const;
//    virtual void setPreferableBackend(Net& net, int backendId);
//    virtual void setPreferableTarget(int targetId);
//    virtual void clear();

    Mat forward(const String& outputName);

    void forward(const OutputArrayOfArrays &outputBlobs, const String &outputName);

    void forward(OutputArrayOfArrays outputBlobs,
                 const std::vector<String>& outBlobNames);

    void setNumThreads(int num);

    void readNet(const String& model);
    void setInput(InputArray blob_, const String& name);

    std::vector<string> getInputName();
    std::vector<MatShape> getInputShape();
    std::vector<string> getOutputName();
    std::vector<MatShape> getOutputShape();

    int thread_num = getNumThreads();

    // ONNXRuntime info
    String instanceName{"opencv-dnn-inference"};
    int inputCount = 0;
    int outputCount = 0;

    std::vector<std::string> inputNamesString; // reserve the session input name.
    std::vector<std::string> outputNamesString;

    std::vector<const char*> inputNames; // for saving the setInput.
    std::vector<Ort::Value> inputTensors;

    // To save the memory
    std::vector<const char*> preOutputName; // used for check if the
    std::vector<Mat> preOuts;
    std::vector<AutoBuffer<char> > buffers;


    std::vector<std::vector<int64_t> > inputInt64; // use for create Ort::Tensor
    std::vector<MatShape> inputMatShape;
    std::vector<std::vector<int64_t> > outputInt64; // use for create Ort::Tensor
    std::vector<MatShape> outputMatShape;           // use for create OpenCV Mat.
    std::vector<ONNXTensorElementDataType> inDataType;
    std::vector<ONNXTensorElementDataType> outDataType;
    Ort::Env env;

    Ort::AllocatorWithDefaultOptions allocator;
    // Currently, we only support the ORT CPU backend. TODO! support ORT cuda, and other backend.
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::SessionOptions sessionOptions;// = Ort::SessionOptions{nullptr};
    Ort::Session session = Ort::Session{nullptr};

};  // Net::Impl


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // __OPENCV_DNN_SRC_NET_IMPL_HPP__
