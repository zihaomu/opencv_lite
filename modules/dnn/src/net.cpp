// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Net::Net()
{
}

Net::~Net()
{
}

Mat Net::forward(const String& outputName)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forward(outputName);
}

void Net::forward(OutputArrayOfArrays outputBlobs, const String& outputName)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forward(outputBlobs, outputName);
}

void Net::forward(OutputArrayOfArrays outputBlobs, const std::vector<String>& outBlobNames)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forward(outputBlobs, outBlobNames);
}

std::vector<std::string> Net::getInputName()
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getInputName();
}

std::vector<std::string> Net::getOutputName()
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getOutputName();
}

std::vector<MatShape> Net::getInputShape()
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getInputShape();
}

std::vector<MatShape> Net::getOutputShape()
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getOutputShape();
}

void Net::readNet(const char* _framework, const char* buffer, size_t sizeBuffer)
{
    CV_TRACE_FUNCTION();
    String framework = _framework;
#ifdef HAVE_MNN
    if (framework == "mnn")
    {
        impl = makePtr<ImplMNN>();
    } else
#endif
#ifdef HAVE_TRT
    else if (framework == "onnx")
    {
        CV_Error(Error::StsError, "read ONNX from buffer is being developed, please contact the developer.");
    } else
#endif
#ifdef HAVE_ORT
    if (framework == "trt")
    {
        CV_Error(Error::StsError, "read TensorRT from buffer is being developed, please contact the developer.");
    }
    else
#endif
    CV_Error(Error::StsError, "Cannot determine an origin framework with a name " + framework);

    return impl->readNet(buffer, sizeBuffer);
}

void Net::readNet(const String& _model)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_model.empty());
    String model = _model;
    const std::string modelExt = model.substr(model.rfind('.') + 1);

    // TODO, to do, let TensorRT load ONNX model!
#ifdef HAVE_ORT
    if (modelExt == "onnx")
    {
        impl = makePtr<ImplORT>();
    }
    else 
#endif
#ifdef HAVE_TRT
    if (modelExt == "trt")
    {
        impl = makePtr<ImplTensorRT>();
    }
    else
#endif
#ifdef HAVE_MNN
    if (modelExt == "mnn")
    {
        impl = makePtr<ImplMNN>();
    }
    else 
#endif
    CV_Assert(impl && "Net::impl is empty! Please make sure the you have compiled OpenCV_lite with ONNXRuntime or MNN or TensorRT!");

    return impl->readNet(model);
}

void Net::setPreferablePrecision(int precisionId)
{
    CV_TRACE_FUNCTION();
    return impl->setPreferablePrecision((Precision)precisionId);
}

//void Net::forward(OutputArrayOfArrays outputBlobs,
//        const std::vector<String>& outBlobNames)
//{
//    CV_TRACE_FUNCTION();
//    CV_Assert(impl);
//    CV_Assert(!empty());
//    return impl->forward(outputBlobs, outBlobNames);
//}
//
//void Net::forward(std::vector<std::vector<Mat>>& outputBlobs,
//        const std::vector<String>& outBlobNames)
//{
//    CV_TRACE_FUNCTION();
//    CV_Assert(impl);
//    CV_Assert(!empty());
//    return impl->forward(outputBlobs, outBlobNames);
//}
//
//void Net::setPreferableBackend(int backendId)
//{
//    CV_TRACE_FUNCTION();
//    CV_TRACE_ARG(backendId);
//    CV_Assert(impl);
//    return impl->setPreferableBackend(*this, backendId);
//}
//
//void Net::setPreferableTarget(int targetId)
//{
//    CV_TRACE_FUNCTION();
//    CV_TRACE_ARG(targetId);
//    CV_Assert(impl);
//    return impl->setPreferableTarget(targetId);
//}

bool Net::empty() const
{
    CV_Assert(impl);
    return impl->empty();
}

void Net::setInputShape(const cv::String &inputName, const cv::dnn::MatShape &shape)
{
    CV_Assert(impl);
    return impl->setInputShape(inputName, shape);
}

void Net::setNumThreads(int num)
{
    CV_Assert(impl);
    impl->setNumThreads(num);
}

void Net::setInput(InputArray blob, const String& name)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());
    CV_Assert(impl);
    return impl->setInput(blob, name);
}

// TODO replace the following code with useful API.
// FIXIT return old value or add get method
//void Net::enableFusion(bool fusion)
//{
//    CV_TRACE_FUNCTION();
//    CV_Assert(impl);
//    return impl->enableFusion(fusion);
//}

//void Net::enableWinograd(bool useWinograd)
//{
//    CV_TRACE_FUNCTION();
//    CV_Assert(impl);
//    return impl->enableWinograd(useWinograd);
//}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
