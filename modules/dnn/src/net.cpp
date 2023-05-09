// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Net::Net()
    : impl(makePtr<Net::Impl>())
{
}

Net::~Net()
{
}

//Mat Net::forward(const String& outputName)
//{
//    CV_TRACE_FUNCTION();
//    CV_Assert(impl);
//    CV_Assert(!empty());
//    return impl->forward(outputName);
//}

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
    return impl->getInputName();
}

std::vector<std::string> Net::getOutputName()
{
    CV_TRACE_FUNCTION();
    return impl->getOutputName();
}

std::vector<MatShape> Net::getInputShape()
{
    CV_TRACE_FUNCTION();
    return impl->getInputShape();
}

std::vector<MatShape> Net::getOutputShape()
{
    CV_TRACE_FUNCTION();
    return impl->getOutputShape();
}

void Net::readNet(const String& model)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!model.empty());
    return impl->readNet(model);
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
