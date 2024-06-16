// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_ORT
#include "onnxruntime_cxx_api.h"
#endif

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Net::Impl::~Impl()
{
    // nothing
}

Net::Impl::Impl()
{
    // nothing
}

bool Net::Impl::empty() const
{
    return false;
}

void Net::Impl::setNumThreads(int num)
{
    thread_num = num;
    CV_LOG_WARNING(NULL, "The setThreadsNum does nothing!");
}

// TODO finish the implement of the Precision and device
void Net::Impl::setPreferablePrecision(Precision precision)
{
    // TOOD
    CV_LOG_WARNING(NULL, "The setPreferablePrecision does nothing!");
}

void Net::Impl::setInputShape(const cv::String &, const cv::dnn::MatShape &)
{
    // TOOD
    CV_LOG_WARNING(NULL, "The setInputShape does nothing!");
}

void Net::Impl::setPreferableBackend(Backend _device)
{
    device = _device;
    CV_LOG_WARNING(NULL, "The setPreferableBackend does nothing!");
}

void Net::Impl::readNet(const char *, size_t)
{
    CV_Error(Error::StsNotImplemented, "readNet from buffer is not supported on current banckend!");
}

std::vector<std::string> Net::Impl::getInputName()
{
    return inputNamesString;
}

std::vector<MatShape> Net::Impl::getInputShape()
{
    return inputMatShape;
}

std::vector<std::string> Net::Impl::getOutputName()
{
    return outputNamesString;
}

std::vector<MatShape> Net::Impl::getOutputShape()
{
    return outputMatShape;
}

void Net::Impl::setType(ModelType _type)
{
    this->type = _type;
}

ModelType Net::Impl::getType()
{
    return type;
}

int Net::Impl::getNumThreads()
{
    return thread_num;
}

// TODO: Move the following function to a unified function.
Mat Net::Impl::forward(const String& outputName)
{
    std::vector<String> outputNames = {outputName};
    if (outputName.empty())
    {
        CV_Assert(outputNamesString.size() == 1 && "forward error! Please set the correct output name at .forward(\"SET_OUTPUT_NAME_HERE\")!");
        outputNames[0] = outputNamesString[0];
    }

    std::vector<Mat> outs;
    forward(outs, outputNames);

    return outs[0];
}

void Net::Impl::forward(OutputArrayOfArrays outputBlobs, const String& outputName)
{
    std::vector<String> outputNames = {outputName};

    if (outputName.empty())
    {
        CV_Assert(outputNamesString.size() == 1 && "forward error! Please set the correct output name at .forward(\"SET_OUTPUT_NAME_HERE\")!");
        outputNames[0] = outputNamesString[0];
    }

    return forward(outputBlobs, outputNames);
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
