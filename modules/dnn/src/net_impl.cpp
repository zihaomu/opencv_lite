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
    CV_LOG_WARNING(NULL, "The setNumThreads does nothing!");
}

// TODO finish the implement of the Precision and device
void Net::Impl::setPrecision(Compute_Precision precision)
{
    // TOOD
    CV_LOG_WARNING(NULL, "The setPrecision does nothing!");
}

void Net::Impl::setPreferDevice(Compute_Device device)
{
    CV_LOG_WARNING(NULL, "The setPrecision does nothing!");
}

void Net::Impl::readNet(const char *, size_t)
{
    CV_Error(Error::StsNotImplemented, "The readNet need be overrided!");
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
