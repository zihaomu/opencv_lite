// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

#if HAVE_MNN
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void ImplMNN::parseTensorInfoFromSession()
{
    const std::map<std::string, MNN::Tensor*> inputs = this->netPtr->getSessionInputAll(this->session);
    const std::map<std::string, MNN::Tensor*> outputs = this->netPtr->getSessionOutputAll(this->session);

    inputCount = inputs.size();
    outputCount = outputs.size();

    inputNamesString.clear();
    inTensorsPtr.clear();
    inputMatShape.clear();
    // get all input and output name and shape
    for (auto iter = inputs.begin(); iter != inputs.end(); iter++)
    {
        inputNamesString.push_back(iter->first);
        inTensorsPtr.push_back(iter->second);

        auto shape = iter->second->shape();
        inputMatShape.push_back(shape);
    }

    outputNamesString.clear();
    outTensorsPtr.clear();
    outputMatShape.clear();
    for (auto iter = outputs.begin(); iter != outputs.end(); iter++)
    {
        outputNamesString.push_back(iter->first);
        outTensorsPtr.push_back(iter->second);

        auto shape = iter->second->shape();
        outputMatShape.push_back(shape);
    }
}

void ImplMNN::readNet(const cv::String &model)
{
    netPtr = MNN::Interpreter::createFromFile(model.c_str());
    session = netPtr->createSession(config);

    // get the tensor info from session!
    parseTensorInfoFromSession();
}

static inline int getTypeSize(int cvType)
{
    int esz = 0;

    switch (CV_MAT_DEPTH(cvType))
    {
        case CV_8U:
        case CV_8S:
        {
            esz = sizeof(uint8_t);
            break;
        }
        case CV_16U:
        case CV_16S:
        {
            esz = sizeof(int16_t);
            break;
        }
        case CV_32U:
        case CV_32S:
        {
            esz = sizeof(int);
            break;
        }
        case CV_32F:
        {
            esz = sizeof(float );
            break;
        }
        case CV_64F:
        {
            esz = sizeof(double );
            break;
        }
        case CV_64S:
        case CV_64U:
        {
            esz = sizeof(int64_t);
            break;
        }
        default:
        {
            CV_Error(CV_StsError, "Un-known OpenCV data CV type in getTypeSize()!");
        }
    }

    return esz;
}

static inline int convertMNN2CVType(const halide_type_t& mnn_halide_type)
{
    auto halide_code = mnn_halide_type.code;
    auto halide_bytes = mnn_halide_type.bytes();
    int cvType = -1;

    if (halide_code == halide_type_int && halide_bytes == 1)
    {
        cvType = CV_8S;
    }
    else if (halide_code == halide_type_int && halide_bytes == 4)
    {
        cvType = CV_32S;
    }
    else if (halide_code == halide_type_uint && halide_bytes == 1)
    {
        cvType = CV_8U;
    }
    else if (halide_code == halide_type_uint && halide_bytes == 4)
    {
        cvType = CV_32U;
    }
    else if (halide_code == halide_type_float && halide_bytes == 4)
    {
        cvType = CV_32F;
    }
    else
    {
        CV_Error(CV_StsError, "Unsupported MNN Tensor type!");
    }
    return cvType;
}

static inline void tensor2Mat(const MNN::Tensor* tensor, Mat& out)
{
    // get shape,
    auto tensorShape = tensor->shape();

    auto mnn_halide_type = tensor->getType();
    int cvType = convertMNN2CVType(mnn_halide_type);

    // deep copy the data from Tensor to output Mat.
    Mat(tensorShape.size(), &tensorShape[0], cvType, tensor->host<void>()).copyTo(out);

    // TODO check if the 1D has the correct shape!

    if (tensorShape.size() == 1)
        out.dims = 1;
}

static void tensors2Mats(const std::vector<MNN::Tensor*>& tensors, std::vector<Mat>& outs)
{
    if (outs.empty() || outs.size() != tensors.size())
        outs.resize(tensors.size());

    int len = tensors.size();

    for (int i = 0; i < len; i++)
    {
        tensor2Mat(tensors[i], outs[i]);
    }
}

// TODO check that if we set new thread number, should reset the session and the input and output tensor?
void ImplMNN::setNumThreads(int num)
{
    if (num <= 0)
    {
        CV_LOG_WARNING(NULL, "MNN Backend: the threads number is smaller than 0, USE 4 threads as default!")
        num = std::max(getNumThreads(), 1);
    }

    thread_num = num;
    config.numThread = thread_num;
    config.type = MNN_FORWARD_CPU;

    if (session)
    {
        netPtr->releaseSession(session);
        session = netPtr->createSession(config);
        CV_LOG_ONCE_WARNING(NULL, "MNN Backend: set num threads success. Please make sure call setNumThreads() before setInput()!");
        // get the tensor info from session!
        parseTensorInfoFromSession();
    }
}

ImplMNN::~ImplMNN()
{
    if (netPtr)
        netPtr->releaseModel();
    if (session)
        netPtr->releaseSession(session);
}

ImplMNN::ImplMNN()
{
    config.numThread = thread_num;
    config.type = MNN_FORWARD_CPU;
}

void ImplMNN::forward(cv::OutputArrayOfArrays outputBlobs, const std::vector<String> &outBlobNames)
{
    CV_Assert(!empty());
    CV_Assert(outputBlobs.isMatVector());
    // Output depth can be CV_32F or CV_8S
    std::vector<Mat>& outputvec = *(std::vector<Mat>*)outputBlobs.getObj();

    std::vector<const char*> outputNames;
    int outSize = outBlobNames.size();
    CV_Assert(outSize <= outputCount && "OpenCV DNN forward() error, expected value exceeds existing value.");

    // set output file.
    for (int i = 0; i < outSize; i++)
    {
        auto iter = std::find(outputNamesString.begin(), outputNamesString.end(), outBlobNames[i]);
        CV_Assert(iter != outputNamesString.end() && "Can not found the expacted output!");
        outputNames.push_back(outBlobNames[i].c_str());
    }

    this->netPtr->runSession(this->session);

    std::vector<MNN::Tensor* > outputTensors(outputNames.size(), nullptr);
    // copy all the output Tensor to Host
    for (int i = 0; i < outputCount; i++)
    {
        outputTensors[i] = netPtr->getSessionOutput(session, outputNames[i]);
        outputTensors[i]->copyToHostTensor(outputTensors[i]);
    }

    tensors2Mats(outputTensors, outputvec);
}

void ImplMNN::setInput(cv::InputArray blob_, const cv::String &name)
{
    Mat blob = blob_.getMat();

    int indexRes = -1;

    // if the name is empty, we need to check out if the mode only need 1 input,
    // if it's true, then we set this input as this input.
    if (name.empty())
    {
        CV_Assert(inputNamesString.size() == 1 && "Please set the input name, the default input name can only be used in single input model.");
        indexRes = 0;
    }

    // find input index to get shape info.
    if (indexRes == -1)
    {
        auto iter = std::find(inputNamesString.begin(), inputNamesString.end(), name);

        if (iter != inputNamesString.end())
        {
            indexRes = iter - inputNamesString.begin();
        }
    }

    CV_Assert(indexRes != -1 && indexRes < inputCount && inTensorsPtr[indexRes] && "MNN Backend: indexRes error called in setInput()!");

    // Use MNN converter to transfer the Mat to MNN::tensor.
    std::vector<int> tensorShape = inTensorsPtr[indexRes]->shape();
    std::vector<int> matShape = shape(blob);
    size_t totalValueT = total(tensorShape);
    size_t totalValueM = blob.total();

    if (totalValueT != totalValueM || tensorShape.size() != matShape.size())
    {
        CV_Error(CV_StsError, "The input shape dose not match the expacted input shape! \n"
                              "NOTE: the setInput only accept NCHW format which can be generated by blobFromImage function.");
    }

    auto halide_type = inTensorsPtr[indexRes]->getType();
    int cvType = convertMNN2CVType(halide_type);

    CV_Assert(cvType == blob.depth() && "The input Mat type is not match the MNN Tensor type!");

    // Create template Tensor of CAFFE data layout, and transform it to expected data layout.
    MNN::Tensor* tmp = MNN::Tensor::create(matShape, halide_type, blob.data, MNN::Tensor::CAFFE);
    inTensorsPtr[indexRes]->copyFromHostTensor(tmp);
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif
