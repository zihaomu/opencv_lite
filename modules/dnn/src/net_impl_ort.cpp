// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

#ifdef HAVE_ORT
#include "onnxruntime_cxx_api.h"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// function convert ONNX Tensor to Mat.
static void MatShapeToInt64(const MatShape& inShape, std::vector<int64>& outShape)
{
    int sizeLen = inShape.size();
    if (outShape.empty() || (int)outShape.size() != sizeLen)
        outShape.resize(sizeLen);
    for (int i = 0; i < sizeLen; i++)
    {
        CV_Assert(inShape[i] > 0);
        outShape[i] = inShape[i];
    }
}

// convert int64 matshape (ORT type) to int 32 matshape (opencv type)
static void int64ToMatShape(const std::vector<int64>& inShape, MatShape& outShape)
{
    int sizeLen = inShape.size();
    if (outShape.empty() || (int)outShape.size() != sizeLen)
        outShape.resize(sizeLen);

    for (int i = 0; i < sizeLen; i++)
    {
        if (inShape[i] >= 0)
            outShape[i] = inShape[i];
        else
            outShape[i] = 1;
    }
}

// this should be removed after opencv has full 1D supported.
static void fix1DError(std::vector<int64>& int64Shape)
{
    if (int64Shape.size() == 0)
        int64Shape.assign(1, 1);
}

static int typeConvertFromORT2CV(ONNXTensorElementDataType ORTtype)
{
    int cvType = -1;
    switch (ORTtype)
    {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        {
            cvType = CV_32F;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        {
            cvType = CV_8S;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        {
            cvType = CV_8U;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        {
            cvType = CV_16S;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        {
            cvType = CV_16U;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        {
            cvType = CV_32S;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        {
            cvType = CV_32U;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        {
            cvType = CV_16F;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        {
            cvType = CV_64F;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        {
            cvType = CV_64S;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        {
            cvType = CV_64U;
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        {
            cvType = CV_BOOL;
            break;
        }
        default:
        {
            cvType = -1;
            break;
        }
    }

    CV_Assert(cvType != -1 && "Unsupported data type in typeConvertFromORT2CV!");
    return cvType;
}

// TODO: support the different data type.
// convert ORT tensor to OpenCV Mat.
static inline void tensor2Mat(const Ort::Value& tensor, Mat& out)
{
    Ort::TypeInfo tInfo = tensor.GetTypeInfo();
    Ort::ConstTensorTypeAndShapeInfo tAndS = tInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tAndS.GetElementType();

    int cvType = typeConvertFromORT2CV(type);
    void* data = const_cast<void*>(tensor.GetTensorRawData());

    std::vector<int64_t> dimsInt64 = tAndS.GetShape();

    fix1DError(dimsInt64);

    MatShape dims;
    int64ToMatShape(dimsInt64, dims);

    out = Mat(dims.size(), &dims[0], cvType, data);
    // To force 1-dimensional cv::Mat for scalars.
    if (dimsInt64.size() == 1)
        out.dims = 1;
}

static void tensors2Mats(const std::vector<Ort::Value>& tensors, std::vector<Mat>& outs)
{
    if (outs.empty() || outs.size() != tensors.size())
        outs.resize(tensors.size());

    int len = tensors.size();

    for (int i = 0; i < len; i++)
    {
        tensor2Mat(tensors[i], outs[i]);
    }
}

ImplORT::~ImplORT()
{
    // nothing
}

ImplORT::ImplORT()
{
    // ONNXRuntime
    sessionOptions.SetIntraOpNumThreads(thread_num);
}

void ImplORT::setNumThreads(int num)
{
    if (num <= 0)
    {
        CV_LOG_WARNING(NULL, "The threads number is smaller than 0, USE all threads as default!")
        num = std::max(getNumThreads(), 1);
    }

    thread_num = num;
    sessionOptions.SetIntraOpNumThreads(thread_num);
}

// TODO, Since TensorRT can support ONNX, we need add the flag for switching ONNX to ORT and TensorRT backend.
void ImplORT::readNet(const String& model)
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
#ifdef _WIN32
    std::wstring wModel_path(model.begin(), model.end());
    session = Ort::Session(env, wModel_path.c_str(), sessionOptions);
#else
    session = Ort::Session(env, model.c_str(), sessionOptions);
#endif

    inputCount = session.GetInputCount();
    outputCount = session.GetOutputCount();

    // TODO: check if we need the following code?
    for (int i = 0; i < inputCount; i++)
    {
        auto tmp_name = session.GetInputNameAllocated(i, allocator);
        inputNamesString.push_back(tmp_name.get());

        Ort::TypeInfo inpInfor = session.GetInputTypeInfo(i);
        auto tAndS = inpInfor.GetTensorTypeAndShapeInfo();

        inDataType.push_back(tAndS.GetElementType());
        std::vector<int64_t> dimsInt64 = tAndS.GetShape();
        inputInt64.push_back(dimsInt64);

        fix1DError(dimsInt64);

        MatShape dims;
        int64ToMatShape(dimsInt64, dims);
        inputMatShape.push_back(dims);
    }

    for (int i = 0; i < outputCount; i++)
    {
        auto tmp_name = session.GetOutputNameAllocated(i, allocator);
        outputNamesString.push_back(tmp_name.get());

        Ort::TypeInfo outInfor = session.GetOutputTypeInfo(i);
        auto tAndS = outInfor.GetTensorTypeAndShapeInfo();

        outDataType.push_back(tAndS.GetElementType());

        std::vector<int64_t> dimsInt64 = tAndS.GetShape();
        outputInt64.push_back(dimsInt64);

        fix1DError(dimsInt64);

        MatShape dims;
        int64ToMatShape(dimsInt64, dims);
        outputMatShape.push_back(dims);
    }
}

void ImplORT::forward(OutputArrayOfArrays outputBlobs, const std::vector<String>& outBlobNames)
{
    CV_Assert(!empty());
    CV_Assert(outputBlobs.isMatVector());
    // Output depth can be CV_32F or CV_8S
    std::vector<Mat>& outputvec = *(std::vector<Mat>*)outputBlobs.getObj();

    std::vector<const char*> outputNames;
    int outSize = outBlobNames.size();
    CV_Assert(outSize <= outputCount && "ONNXRuntime error, expected value exceeds existing value.");

    // set output file.
    for (int i = 0; i < outSize; i++)
    {
        auto iter = std::find(outputNamesString.begin(), outputNamesString.end(), outBlobNames[i]);
        CV_Assert(iter != outputNamesString.end() && "Can not found the expacted output!");
        outputNames.push_back(outBlobNames[i].c_str());
    }

    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
                                                        inputTensors.size(), outputNames.data(), outputNames.size());

    tensors2Mats(outputTensors, outputvec);
}

// buffers are only used at some case, like int64, since we do not support the int64 of mat, so we use buffers reserve the int64 data.
static void mat2Tensor(Mat& m, const std::vector<int64_t>& expactedInt64, int, Ort::MemoryInfo& memoryInfo, Ort::Value& t)
{
    MatShape blobShape = shape(m);

    // since opencv does not support 1d mat, all 1d will be parsed as 2d. [x] -> [x,1].
    // we need to delete the final extra 1 to recover 1D.
    if (blobShape.size() == expactedInt64.size() + 1 && blobShape[expactedInt64.size()] == 1)
    {
        blobShape.erase(blobShape.end() - 1);
    }

    CV_Assert(blobShape.size() == expactedInt64.size() && "The expacted dimension is not correct.");

    size_t mTotal = m.total();
    int cvType = m.depth();

    if(cvType == CV_32F)
    {
        t = Ort::Value::CreateTensor<float>(memoryInfo, m.ptr<float>(), mTotal,
                                            expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_8S)
    {
        t = Ort::Value::CreateTensor<int8_t>(memoryInfo, m.ptr<int8_t>(), mTotal,
                                             expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_8U)
    {
        t = Ort::Value::CreateTensor<uint8_t>(memoryInfo, m.ptr<uint8_t>(), mTotal,
                                              expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_16S)
    {
        t = Ort::Value::CreateTensor<int16_t>(memoryInfo, m.ptr<int16_t>(), mTotal,
                                               expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_16U)
    {
        t = Ort::Value::CreateTensor<uint16_t>(memoryInfo, m.ptr<uint16_t>(), mTotal,
                                                     expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_16F)
    {
        t = Ort::Value::CreateTensor<Ort::Float16_t>(memoryInfo, (Ort::Float16_t *)m.ptr<uint16_t>(), mTotal,
                                                     expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_32U)
    {
        t = Ort::Value::CreateTensor<uint>(memoryInfo, m.ptr<uint>(), mTotal,
                                          expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_32S)
    {
        t = Ort::Value::CreateTensor<int>(memoryInfo, m.ptr<int>(), mTotal,
                                          expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_64F) // double type
    {
        t = Ort::Value::CreateTensor<double>(memoryInfo, m.ptr<double>(), mTotal,
                                             expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_64S) // int 64
    {
        t = Ort::Value::CreateTensor<int64_t>(memoryInfo, m.ptr<int64_t>(), mTotal,
                                              expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_64U) // uint 64
    {
        t = Ort::Value::CreateTensor<uint64_t>(memoryInfo, m.ptr<uint64_t>(), mTotal,
                                              expactedInt64.data(), expactedInt64.size());
    }
    else if (cvType == CV_BOOL)
    {
        t = Ort::Value::CreateTensor<bool>(memoryInfo, m.ptr<bool>(), mTotal,
                                           expactedInt64.data(), expactedInt64.size());
    }
    else
    {
        CV_Error(CV_StsError, "Unsupported data format!");
    }
}

static inline int totalInt64(const std::vector<int64_t>& int64Shape)
{
    int total = 1;
    for (int i = 0; i < (int)int64Shape.size(); i++)
    {
        total *= int64Shape[i];
    }
    return  total;
}

// Set only one input, set more than one input.
// TODO: add new function, and let this function works on different backend.
void ImplORT::setInput(InputArray blob_, const String& name)
{
    Mat blob = blob_.getMat();

    int index = -1, indexRes = -1;

    // if the name is empty, we need to check out if the mode only need 1 input,
    // if it's true, then we set this input as this input.
    if (name.empty())
    {
        CV_Assert(inputNamesString.size() == 1 && "Please set the input name, the default input name can only be used in single input model.");
        indexRes = 0;
    }

    // reuse input tensor.
    if (!inputNames.empty())
    {
        auto iter = std::find(inputNames.begin(), inputNames.end(), name.c_str());

        if (iter != inputNames.end())
        {
            index = iter - inputNames.begin();
        }
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

    CV_Assert(indexRes != -1);

    // create Tensor.
    Ort::Value tensor(nullptr);

    // fix the dynamic shape input.
    MatShape blobShape = shape(blob);
    if (total(blobShape) != total(inputMatShape[indexRes]))
    {
        if (blobShape.size() == inputMatShape[indexRes].size())
        {
            inputMatShape[indexRes] = blobShape;
            MatShapeToInt64(inputMatShape[indexRes], inputInt64[indexRes]);
        }
        else
            CV_Error(CV_StsError, "The input shape dose not match the expacted input shape!");
    }

    std::vector<int64_t> int64ShapeCopy(inputInt64[indexRes]);
    int total_int64 = totalInt64(int64ShapeCopy);

    // Fix dynamic shape error, since the dynamic shape will be set as -1.
    // If total_int64 < 0, we think input is dynamic shape, we need to set correct input shape here.
    if (total(blobShape) != total_int64 && total_int64 < 0)
    {
        CV_Assert(int64ShapeCopy.size() == blobShape.size());
        for (int i = 0; i < int64ShapeCopy.size(); i++)
        {
            int64ShapeCopy[i] = (int64_t)blobShape[i];
        }
    }

    mat2Tensor(blob, int64ShapeCopy, index, memoryInfo, tensor);

    if (index != -1)
    {
        // TODO: if the following code.
        inputTensors[index].release();
        inputTensors[index] = std::move(tensor);
    }
    else
    {
        inputNames.push_back(inputNamesString[indexRes].c_str());
        inputTensors.push_back(std::move(tensor));
    }
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif