// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

#ifdef HAVE_TFLITE

#include "tflite_custom_op/op_resolver.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace cv {
namespace dnn {

CV__DNN_INLINE_NS_BEGIN
namespace dnn_tflite {

void ImplTflite::readNet(const char* buffer, size_t sizeBuffer)
{
    modelTF = tflite::FlatBufferModel::BuildFromBuffer(buffer, sizeBuffer);
    this->init();
}

void ImplTflite::readNet(const cv::String &model)
{
    modelTF = tflite::FlatBufferModel::BuildFromFile(model.c_str());
    this->init();
}

void ImplTflite::parseInOutTensor()
{
    CV_Assert(interpreterTF);

    inputNamesString.clear();
    inputMatShape.clear();

    outputNamesString.clear();
    outputMatShape.clear();

    // TODO: how to get the media output? by setting the output name?
    // parsing the input and output tensor information.
    // Currently, we infer all the output.
    inputCount = interpreterTF->inputs().size();
    outputCount = interpreterTF->outputs().size();

    // parse input information
    for (int idx = 0; idx < interpreterTF->inputs().size(); idx++)
    {
        // get name
        inputNamesString.push_back(std::string(interpreterTF->GetInputName(idx)));

        // get shape
        auto tensorPtr = interpreterTF->input_tensor(idx);
        std::vector<int> shape(tensorPtr->dims->size, 0);
        memcpy(shape.data(), tensorPtr->dims->data, shape.size() * sizeof(int));
        inputMatShape.push_back(shape);
    }

    // parse output information
    for (int idx = 0; idx < interpreterTF->outputs().size(); idx++)
    {
        // get name
        outputNamesString.push_back(std::string(interpreterTF->GetOutputName(idx)));

        // get shape
        auto tensorPtr = interpreterTF->output_tensor(idx);
        std::vector<int> shape(tensorPtr->dims->size, 0);
        memcpy(shape.data(), tensorPtr->dims->data, shape.size() * sizeof(int));
        outputMatShape.push_back(shape);
    }
}

void ImplTflite::setPreferableBackend(Backend _device)
{
    if (device == _device)
        return;

#if defined(__ANDROID__)
    if (device == Backend::DNN_BACKEND_CPU && _device == Backend::DNN_BACKEND_GPU)
    {
        TfLiteGpuDelegateOptionsV2 options;
        options.is_precision_loss_allowed = 1;
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;

        delegate_ = TfLiteGpuDelegateV2Create(&options);
        if (interpreterTF->ModifyGraphWithDelegate(delegate_) != kTfLiteOk)
        {
            CV_Error(NULL, "DNN Tflite backend create GPU delegate failed!");
        }
        else
        {
            device = _device;
            if (interpreterTF->AllocateTensors() != kTfLiteOk)
            {
                CV_Error(cv::Error::StsError, "DNN TF-lite backend: Fail to allocate tflite AllocateTensors!");
            }

            this->parseInOutTensor();
        }
    }
    else if (device == Backend::DNN_BACKEND_GPU && _device == Backend::DNN_BACKEND_CPU)
    {
        auto options = TfLiteXNNPackDelegateOptionsDefault();
        options.num_threads = thread_num;
        delegate_ = TfLiteXNNPackDelegateCreate(&options);
        interpreterTF->ModifyGraphWithDelegate(delegate_);
        device = _device;
        this->parseInOutTensor();
    }
    else
#endif
    {
        CV_LOG_WARNING(NULL, "DNN Tflite backend create GPU delegate failed! Fallback to CPU");
    }
}

void ImplTflite::init()
{
    resolveTF = mp_op::MediaPipeBuiltinOpResolver();
    tflite::InterpreterBuilder builder(*modelTF, resolveTF);
    builder(&interpreterTF);

    if (interpreterTF == nullptr)
    {
        CV_Error(cv::Error::StsError, "DNN TF-lite backend: Fail to allocate tflite interpreter!");
    }

    if (interpreterTF->AllocateTensors() != kTfLiteOk)
    {
        CV_Error(cv::Error::StsError, "DNN TF-lite backend: Fail to allocate tflite AllocateTensors!");
    }

    this->parseInOutTensor();
}

// map tflite tensor to cv::Mat
static inline int convertTfTensor2CVType(const TfLiteType& tftensor_type)
{
    int cvType = -1;

    if (tftensor_type == kTfLiteInt8)
    {
        cvType = CV_8S;
    }
    else if (tftensor_type == kTfLiteInt32)
    {
        cvType = CV_32S;
    }
    else if (tftensor_type == kTfLiteUInt8)
    {
        cvType = CV_8U;
    }
    else if (tftensor_type == kTfLiteUInt32)
    {
        cvType = CV_32U;
    }
    else if (tftensor_type == kTfLiteFloat32)
    {
        cvType = CV_32F;
    }
    else if (tftensor_type == kTfLiteFloat16)
    {
        cvType = CV_FP16;
    }
    else
    {
        CV_Error(CV_StsError, "Unsupported Tflite Tensor type!");
    }
    return cvType;
}

static inline void tf_tensor2Mat(const TfLiteTensor* tensor, Mat& out)
{
    // get shape,
    std::vector<int> tensorShape(tensor->dims->size, 0);
    memcpy(tensorShape.data(), tensor->dims->data, sizeof(int) * tensor->dims->size);

    auto tf_tensor_type = tensor->type;
    int cvType = convertTfTensor2CVType(tf_tensor_type);

    // deep copy the data from Tensor to output Mat.
    Mat(tensorShape.size(), &tensorShape[0], cvType, tensor->data.raw).copyTo(out);

    // TODO check if the 1D has the correct shape!
    if (tensorShape.size() == 1)
        out.dims = 1;
}

static void tf_tensors2Mats(const std::vector<const TfLiteTensor*>& tensors, std::vector<Mat>& outs)
{
    if (outs.empty() || outs.size() != tensors.size())
        outs.resize(tensors.size());

    int len = tensors.size();

    for (int i = 0; i < len; i++)
    {
        tf_tensor2Mat(tensors[i], outs[i]);
    }
}

// TODO check that if we set new thread number, should reset the session and the input and output tensor?
void ImplTflite::setNumThreads(int num)
{
    if (num <= 0)
    {
        CV_LOG_WARNING(NULL, "MNN Backend: the threads number is smaller than 0, USE 4 threads as default!")
        num = std::max(getNumThreads(), 1);
    }

    thread_num = num;
    CV_Assert(interpreterTF && "Error in DNN Tflite backend: interpreterTF is empty!");
    interpreterTF->SetNumThreads(thread_num);
}

void ImplTflite::setInputShape(const cv::String &inputName, const cv::dnn::MatShape &shape)
{
    auto iter = std::find(inputNamesString.begin(), inputNamesString.end(), inputName);

    if (iter == inputNamesString.end())
    {
        CV_Error(CV_StsError, "DNN Tf-lite Backend: setInputShape get wrong input name!");
    }

    int indexRes = iter - inputNamesString.begin();
    if (interpreterTF->ResizeInputTensor(indexRes, shape) != kTfLiteOk)
    {
        CV_Error(CV_StsError, "DNN Tf-lite Backend: can not resize the specific input shape!");
    }

    if (interpreterTF->AllocateTensors() != kTfLiteOk) {
        CV_Error(CV_StsError, "DNN Tf-lite Backend: can not allocateTensor in setInputShape func!");
    }

    this->parseInOutTensor();
}

void ImplTflite::setPreferablePrecision(Precision precision)
{
    if (precision == Precision::DNN_PRECISION_NORMAL)
    {
        return;
    }
    else if (precision == Precision::DNN_PRECISION_LOW)
    {
        interpreterTF->SetAllowFp16PrecisionForFp32(true);
    }
    else if (precision == Precision::DNN_PRECISION_HIGH)
    {
        interpreterTF->SetAllowFp16PrecisionForFp32(false);
    }
    else
    {
        CV_Error(CV_StsNotImplemented, "The precision is not supported!");
    }
}

ImplTflite::~ImplTflite()
{

#if defined(__ANDROID__)
    if (device == Backend::DNN_BACKEND_GPU)
        TfLiteGpuDelegateV2Delete(delegate_);
#endif
    delete delegate_;
}

ImplTflite::ImplTflite()
{
}

void ImplTflite::forward(cv::OutputArrayOfArrays outputBlobs, const std::vector<String> &outBlobNames)
{
    CV_Assert(!empty());
    CV_Assert(outputBlobs.isMatVector());
    // Output depth can be CV_32F or CV_8S
    std::vector<Mat>& outputvec = *(std::vector<Mat>*)outputBlobs.getObj();

    std::vector<const char*> outputNames;
    int outSize = outBlobNames.size();
    CV_Assert(outSize <= outputCount && "OpenCV DNN forward() error, expected value exceeds existing value.");

    // set output file.
    std::vector<int> expectOutIndex;
    for (int i = 0; i < outSize; i++)
    {
        auto iter = std::find(outputNamesString.begin(), outputNamesString.end(), outBlobNames[i]);
        CV_Assert(iter != outputNamesString.end() && "Can not found the expected output!");
        outputNames.push_back(outBlobNames[i].c_str());
        expectOutIndex.push_back(iter - outputNamesString.begin());
    }

    CV_Assert(expectOutIndex.size() > 0);
    this->interpreterTF->Invoke();

    std::vector<const TfLiteTensor *> outputTensors;
    for (auto& idx : expectOutIndex)
    {
        outputTensors.push_back(interpreterTF->output_tensor(idx));
    }

    tf_tensors2Mats(outputTensors, outputvec);
}

void ImplTflite::setInput(cv::InputArray blob_, const cv::String &name)
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

    CV_Assert(indexRes != -1 && indexRes < inputCount && "Tf-lite Backend: indexRes error called in setInput(), please check the input name!");

    std::vector<int> tensorShape = inputMatShape[indexRes];
    std::vector<int> matShape = shape(blob);
    size_t totalValueT = total(tensorShape);
    size_t totalValueM = blob.total();

    if (totalValueT != totalValueM || tensorShape.size() != matShape.size())
    {
        if (tensorShape.size() == 4 && matShape.size() == 4)
        {
            CV_LOG_WARNING(NULL, cv::format("The given input shape is [%d x %d x %d x %d], and the expected input shape "
                                            "is [%d x %d x %d x %d]", matShape[0], matShape[1], matShape[2], matShape[3],
                                            tensorShape[0], tensorShape[1], tensorShape[2], tensorShape[3]));
        }
        if (tensorShape.size() == 3 && matShape.size() == 3)
            CV_LOG_WARNING(NULL, cv::format("The given input shape is [%d x %d x %d], and the expected input shape is "
                                            "[%d x %d x %d]", matShape[0], matShape[1], matShape[2], tensorShape[0],
                                            tensorShape[1], tensorShape[2]));
        CV_Error(CV_StsError, "The input shape dose not match the expacted input shape! \n"
                              "NOTE: the setInput only accept NCHW format which can be generated by blobFromImage function.");
    }

    auto inpTensor = interpreterTF->input_tensor(indexRes);
    auto tf_type = inpTensor->type;
    int cvType = convertTfTensor2CVType(tf_type);

    CV_Assert(cvType == blob.depth() && "The input Mat type is not match the Tf-lite Tensor type!");

    // copy data from opencv Mat to tf-lite tensor input.
    memcpy(inpTensor->data.raw, blob.data, blob.total() * CV_ELEM_SIZE1(cvType));
}

CV__DNN_INLINE_NS_END
}}
}  // namespace cv::dnn::tflite

#endif
