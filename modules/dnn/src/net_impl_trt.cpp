// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "./trt_utils/trt_utils.h"
#include "./trt_utils/trt_logger.h"

#ifdef HAVE_TRT

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN
namespace dnn_trt {

#define OPT_MAX_WORK_SPACE_SIZE ((size_t)1 << 30)

inline int convertTrt2CVType(const ::nvinfer1::DataType type)
{
    int cvType = -1;

    if (::nvinfer1::DataType::kFLOAT == type)
    {
        cvType = CV_32F;
    }
    else if (::nvinfer1::DataType::kHALF == type)
    {
        cvType = CV_16F;
    }
    else if (::nvinfer1::DataType::kINT8 == type)
    {
        cvType = CV_8S;
    }
    else if (::nvinfer1::DataType::kINT32 == type)
    {
        cvType = CV_32S;
    }
    else if (::nvinfer1::DataType::kUINT8 == type)
    {
        cvType = CV_8U;
    }
    else
    {
        CV_Error(CV_StsError, "TensorRT: Unsupported Trt Tensor type!");
    }

    return cvType;
}

// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& getTensorrtLogger(bool verbose_log = false)
{
    const auto log_level = verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING;
    static TensorrtLogger trt_logger(log_level);

    if (log_level != trt_logger.get_level())
    {
        trt_logger.set_level(verbose_log ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    }

    return trt_logger;
}

ImplTensorRT::ImplTensorRT()
{
    configTRT = {};
}

ImplTensorRT::~ImplTensorRT()
{
    for (int i = 0; i < bufferListDevice.size(); i++)
    {
        cudaFree(bufferListDevice[i]);
    }

    cudaStreamDestroy(stream_);
}

void ImplTensorRT::setNumThreads(int num)
{
    configTRT.threadNum = num;
}

void ImplTensorRT::setConfig(NetConfig* config)
{
    CV_Assert(config && "config pointer is empty!");
    NetConfig_TRT* configTrtTmp = static_cast<NetConfig_TRT*>(config);

    if (!configTrtTmp)
    {
        configTRT = NetConfig_TRT(*config);
    }
    else
    {
        configTRT = *configTrtTmp;  // copy trt to trt
    }
}

// remove the file name and return the file prefix path.
inline std::string removeFileName(const std::string& filePath)
{
    size_t pos = filePath.find_last_of("/\\");

    if (pos != std::string::npos)
    {
        return filePath.substr(0, pos);
    }
    else
    {
        return "";
    }
}

// return file name with file extention suffix
inline std::string extractFileName(const std::string& filePath)
{
    size_t pos = filePath.find_last_of("/\\");

    if (pos != std::string::npos)
    {
        return filePath.substr(pos + 1);
    }
    else
    {
        return filePath;
    }
}

// remove file extention suffix
inline std::string removeFileSuffix(const std::string& filePath)
{
    size_t pos = filePath.find_last_of(".");

    if (pos != std::string::npos)
    {
        return filePath.substr(0, pos);
    }
    else
    {
        return filePath;
    }
}

inline MatShape convertDim2Shape(const ::nvinfer1::Dims& dim)
{
    MatShape shape(dim.nbDims, 0);
    memcpy(shape.data(), dim.d, dim.nbDims * sizeof(int));

    return shape;
}

inline std::vector<char> loadTimingCacheFile(const std::string inFileName)
{
    std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);

    if (!iFile)
    {
        CV_LOG_INFO(NULL, cv::String("[TensorRT EP] Could not read timing cache from: "+inFileName
                            +". A new timing cache will be generated and written."));
        return std::vector<char>();
    }

    iFile.seekg(0, std::ifstream::end);
    size_t fsize = iFile.tellg();
    iFile.seekg(0, std::ifstream::beg);
    std::vector<char> content(fsize);
    iFile.read(content.data(), fsize);
    iFile.close();

    return content;
}

inline void saveTimingCacheFile(const std::string outFileName, const nvinfer1::IHostMemory* blob)
{
    std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);

    if (!oFile)
    {
        CV_LOG_INFO(NULL, cv::String("[TensorRT EP] Could not write timing cache to: "+outFileName));
        return;
    }

    oFile.write((char*)blob->data(), blob->size());
    oFile.close();
}

void ImplTensorRT::tensors2Mats(const std::vector<int>& outputIdxs, std::vector<Mat>& outs)
{
    if (outs.empty() || outs.size() != outputIdxs.size())
        outs.resize(outputIdxs.size());

    for (int i = 0; i < outputIdxs.size(); i++)
    {
        int idx = outputIdxs[i];
        int bindingIdx = output_idxs[idx];
        int cvType = convertTrt2CVType(engine_->getBindingDataType(bindingIdx));

        CV_Assert(cvType != -1 && "Unsupported data type");
        Mat(outputMatShape[idx], cvType, bufferListHost[bindingIdx].first.data()).copyTo(outs[idx]);
    }
}

void ImplTensorRT::readNet(const String& model_name)
{
    device_id_ = configTRT.deviceId;
    cudaSetDevice(device_id_); // set specific GPU device for TensorRT backend.
    //cudaDeviceProp prop;
    //cudaGetDeviceProperties(&prop, device_id_);
    //compute_capability_ = getComputeCapacity(prop);

    std::string trt_model_filename = "";
    std::string root_path_name = "";

    bool is_trt_model = false;
    bool is_onnx_model = false;

    // if model is trt
    if (model_name.find(".trt") != std::string::npos)
    {
        trt_model_filename = std::string(model_name);
        is_trt_model = true;
    }
    else if (model_name.find(".onnx") != std::string::npos)
    {
        if (configTRT.useCache && !configTRT.cachePath)
        {
            // for onnx model, when we use trt cache, we need check if the cache has related trt cache file.
            std::string modelFileName = removeFileSuffix(extractFileName(model_name));
            root_path_name = removeFileName(model_name);

            trt_model_filename = root_path_name + modelFileName + ".trt\0";

            std::ifstream trtFile(trt_model_filename);

            if (trtFile.is_open())  // related cache file exist, use cache file.
            {
                CV_LOG_INFO(NULL, cv::String("DNN TensorRT backend: found Trt cache model:" + trt_model_filename));
                is_trt_model = true;
            }
            else // no cache file was found,
            {
                is_trt_model = false;
            }
        }
        else // no cache file was found,
        {
            is_trt_model = false;
        }
    }
    else
    {
        CV_Error(Error::StsNotImplemented, cv::String("Failed to load model in TensorRT backend:" + model_name));
        return;
    }

    {
        AutoLock lock(mutex);
        runtime_ = Ptr<::nvinfer1::IRuntime>(::nvinfer1::createInferRuntime(getTensorrtLogger()));
    }

    if (!runtime_)
    {
        CV_Error(Error::StsError, "TensorRT backend: Failed to create runtime!");
        return;
    }

    std::string timeCachingPath = "";
    /*** create engine from model file ***/
    if (is_trt_model)
    {
        /* Just load TensorRT model (serialized model) */
        std::ifstream engine_file(trt_model_filename, std::ios::binary | std::ios::in);
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        engine_file.read((char*)engine_buf.get(), engine_size);

        {
            AutoLock lock(mutex);
            engine_ = Ptr<::nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
        }

        engine_file.close();
        if (!engine_)
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to create engine!");
            return;
        }
    }
    else // is onnx
    {
        if (configTRT.useTimeCache)
        {
            timeCachingPath = getTimingCachePath(root_path_name, compute_capability_);
        }
        /* Create a TensorRT model from another format */
        AutoLock lock(mutex);
        builder_ = Ptr<::nvinfer1::IBuilder>(::nvinfer1::createInferBuilder(getTensorrtLogger()));

        const auto explicitBatch = 1U << static_cast<uint32_t>(::nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network_ = Ptr<::nvinfer1::INetworkDefinition>(builder_->createNetworkV2(explicitBatch));
        config_ = Ptr<::nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());

        auto parser = Ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_, getTensorrtLogger()));

        if (!parser->parseFromFile(model_name.c_str(), (int)::nvinfer1::ILogger::Severity::kWARNING))
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to parse onnx file!");
            return;
        }

        config_->setMaxWorkspaceSize(OPT_MAX_WORK_SPACE_SIZE);

        // TODO add fp16 support
        if (configTRT.useFP16)
        {
            config_->setFlag(::nvinfer1::BuilderFlag::kFP16);
        }

        // Trying to load time cache file
        std::unique_ptr<nvinfer1::ITimingCache> timing_cache = nullptr;
        if (configTRT.useTimeCache)
        {
            // Loading time cache file, create a fresh cache if the file doesn't exist.
            std::vector<char> loaded_timing_cache = loadTimingCacheFile(timeCachingPath);
            timing_cache.reset(config_->createTimingCache(static_cast<const void*>(loaded_timing_cache.data()), loaded_timing_cache.size()));
            if (timing_cache == nullptr)
            {
                CV_Error(Error::StsError, "TensorRT backend: Failed to create timing cache!");
                return;
            }
            config_->setTimingCache(*timing_cache, false);
        }

        auto plan = Ptr<::nvinfer1::IHostMemory>(builder_->buildSerializedNetwork(*network_, *config_));

        if (!plan)
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to serialized network!");
            return;
        }

        engine_ = Ptr<::nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

        if (!engine_)
        {
            CV_Error(Error::StsError, "TensorRT backend: Failed to create engine!");
            return;
        }

        /* save serialized model for next time */
        if (configTRT.useCache && !configTRT.cachePath)
        {
            std::ofstream ofs(std::string(trt_model_filename), std::ios::out | std::ios::binary);
            ofs.write((char*)(plan->data()), plan->size());
            ofs.close();
        }

        // Trying to save time cache file.
        if (configTRT.useTimeCache)
        {
            auto timing_cache = config_->getTimingCache();
            std::unique_ptr<nvinfer1::IHostMemory> timingCacheHostData{timing_cache->serialize()};

            if (timingCacheHostData == nullptr)
            {
                CV_Error(Error::StsError, cv::String("TensorRT backend: could not serialize timing cache:"+trt_model_filename));
                return;
            }
            saveTimingCacheFile(timeCachingPath, timingCacheHostData.get());
        }
    }

    {
        AutoLock lock(mutex);
        context_ = Ptr<::nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    }

    if (!context_)
    {
        CV_Error(Error::StsError, "TensorRT backend: Failed to create context!");
        return;
    }

    // parsing model i/o name, dim, data type.
    int ioNb = engine_->getNbIOTensors();

    for (int i = 0; i < ioNb; i++)
    {
        bool isInput = engine_->bindingIsInput(i);
        std::string name = std::string(engine_->getBindingName(i));
        ::nvinfer1::Dims dim = engine_->getBindingDimensions(i);
        MatShape shape = convertDim2Shape(dim);

        if (isInput)
        {
            inputCount++;
            inputNamesString.push_back(name);
            inputMatShape.push_back(shape);
            input_idxs.push_back(i);
            inputMatShapeTrt.push_back(dim);
        }
        else
        {
            outputCount++;
            outputNamesString.push_back(name);
            outputMatShape.push_back(shape);
            output_idxs.push_back(i);
            outputMatShapeTrt.push_back(dim);
        }
    }

    this->allocMem();
}

void ImplTensorRT::allocMem()
{
    int allIONb = inputCount + outputCount;

    bufferListDevice.resize(allIONb, nullptr);
    bufferListHost.resize(allIONb, {AutoBuffer<uchar>(), 0});

    for (int i = 0; i < input_idxs.size(); i++)
    {
        int idx = input_idxs[i];
        CV_Assert(idx >=0 && idx < allIONb);
        int cvType = convertTrt2CVType(engine_->getBindingDataType(idx));
        size_t dataSize = CV_ELEM_SIZE1(cvType) * total(inputMatShape[i]);
        bufferListHost[idx].first.allocate(dataSize);
        CV_Assert(bufferListHost[idx].first.data());
        bufferListHost[idx].second = dataSize;
        cudaMalloc(&bufferListDevice[idx], dataSize);
        CV_Assert(bufferListDevice[idx]);

        context_->setTensorAddress(inputNamesString[i].c_str(), bufferListDevice[idx]);
    }

    for (int i = 0; i < output_idxs.size(); i++)
    {
        int idx = output_idxs[i];
        CV_Assert(idx >=0 && idx < allIONb);
        int cvType = convertTrt2CVType(engine_->getBindingDataType(idx));
        size_t dataSize = CV_ELEM_SIZE1(cvType) * total(outputMatShape[i]);
        bufferListHost[idx].first.allocate(dataSize);
        bufferListHost[idx].second = dataSize;
        CV_Assert(bufferListHost[idx].first.data());
        cudaMalloc(&bufferListDevice[idx], dataSize);
        CV_Assert(bufferListDevice[idx]);

        context_->setTensorAddress(outputNamesString[i].c_str(), bufferListDevice[idx]);
    }
}

void ImplTensorRT::setInput(InputArray blob_, const String& name)
{
    Mat blob = blob_.getMat();

    int indexRes = getInputIndex(name);
    CV_Assert(indexRes != -1 && indexRes < inputCount && "TensorRT Backend: indexRes error called in setInput()!");

    // Trt model has two type of input: shape tensor and execution tensor.
    bool isShapeTensor = engine_->isShapeInferenceIO(name.c_str());
    CV_Assert(!isShapeTensor && "TensorRT Backend: Unsupported right now!");

    std::vector<int> tensorShape = inputMatShape[indexRes];
    std::vector<int> matShape = shape(blob);
    size_t totalValueT = total(tensorShape);
    size_t totalValueM = blob.total();

    if (totalValueT != totalValueM || tensorShape.size() != matShape.size())
    {
        if (tensorShape.size() == 4 && matShape.size() == 4)
        {
            CV_LOG_WARNING(NULL, cv::format("The given input shape is [%d x %d x %d x %d], and the expected input shape is [%d x %d x %d x %d]", matShape[0], matShape[1], matShape[2], matShape[3], tensorShape[0], tensorShape[1], tensorShape[2], tensorShape[3]));
        }

        if (tensorShape.size() == 3 && matShape.size() == 3)
            CV_LOG_WARNING(NULL, cv::format("The given input shape is [%d x %d x %d], and the expected input shape is [%d x %d x %d]", matShape[0], matShape[1], matShape[2], tensorShape[0], tensorShape[1], tensorShape[2]));
        CV_Error(CV_StsError, "The input shape dose not match the expacted input shape! \n");
    }

    int bindingIdx = input_idxs[indexRes];// idx in tensorrt
    int cvType = convertTrt2CVType(engine_->getBindingDataType(bindingIdx));
    CV_Assert(cvType == blob.depth() && "The input Mat type is not match the Trt Tensor type!");

    memcpy(bufferListHost[bindingIdx].first.data(), blob.data, bufferListHost[bindingIdx].second);
}

void ImplTensorRT::forward(OutputArrayOfArrays outputBlobs, const std::vector<String>& outBlobNames)
{
    CV_Assert(!empty());
    CV_Assert(outputBlobs.isMatVector());
    // Output depth can be CV_32F or CV_8S
    std::vector<Mat>& outputvec = *(std::vector<Mat>*)outputBlobs.getObj();
    int outSize = outBlobNames.size();
    CV_Assert(outSize <= outputCount && "OpenCV DNN forward() error, expected value exceeds existing value.");

    std::vector<int> outputIdx(outSize, -1);

    for(int i = 0; i < outSize; i++)
    {
        int res = getOutputIndex(outBlobNames[i]);

        if (res == -1) // un-found output name
        {
            CV_Error(Error::StsBadArg, cv::String("Can not found the expacted output name = " + outputNamesString[i] + "!"));
            return;
        }

        outputIdx[i] = res;
    }

    for (int i = 0; i < inputCount; i++)
    {
        cudaMemcpyAsync(bufferListDevice[input_idxs[i]], bufferListHost[input_idxs[i]].first.data(),
            bufferListHost[input_idxs[i]].second, cudaMemcpyHostToDevice, stream_);
    }

    context_->enqueueV3(stream_);

    for (int i = 0; i < outputCount; i++)
    {
        cudaMemcpyAsync(bufferListHost[output_idxs[i]].first.data(), bufferListDevice[output_idxs[i]],
            bufferListHost[output_idxs[i]].second, cudaMemcpyDeviceToHost, stream_);
    }

    tensors2Mats(outputIdx, outputvec);
    cudaStreamSynchronize(stream_);
}

}
CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif