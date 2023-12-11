//
//  Interp3DBufExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/Interp3DBufExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

Interp3DBufExecution::Interp3DBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend) : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    auto interp3DParam = op->main_as_Interp();
    mCordTransform[0] = interp3DParam->widthScale();
    mCordTransform[1] = interp3DParam->widthOffset();
    mCordTransform[2] = interp3DParam->heightScale();
    mCordTransform[3] = interp3DParam->heightOffset();
    mCordTransform[4] = interp3DParam->depthScale();
    mCordTransform[5] = interp3DParam->depthOffset();
    std::set<std::string> buildOptions;
    if (op->main_as_Interp()->resizeType() == 1) {
        mKernelName = "nearest3D_buf";
        mKernel                = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    } else {
        MNN_ERROR("Resize type other than nearest is not supported in Interp3DBuf, change to nearest!");
        mKernelName = "nearest3D_buf";
        mKernel                = runtime->buildKernel("interp_buf", mKernelName, buildOptions);
    }

    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernel));
}

ErrorCode Interp3DBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();

    std::vector<int> inputImageShape  = tensorShapeFormat(input); // {C/4 * H * W, N * D} for 5-D Tensor
    std::vector<int> outputImageShape = tensorShapeFormat(output);

    auto inputShape = input->shape();
    auto outputShape = output->shape();
    const int inputBatch    = inputShape[0];
    const int inputChannels = inputShape[1];
    const int inputDepth    = inputShape[2];
    const int inputHeight   = inputShape[3];
    const int inputWidth    = inputShape[4];

    const int channelBlocks = UP_DIV(inputChannels, 4);

    const int outputDepth    = outputShape[2];
    const int outputHeight   = outputShape[3];
    const int outputWidth    = outputShape[4];

    mGWS = {static_cast<uint32_t>(channelBlocks),
            static_cast<uint32_t>(outputHeight * outputWidth),
            static_cast<uint32_t>(outputDepth * inputBatch)};

    MNN_ASSERT(outputDepth > 0 && outputHeight > 0 && outputWidth > 0);

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGWS[0]);
    ret |= mKernel.setArg(idx++, mGWS[1]);
    ret |= mKernel.setArg(idx++, mGWS[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, mCordTransform[4]);
    ret |= mKernel.setArg(idx++, mCordTransform[2]);
    ret |= mKernel.setArg(idx++, mCordTransform[0]);
    ret |= mKernel.setArg(idx++, mCordTransform[5]);
    ret |= mKernel.setArg(idx++, mCordTransform[3]);
    ret |= mKernel.setArg(idx++, mCordTransform[1]);
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputDepth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputDepth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(outputWidth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(channelBlocks));
    MNN_CHECK_CL_SUCCESS(ret, "setArg Interp3DBufExecution");

    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime, mKernelName, mKernel).first;
    return NO_ERROR;

}

ErrorCode Interp3DBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start InterpBufExecution onExecute... \n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"Interp", event});
#else
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end InterpBufExecution onExecute... \n");
#endif

    return NO_ERROR;
}

class Interp3DBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~Interp3DBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new Interp3DBufExecution(inputs, op, backend);
        ;
    }
};

OpenCLCreatorRegister<Interp3DBufCreator> __Interp3DBuf_op_(OpType_Interp3D, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
