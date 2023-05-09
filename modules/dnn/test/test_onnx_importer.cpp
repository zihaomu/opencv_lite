// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>
namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    return findDataFile(std::string("dnn/onnx/") + filename, required);
}

// TODO speed up softmax function.
static void softmax(InputArray inblob, OutputArray outblob)
{
    const Mat input = inblob.getMat();
    MatShape inputShape = shape(input);
    outblob.create(inputShape.size(), &inputShape[0], inblob.type());

    Mat exp;
    const float max = *std::max_element(input.begin<float>(), input.end<float>());
    cv::exp((input - max), exp);
    outblob.getMat() = exp / cv::sum(exp)[0];
}

class Test_ONNX_layers : public DNNTestLayer
{
public:
    bool required;

    Test_ONNX_layers() : required(true) { }

    enum Extension
    {
        npy,
        pb
    };

    void testInputShapes(const Net& net, const std::vector<Mat>& inps)
    {
        std::vector<MatShape> inLayerShapes;
        std::vector<MatShape> outLayerShapes;

        for (int i = 0; i < inps.size(); ++i) {
            bool hasDynamicShapes = inLayerShapes[i].empty();
            if (hasDynamicShapes)
                continue;
            if (inLayerShapes[i].size() == 1) {  // 1D input
                ASSERT_EQ(shape(inLayerShapes[i][0], 1), shape(inps[i]));
            } else {
                // Compare all axes except batch dimension which is variable.
                inLayerShapes[i][0] = inps[i].size[0];
                ASSERT_EQ(inLayerShapes[i], shape(inps[i]));
            }
        }
    }

    void testONNXModels(const String& basename, const Extension ext = npy,
                        const double l1 = 0, const float lInf = 0, const bool useSoftmax = false,
                        bool checkNoFallbacks = true, int numInps = 1)
    {
        String onnxmodel = _tf("models/" + basename + ".onnx", required);
        std::vector<Mat> inps(numInps);
        Mat ref;
        if (ext == npy) {
            for (int i = 0; i < numInps; ++i)
                inps[i] = blobFromNPY(_tf("data/input_" + basename + (numInps > 1 ? format("_%d", i) : "") + ".npy"));
            ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));
        }
        else if (ext == pb) {
            for (int i = 0; i < numInps; ++i)
                inps[i] = readTensorFromONNX(_tf("data/input_" + basename + (numInps > 1 ? format("_%d", i) : "") + ".pb"));
            ref = readTensorFromONNX(_tf("data/output_" + basename + ".pb"));
        }
        else
            CV_Error(Error::StsUnsupportedFormat, "Unsupported extension");

        checkBackend(&inps[0], &ref);
        Net net = readNetFromONNX(onnxmodel);

        std::vector<String> inputNames = net.getInputName();
        std::vector<String> outputNames = net.getOutputName();

        // TODO add CUDA ORT supported..
//        net.setPreferableBackend(backend);
//        net.setPreferableTarget(target);

        CV_Assert(inputNames.size() == inps.size());
        for (int i = 0; i < numInps; ++i)
            net.setInput(inps[i], inputNames[i]);
        Mat out = net.forward(outputNames[0]);

        if (useSoftmax)
        {
            softmax(out, out);
            softmax(ref, ref);
        }

        normAssert(out, ref, "", l1 ? l1 : default_l1, lInf ? lInf : default_lInf);
    }
};

TEST_P(Test_ONNX_layers, InstanceNorm)
{
    testONNXModels("instancenorm", npy);
}

TEST_P(Test_ONNX_layers, MaxPooling)
{
    testONNXModels("maxpooling", npy, 0, 0, false, false);
}
TEST_P(Test_ONNX_layers, MaxPooling_2)
{
    testONNXModels("two_maxpooling", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testONNXModels("convolution");
    testONNXModels("conv_asymmetric_pads");
}

TEST_P(Test_ONNX_layers, Convolution_variable_weight)
{
    String basename = "conv_variable_w";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    for (int i = 0; i < 2; i++)
    {
        Mat input = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_0.npy"));
        Mat weights = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_1.npy"));
        Mat ref = blobFromNPY(_tf("data/output_" + basename + format("_%d", i) + ".npy"));

        net.setInput(input, "0");
        net.setInput(weights, "1");

        Mat out = net.forward("2");
        normAssert(ref, out, "", default_l1, default_lInf);
    }
}

TEST_P(Test_ONNX_layers, Convolution_variable_weight_bias)
{
    String basename = "conv_variable_wb";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    for (int i = 0; i < 2; i++)
    {
        Mat input = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_0.npy"));
        Mat weights = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_1.npy"));
        Mat bias = blobFromNPY(_tf("data/input_" + basename + format("_%d", i) + "_2.npy"));
        Mat ref = blobFromNPY(_tf("data/output_" + basename + format("_%d", i) + ".npy"));

        net.setInput(input, "0");
        net.setInput(weights, "1");
        net.setInput(bias, "bias");

        Mat out = net.forward();
        normAssert(ref, out, "", default_l1, default_lInf);
    }
}

TEST_P(Test_ONNX_layers, Gather)
{
    testONNXModels("gather", npy, 0, 0, false, false);
}

//TEST_P(Test_ONNX_layers, Gather_Scalar)
//{
//    testONNXModels("gather_scalar", npy, 0, 0, false, false); // onnxruntime can not run this test.
//}

TEST_P(Test_ONNX_layers, GatherMulti)
{
    testONNXModels("gather_multi", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Convolution3D)
{
    testONNXModels("conv3d");
}

TEST_P(Test_ONNX_layers, Convolution3D_bias)
{
    testONNXModels("conv3d_bias");
}

TEST_P(Test_ONNX_layers, Two_convolution)
{
#if defined(INF_ENGINE_RELEASE)
    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_MYRIAD
        && getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X
    )
        applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD_X, CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
#endif
    // Reference output values are in range [-0.855, 0.611]
    testONNXModels("two_convolution");
}

TEST_P(Test_ONNX_layers, Deconvolution)
{
//    testONNXModels("deconvolution", npy, 0, 0, false, false);
//    testONNXModels("two_deconvolution", npy, 0, 0, false, false);
//    testONNXModels("deconvolution_group", npy, 0, 0, false, false);
    testONNXModels("deconvolution_output_shape", npy, 0, 0, false, false); // the output is dynamic.
//    testONNXModels("deconv_adjpad_2d", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Deconvolution3D)
{
    testONNXModels("deconv3d");
}

TEST_P(Test_ONNX_layers, Deconvolution3D_bias)
{
    testONNXModels("deconv3d_bias");
}

TEST_P(Test_ONNX_layers, Deconvolution3D_pad)
{
    testONNXModels("deconv3d_pad");
}

TEST_P(Test_ONNX_layers, Deconvolution3D_adjpad)
{
    testONNXModels("deconv3d_adjpad");
}

// opset 6
//TEST_P(Test_ONNX_layers, Dropout)
//{
//    testONNXModels("dropout");
//}

// opset 6
//TEST_P(Test_ONNX_layers, Linear)
//{
//    testONNXModels("linear");
//}

TEST_P(Test_ONNX_layers, ReLU)
{
    testONNXModels("ReLU");
}

TEST_P(Test_ONNX_layers, PReLU)
{
    testONNXModels("PReLU_slope");
}

TEST_P(Test_ONNX_layers, Clip)
{
    testONNXModels("clip", npy);
}

TEST_P(Test_ONNX_layers, Clip_init)
{
    testONNXModels("clip_init_min_max");
    testONNXModels("clip_init_min");
    testONNXModels("clip_init_max");
}

TEST_P(Test_ONNX_layers, Shape)
{
    testONNXModels("shape_of_constant");
}

TEST_P(Test_ONNX_layers, ReduceMean)
{
    testONNXModels("reduce_mean");
    // onnxruntime can not run it.
//    testONNXModels("reduce_mean_axis1");
//    testONNXModels("reduce_mean_axis2");
}

TEST_P(Test_ONNX_layers, ReduceSum)
{
//    testONNXModels("reduce_sum");
    testONNXModels("reduce_sum_axis_dynamic_batch");
}

TEST_P(Test_ONNX_layers, ReduceMax)
{
    testONNXModels("reduce_max");
}

TEST_P(Test_ONNX_layers, ReduceMax_axis_0)
{
    testONNXModels("reduce_max_axis_0");
}

TEST_P(Test_ONNX_layers, ReduceMax_axis_1)
{
    testONNXModels("reduce_max_axis_1");
}

TEST_P(Test_ONNX_layers, Min)
{
    testONNXModels("min", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, ArgLayer)
{
    testONNXModels("argmax");
    testONNXModels("argmin");
}

TEST_P(Test_ONNX_layers, Scale)
{
    testONNXModels("scale");
}

TEST_P(Test_ONNX_layers, Scale_broadcast)
{
    testONNXModels("scale_broadcast", npy, 0, 0, false, true, 3);
}

TEST_P(Test_ONNX_layers, Scale_broadcast_mid)
{
    testONNXModels("scale_broadcast_mid", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, ReduceMean3D)
{
    testONNXModels("reduce_mean3d");
}

TEST_P(Test_ONNX_layers, MaxPooling_Sigmoid)
{
    testONNXModels("maxpooling_sigmoid");
}

TEST_P(Test_ONNX_layers, Cast)
{
    testONNXModels("cast");
}

TEST_P(Test_ONNX_layers, Power)
{
    testONNXModels("pow2", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Exp)
{
    testONNXModels("exp");
}

TEST_P(Test_ONNX_layers, Elementwise_Ceil)
{
    testONNXModels("ceil");
}

TEST_P(Test_ONNX_layers, Elementwise_Floor)
{
    testONNXModels("floor");
}

TEST_P(Test_ONNX_layers, Elementwise_Log)
{
    testONNXModels("log");
}

TEST_P(Test_ONNX_layers, Elementwise_Round)
{
    testONNXModels("round");
}

TEST_P(Test_ONNX_layers, Elementwise_Sqrt)
{
    testONNXModels("sqrt");
}

//TEST_P(Test_ONNX_layers, Elementwise_not)
//{
//    testONNXModels("not");
//}

TEST_P(Test_ONNX_layers, Compare_EQ)
{
    testONNXModels("equal");
}

TEST_P(Test_ONNX_layers, Compare_GT)
{
    testONNXModels("greater");
}

TEST_P(Test_ONNX_layers, Compare_LT)
{
    testONNXModels("less");
}

TEST_P(Test_ONNX_layers, Compare_GTorEQ)
{
    testONNXModels("greater_or_equal");
}

TEST_P(Test_ONNX_layers, Compare_LEorEQ)
{
    testONNXModels("less_or_equal");
}

// onnxruntime can not run it, invalid model
//TEST_P(Test_ONNX_layers, CompareSameDims_EQ)
//{
//    testONNXModels("equal_same_dims", npy, 0, 0, false, true, 2);
//}

TEST_P(Test_ONNX_layers, CompareSameDims_GT)
{
    testONNXModels("greater_same_dims", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, CompareSameDims_LT)
{
    testONNXModels("less_same_dims", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, Concatenation)
{
    testONNXModels("concatenation");
    testONNXModels("concat_const_blobs");
}

TEST_P(Test_ONNX_layers, Eltwise3D)
{
    testONNXModels("eltwise3d");
}

// opset v6
//TEST_P(Test_ONNX_layers, AveragePooling)
//{
//    testONNXModels("average_pooling");
//}

TEST_P(Test_ONNX_layers, MaxPooling3D)
{
    testONNXModels("max_pool3d", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, AvePooling3D)
{
    testONNXModels("ave_pool3d");
}

TEST_P(Test_ONNX_layers, PoolConv3D)
{
    testONNXModels("pool_conv_3d");
}

// opset 6
//TEST_P(Test_ONNX_layers, BatchNormalization)
//{
//    testONNXModels("batch_norm");
//}

TEST_P(Test_ONNX_layers, BatchNormalization3D)
{
    testONNXModels("batch_norm_3d");
}

TEST_P(Test_ONNX_layers, BatchNormalizationUnfused)
{
    testONNXModels("frozenBatchNorm2d");
}

TEST_P(Test_ONNX_layers, BatchNormalizationSubgraph)
{
    testONNXModels("batch_norm_subgraph");
}

TEST_P(Test_ONNX_layers, NormalizeFusionSubgraph)
{
    testONNXModels("normalize_fusion");
}

TEST_P(Test_ONNX_layers, Transpose)
{
    testONNXModels("transpose");
}

// opset 6
//TEST_P(Test_ONNX_layers, Multiplication)
//{
//    testONNXModels("mul");
//}

TEST_P(Test_ONNX_layers, MatMul_2d)
{
    testONNXModels("matmul_2d");
}
TEST_P(Test_ONNX_layers, MatMul_3d)
{
    testONNXModels("matmul_3d");
}
TEST_P(Test_ONNX_layers, MatMul_4d)
{
    testONNXModels("matmul_4d");
}

TEST_P(Test_ONNX_layers, MatMul_2d_init)
{
    testONNXModels("matmul_2d_init");
}
TEST_P(Test_ONNX_layers, MatMul_3d_init)
{
    testONNXModels("matmul_3d_init");
}
TEST_P(Test_ONNX_layers, MatMul_4d_init)
{
    testONNXModels("matmul_4d_init");
}
TEST_P(Test_ONNX_layers, MatMul_init_2)
{
    testONNXModels("matmul_init_2");
}
TEST_P(Test_ONNX_layers, MatMul_init_bcast)
{
    testONNXModels("matmul_init_bcast");
}

TEST_P(Test_ONNX_layers, MatMulAdd)
{
    testONNXModels("matmul_add");
}

TEST_P(Test_ONNX_layers, Expand)
{
    testONNXModels("expand");
    testONNXModels("expand_identity");
    testONNXModels("expand_batch");
    testONNXModels("expand_channels");
    testONNXModels("expand_neg_batch");
}

TEST_P(Test_ONNX_layers, ExpandHW)
{
    testONNXModels("expand_hw");
}

// opset 6
//TEST_P(Test_ONNX_layers, Constant)
//{
//    testONNXModels("constant");
//}

TEST_P(Test_ONNX_layers, Padding)
{
    testONNXModels("padding");
}

//TEST_P(Test_ONNX_layers, Resize)
//{
//    testONNXModels("resize_nearest"); // invalid model
//    testONNXModels("resize_bilinear"); // accuracy is not the same.
//}

TEST_P(Test_ONNX_layers, ResizeUnfused)
{
    testONNXModels("upsample_unfused_torch1.2");
    testONNXModels("upsample_unfused_opset9_torch1.4");
    testONNXModels("resize_nearest_unfused_opset11_torch1.4");
//    testONNXModels("resize_nearest_unfused_opset11_torch1.3"); // resize is different.
    testONNXModels("resize_bilinear_unfused_opset11_torch1.4");
}

TEST_P(Test_ONNX_layers, ResizeUnfusedTwoInputs)
{
    testONNXModels("upsample_unfused_two_inputs_opset9_torch1.4", npy, 0, 0, false, true, 2);
    testONNXModels("upsample_unfused_two_inputs_opset11_torch1.4", npy, 0, 0, false, true, 2);
}

// opset 6
//TEST_P(Test_ONNX_layers, MultyInputs)
//{
//    testONNXModels("multy_inputs", npy, 0, 0, false, true, 2);
//}

TEST_P(Test_ONNX_layers, Broadcast)
{
    testONNXModels("channel_broadcast", npy, 0, 0, false, true, 2);
}

TEST_P(Test_ONNX_layers, DynamicResize)
{
//    testONNXModels("dynamic_resize_9", npy, 0, 0, false, true, 2); // can not be loaded.
//    testONNXModels("dynamic_resize_10", npy, 0, 0, false, true, 2); // result is not correct.
    testONNXModels("dynamic_resize_11", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_13", npy, 0, 0, false, true, 2);
//    testONNXModels("dynamic_resize_scale_9", npy, 0, 0, false, true, 2);
//    testONNXModels("dynamic_resize_scale_10", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_scale_11", npy, 0, 0, false, true, 2);
    testONNXModels("dynamic_resize_scale_13", npy, 0, 0, false, true, 2);

    testONNXModels("resize_size_opset11");
    testONNXModels("resize_size_opset13");
}

TEST_P(Test_ONNX_layers, Resize_HumanSeg)
{
    testONNXModels("resize_humanseg");
}

TEST_P(Test_ONNX_layers, Div)
{
    const String model =  _tf("models/div.onnx");
    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    // Reference output values range is -68.80928, 2.991873. So to avoid computational
    // difference for FP16 we'll perform reversed division (just swap inputs).
    Mat inp1 = blobFromNPY(_tf("data/input_div_1.npy"));
    Mat inp2 = blobFromNPY(_tf("data/input_div_0.npy"));
    Mat ref  = blobFromNPY(_tf("data/output_div.npy"));
    cv::divide(1.0, ref, ref);
    checkBackend(&inp1, &ref);

    net.setInput(inp1, "0");
    net.setInput(inp2, "1");
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);

    // NaryEltwise layer suuports only CPU for now
    testONNXModels("div_test_1x1", npy, 0, 0, false, false, 2);
}

TEST_P(Test_ONNX_layers, DynamicReshape)
{
    testONNXModels("dynamic_reshape");
    testONNXModels("dynamic_reshape_opset_11");
    testONNXModels("flatten_by_prod");
//    testONNXModels("flatten_const"); // onnxruntime can not run it.
}

TEST_P(Test_ONNX_layers, Reshape)
{
    testONNXModels("unsqueeze");
    testONNXModels("unsqueeze_opset_13");
}

TEST_P(Test_ONNX_layers, Unsqueeze_Neg_Axes)
{
    testONNXModels("unsqueeze_neg_axes");
}

TEST_P(Test_ONNX_layers, Squeeze)
{
    testONNXModels("squeeze");
    testONNXModels("squeeze_axes_op13");
}

TEST_P(Test_ONNX_layers, ReduceL2)
{
    testONNXModels("reduceL2");
    testONNXModels("reduceL2_subgraph");
    testONNXModels("reduceL2_subgraph_2");
    testONNXModels("reduceL2_subgraph2_2");
}

TEST_P(Test_ONNX_layers, Split)
{
    testONNXModels("split_1");
//    testONNXModels("split_2");
//    testONNXModels("split_3");
//    testONNXModels("split_4");
//    testONNXModels("split_neg_axis");
}

// Mul inside with 0-d tensor, output should be A x 1, but is 1 x A. PR #22652
TEST_P(Test_ONNX_layers, DISABLED_Split_sizes_0d)
{
    testONNXModels("split_sizes");
}

TEST_P(Test_ONNX_layers, Slice)
{
    testONNXModels("slice");
    testONNXModels("slice_neg_starts");
    testONNXModels("slice_opset_11");
    testONNXModels("slice_neg_steps", pb);
}

TEST_P(Test_ONNX_layers, Slice_Steps_2DInput)
{
    testONNXModels("slice_opset_11_steps_2d");
}

TEST_P(Test_ONNX_layers, Slice_Steps_3DInput)
{
    testONNXModels("slice_opset_11_steps_3d");
}

TEST_P(Test_ONNX_layers, Slice_Steps_4DInput)
{
    testONNXModels("slice_opset_11_steps_4d");
}

TEST_P(Test_ONNX_layers, Slice_Steps_5DInput)
{
    testONNXModels("slice_opset_11_steps_5d");
}

TEST_P(Test_ONNX_layers, Slice_Nonseq_Axes)
{
    testONNXModels("slice_nonseq_axes");
    testONNXModels("slice_nonseq_axes_steps");
    testONNXModels("slice_nonseq_miss_axes_steps");
}

TEST_P(Test_ONNX_layers, Slice_Neg_Axes)
{
    testONNXModels("slice_neg_axes");
    testONNXModels("slice_neg_axes_steps");
    testONNXModels("slice_neg_miss_axes_steps");
}

TEST_P(Test_ONNX_layers, Softmax)
{
    testONNXModels("softmax");
    testONNXModels("log_softmax", npy, 0, 0, false, false);
    testONNXModels("softmax_unfused");
}

TEST_P(Test_ONNX_layers, Split_EltwiseMax)
{
    testONNXModels("split_max");
}

TEST_P(Test_ONNX_layers, LSTM_Activations)
{
    testONNXModels("lstm_cntk_tanh", pb, 0, 0, false, false);
}

// disabled due to poor handling of 1-d mats
TEST_P(Test_ONNX_layers, DISABLED_LSTM)
{
    testONNXModels("lstm", npy, 0, 0, false, false);
}

// disabled due to poor handling of 1-d mats
TEST_P(Test_ONNX_layers, DISABLED_LSTM_bidirectional)
{
    testONNXModels("lstm_bidirectional", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_hidden)
{
    testONNXModels("hidden_lstm", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_hidden_bidirectional)
{
    testONNXModels("hidden_lstm_bi", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, GRU)
{
    testONNXModels("gru", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, GRU_bidirectional)
{
    testONNXModels("gru_bi", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, LSTM_cell_forward)
{
    testONNXModels("lstm_cell_forward", npy, 0, 0, false, false);
}
TEST_P(Test_ONNX_layers, LSTM_cell_bidirectional)
{
    testONNXModels("lstm_cell_bidirectional", npy, 0, 0, false, false);
}
TEST_P(Test_ONNX_layers, LSTM_cell_with_peepholes)
{
    testONNXModels("lstm_cell_with_peepholes", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, Pad2d_Unfused)
{
    testONNXModels("ReflectionPad2d");
    testONNXModels("ZeroPad2d");
}

TEST_P(Test_ONNX_layers, LinearWithConstant)
{
    testONNXModels("lin_with_constant");
}

TEST_P(Test_ONNX_layers, MatmulWithTwoInputs)
{
    testONNXModels("matmul_with_two_inputs");
}

TEST_P(Test_ONNX_layers, ResizeOpset11_Torch1_6)
{
    testONNXModels("resize_opset11_torch1.6");
}

TEST_P(Test_ONNX_layers, Mish)
{
    testONNXModels("mish");
    testONNXModels("mish_no_softplus");
}

TEST_P(Test_ONNX_layers, CalculatePads)
{
    testONNXModels("calc_pads");
}

TEST_P(Test_ONNX_layers, Conv1d)
{
    testONNXModels("conv1d");
}

TEST_P(Test_ONNX_layers, Conv1d_bias)
{
    testONNXModels("conv1d_bias");
}

TEST_P(Test_ONNX_layers, Conv1d_variable_weight)
{
    String basename = "conv1d_variable_w";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));
    Mat weights = blobFromNPY(_tf("data/input_" + basename + "_1.npy"));
    Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    net.setInput(input, "0");
    net.setInput(weights, "1");

    Mat out = net.forward();
    normAssert(ref, out, "", default_l1, default_lInf);
}

TEST_P(Test_ONNX_layers, Conv1d_variable_weight_bias)
{
    String basename = "conv1d_variable_wb";
    Net net = readNetFromONNX(_tf("models/" + basename + ".onnx"));
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat input = blobFromNPY(_tf("data/input_" + basename + "_0.npy"));
    Mat weights = blobFromNPY(_tf("data/input_" + basename + "_1.npy"));
    Mat bias = blobFromNPY(_tf("data/input_" + basename + "_2.npy"));
    Mat ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));

    net.setInput(input, "0");
    net.setInput(weights, "1");
    net.setInput(bias, "bias");

    Mat out = net.forward();
    normAssert(ref, out, "", default_l1, default_lInf);
}

TEST_P(Test_ONNX_layers, GatherMultiOutput)
{
    testONNXModels("gather_multi_output", npy, 0, 0, false, false);
}

TEST_P(Test_ONNX_layers, DynamicAxes_squeeze_and_conv)
{
    testONNXModels("squeeze_and_conv_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_unsqueeze_and_conv)
{
    testONNXModels("unsqueeze_and_conv_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_gather)
{
    testONNXModels("gather_dynamic_axes", npy, 0, 0, false, false);
}

//TEST_P(Test_ONNX_layers, DynamicAxes_gather_scalar)
//{
//    testONNXModels("gather_scalar_dynamic_axes", npy, 0, 0, false, false); // onnxruntime can not run it.
//}

TEST_P(Test_ONNX_layers, DynamicAxes_slice)
{
    testONNXModels("slice_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_slice_opset_11)
{
    testONNXModels("slice_opset_11_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_resize_opset11_torch16)
{
    testONNXModels("resize_opset11_torch1.6_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_average_pooling)
{
    testONNXModels("average_pooling_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_maxpooling_sigmoid)
{
    testONNXModels("maxpooling_sigmoid_dynamic_axes");
}

TEST_P(Test_ONNX_layers, DynamicAxes_dynamic_batch)
{
    testONNXModels("dynamic_batch");
}


TEST_P(Test_ONNX_layers, MaxPool1d)
{
    testONNXModels("maxpooling_1d");
}

TEST_P(Test_ONNX_layers, MaxPoolSigmoid1d)
{
    testONNXModels("maxpooling_sigmoid_1d");
}

TEST_P(Test_ONNX_layers, MaxPool1d_Twise)
{
    testONNXModels("two_maxpooling_1d");
}

TEST_P(Test_ONNX_layers, AvePool1d)
{
    testONNXModels("average_pooling_1d");
}

TEST_P(Test_ONNX_layers, PoolConv1d)
{
    testONNXModels("pool_conv_1d");
}

TEST_P(Test_ONNX_layers, ConvResizePool1d)
{
    testONNXModels("conv_resize_pool_1d");
}

TEST_P(Test_ONNX_layers, DepthWiseAdd)
{
    testONNXModels("depthwiseconv_add");
}

TEST_P(Test_ONNX_layers, DepthStride2)
{
    testONNXModels("depthwise_stride2");
}

TEST_P(Test_ONNX_layers, SubFromConst)
{
    testONNXModels("sub_from_const1");
    testONNXModels("sub_from_const_eltwise");
    testONNXModels("sub_from_const_broadcast");
}

TEST_P(Test_ONNX_layers, DivConst)
{
    testONNXModels("div_const");
}

TEST_P(Test_ONNX_layers, Gemm)
{
    testONNXModels("gemm_no_transB");
    testONNXModels("gemm_transB_0");
    testONNXModels("gemm_first_const");
}

TEST_P(Test_ONNX_layers, Gemm_bias)
{
    testONNXModels("gemm_vector_bias");
}

TEST_P(Test_ONNX_layers, Quantized_Convolution)
{
    // The difference of QOperator and QDQ format:
    // https://onnxruntime.ai/docs/performance/quantization.html#onnx-quantization-representation-format.
    {
        SCOPED_TRACE("QOperator quantized model.");
        testONNXModels("quantized_conv_uint8_weights", npy, 0.004, 0.02);
        testONNXModels("quantized_conv_int8_weights", npy, 0.03, 0.5);
        testONNXModels("quantized_conv_per_channel_weights", npy, 0.06, 0.4);
        testONNXModels("quantized_conv_asymmetric_pads_int8_weights");
    }

    {
        SCOPED_TRACE("QDQ quantized model.");
        testONNXModels("quantized_conv_uint8_weights_qdq", npy, 0.004, 0.02);
        testONNXModels("quantized_conv_int8_weights_qdq", npy, 0.03, 0.5);
        testONNXModels("quantized_conv_per_channel_weights_qdq", npy, 0.06, 0.4);
    }
}

TEST_P(Test_ONNX_layers, Quantized_MatMul)
{
    testONNXModels("quantized_matmul_uint8_weights", npy, 0.005, 0.007);
    testONNXModels("quantized_matmul_int8_weights", npy, 0.06, 0.2);
    testONNXModels("quantized_matmul_per_channel_weights", npy, 0.06, 0.22);
}

TEST_P(Test_ONNX_layers, Quantized_Gemm)
{
    testONNXModels("quantized_gemm", npy);
}

TEST_P(Test_ONNX_layers, Quantized_MatMul_Variable_Weights)
{
    testONNXModels("quantized_matmul_variable_inputs");
}

TEST_P(Test_ONNX_layers, Quantized_Eltwise)
{
    testONNXModels("quantized_eltwise");
}

TEST_P(Test_ONNX_layers, Quantized_Eltwise_Scalar)
{
    testONNXModels("quantized_eltwise_scalar");
}

TEST_P(Test_ONNX_layers, Quantized_Eltwise_Broadcast)
{
    testONNXModels("quantized_eltwise_broadcast");
}

TEST_P(Test_ONNX_layers, Quantized_LeakyReLU)
{
    testONNXModels("quantized_leaky_relu");
}

TEST_P(Test_ONNX_layers, Quantized_Sigmoid)
{
    testONNXModels("quantized_sigmoid");
}

TEST_P(Test_ONNX_layers, Quantized_MaxPool)
{
    testONNXModels("quantized_maxpool");
}

TEST_P(Test_ONNX_layers, Quantized_AvgPool)
{
    testONNXModels("quantized_avgpool");
}

TEST_P(Test_ONNX_layers, Quantized_Split)
{
    testONNXModels("quantized_split");
}

TEST_P(Test_ONNX_layers, Quantized_Pad)
{
    testONNXModels("quantized_padding");
}

TEST_P(Test_ONNX_layers, Quantized_Reshape)
{
    testONNXModels("quantized_reshape");
}

TEST_P(Test_ONNX_layers, Quantized_Transpose)
{
    testONNXModels("quantized_transpose");
}

TEST_P(Test_ONNX_layers, Quantized_Squeeze)
{
    testONNXModels("quantized_squeeze");
}

TEST_P(Test_ONNX_layers, Quantized_Unsqueeze)
{
    testONNXModels("quantized_unsqueeze");
}

TEST_P(Test_ONNX_layers, Quantized_Resize)
{
    testONNXModels("quantized_resize_nearest");
    testONNXModels("quantized_resize_bilinear", npy, 2e-4, 0.003);
    testONNXModels("quantized_resize_bilinear_align", npy, 3e-4, 0.003);
}

TEST_P(Test_ONNX_layers, Quantized_Concat)
{
    testONNXModels("quantized_concat");
    testONNXModels("quantized_concat_const_blob");
}

TEST_P(Test_ONNX_layers, Quantized_Constant)
{
    testONNXModels("quantized_constant", npy, 0.002, 0.008);
}

TEST_P(Test_ONNX_layers, OutputRegistration)
{
    testONNXModels("output_registration", npy, 0, 0, false, true, 2);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_ONNX_layers, dnnBackendsAndTargets());

class Test_ONNX_nets : public Test_ONNX_layers
{
public:
    Test_ONNX_nets() { required = false; }
};

TEST_P(Test_ONNX_nets, Alexnet)
{
    const String model =  _tf("models/alexnet.onnx", false);

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = imread(_tf("../grace_hopper_227.png"));
    Mat ref = blobFromNPY(_tf("../caffe_alexnet_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(blobFromImage(inp, 1.0f, Size(224, 224), Scalar(), false));
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, Squeezenet)
{
    testONNXModels("squeezenet", pb);
}

//TEST_P(Test_ONNX_nets, Googlenet)
//{
//    const String model = _tf("models/googlenet.onnx", false);
//
//    Net net = readNetFromONNX(model);
//    ASSERT_FALSE(net.empty());
//
//    net.setPreferableBackend(backend);
//    net.setPreferableTarget(target);
//
//    std::vector<Mat> images;
//    images.push_back( imread(_tf("../googlenet_0.png")) ); // The 2 batch size can not be supported by 1.
//    images.push_back( imread(_tf("../googlenet_1.png")) );
//    Mat inp = blobFromImages(images, 1.0f, Size(), Scalar(), false);
//    Mat ref = blobFromNPY(_tf("../googlenet_prob.npy"));
//    checkBackend(&inp, &ref);
//
//    net.setInput(inp);
//    ASSERT_FALSE(net.empty());
//    Mat out = net.forward();
//
//    normAssert(ref, out, "", default_l1,  default_lInf);
//}

TEST_P(Test_ONNX_nets, CaffeNet)
{
    testONNXModels("caffenet", pb);
}

TEST_P(Test_ONNX_nets, RCNN_ILSVRC13)
{
    // Reference output values are in range [-4.992, -1.161]
    testONNXModels("rcnn_ilsvrc13", pb, 0.0046);
}

TEST_P(Test_ONNX_nets, VGG16_bn)
{
    applyTestTag(CV_TEST_TAG_MEMORY_6GB);  // > 2.3Gb

    testONNXModels("vgg16-bn", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, ZFNet)
{
    applyTestTag(CV_TEST_TAG_MEMORY_2GB);
    testONNXModels("zfnet512", pb);
}

TEST_P(Test_ONNX_nets, ResNet18v1)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    // output range: [-16; 22], after Softmax [0, 0.51]
    testONNXModels("resnet18v1", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, ResNet50v1)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    // output range: [-67; 75], after Softmax [0, 0.98]
    testONNXModels("resnet50v1", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, ResNet50_Int8)
{
    testONNXModels("resnet50_int8", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, ResNet101_DUC_HDC)
{
    applyTestTag(CV_TEST_TAG_VERYLONG);

    testONNXModels("resnet101_duc_hdc", pb);
}

// invalid model
//TEST_P(Test_ONNX_nets, TinyYolov2)
//{
//    applyTestTag(CV_TEST_TAG_MEMORY_512MB);
//
//    if (cvtest::skipUnstableTests)
//        throw SkipTestException("Skip unstable test");
//
//    // output range: [-11; 8]
//    double l1 =  default_l1, lInf = default_lInf;
//    if (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_MYRIAD)
//    {
//        l1 = 0.02;
//        lInf = 0.2;
//    }
//    else if (target == DNN_TARGET_CUDA_FP16)
//    {
//        l1 = 0.018;
//        lInf = 0.16;
//    }
//
//    testONNXModels("tiny_yolo2", pb, l1, lInf);
//}

TEST_P(Test_ONNX_nets, CNN_MNIST)
{
    // output range: [-1952; 6574], after Softmax [0; 1]
    testONNXModels("cnn_mnist", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, MobileNet_v2)
{
    // output range: [-166; 317], after Softmax [0; 1]
    testONNXModels("mobilenetv2", pb, default_l1, default_lInf, true);
}

// the model is fp16, but the input is fp32.
//TEST_P(Test_ONNX_nets, MobileNet_v2_FP16)
//{
//    testONNXModels("mobilenetv2_fp16", npy, default_l1, default_lInf, true);
//}

// Name:'stage1_unit1_bn1' Status Message: Invalid input scale: NumDimensions() != 3
//TEST_P(Test_ONNX_nets, LResNet100E_IR)
//{
//    double l1 = default_l1, lInf = default_lInf;
//    // output range: [-3; 3]
//    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
//    {
//        l1 = 0.009;
//        lInf = 0.035;
//    }
//    else if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target == DNN_TARGET_CPU)
//    {
//        l1 = 4.6e-5;
//        lInf = 1.9e-4;
//    }
//    else if (target == DNN_TARGET_CUDA_FP16)
//    {
//        l1 = 0.009;
//        lInf = 0.04;
//    }
//    testONNXModels("LResNet100E_IR", pb, l1, lInf);
//}

TEST_P(Test_ONNX_nets, Emotion_ferplus)
{
    double l1 = default_l1;
    double lInf = default_lInf;

    testONNXModels("emotion_ferplus", pb, l1, lInf);
}

TEST_P(Test_ONNX_nets, Inception_v2)
{
    testONNXModels("inception_v2", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, DenseNet121)
{
    applyTestTag(CV_TEST_TAG_MEMORY_512MB);

    // output range: [-87; 138], after Softmax [0; 1]
    testONNXModels("densenet121", pb, default_l1, default_lInf, true);
}

TEST_P(Test_ONNX_nets, Inception_v1)
{
    testONNXModels("inception_v1", pb);
}

TEST_P(Test_ONNX_nets, Shufflenet)
{
    testONNXModels("shufflenet", pb);
}

//TEST_P(Test_ONNX_nets, Resnet34_kinetics)
//{
//    applyTestTag(CV_TEST_TAG_DEBUG_VERYLONG);
//    if (backend == DNN_BACKEND_OPENCV && target != DNN_TARGET_CPU)
//        throw SkipTestException("Only CPU is supported");  // FIXIT use tags
//
//    String onnxmodel = findDataFile("dnn/resnet-34_kinetics.onnx", false);
//    Mat image0 = imread(findDataFile("dnn/dog416.png"));
//    Mat image1 = imread(findDataFile("dnn/street.png"));
//
//    Mat ref0 = blobFromNPY(_tf("data/output_kinetics0.npy"));
//    Mat ref1 = blobFromNPY(_tf("data/output_kinetics1.npy"));
//
//    std::vector<Mat> images_0(16, image0);
//    std::vector<Mat> images_1(16, image1);
//    Mat blob0 = blobFromImages(images_0, 1.0, Size(112, 112), Scalar(114.7748, 107.7354, 99.4750), true, true);
//    Mat blob1 = blobFromImages(images_1, 1.0, Size(112, 112), Scalar(114.7748, 107.7354, 99.4750), true, true);
//
//    Net permute;
//    LayerParams lp;
//    int order[] = {1, 0, 2, 3};
//    lp.set("order", DictValue::arrayInt<int*>(&order[0], 4));
//    permute.addLayerToPrev("perm", "Permute", lp);
//
//    permute.setPreferableBackend(backend);
//    permute.setPreferableTarget(target);
//
//    permute.setInput(blob0);
//    Mat input0 = permute.forward().clone();
//
//    permute.setInput(blob1);
//    Mat input1 = permute.forward().clone();
//
//    int dims[] = {1, 3, 16, 112, 112};
//    input0 = input0.reshape(0, 5, &dims[0]);
//    input1 = input1.reshape(0, 5, &dims[0]);
//
//    Net net = readNetFromONNX(onnxmodel);
//    ASSERT_FALSE(net.empty());
//    net.setPreferableBackend(backend);
//    net.setPreferableTarget(target);
//
//    // output range [-5, 11]
//    float l1 = 0.0013;
//    float lInf = 0.009;
//    if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == DNN_TARGET_OPENCL_FP16)
//    {
//        l1 = 0.02;
//        lInf = 0.07;
//    }
//    if (target == DNN_TARGET_CUDA_FP16)
//    {
//        l1 = 0.01;
//        lInf = 0.06;
//    }
//
//    testInputShapes(net, {input0});
//
//    checkBackend(&input0, &ref0);
//    net.setInput(input0);
//    Mat out = net.forward().clone();
//    normAssert(ref0, out, "", l1, lInf);
//
//    checkBackend(&input1, &ref1);
//    net.setInput(input1);
//    out = net.forward().clone();
//    normAssert(ref1, out, "", l1, lInf);
//
//    expectNoFallbacksFromIE(net);
//}

// input is not matched.
//TEST_P(Test_ONNX_layers, CumSum)
//{
//    testONNXModels("cumsum_1d_exclusive_1");
//    testONNXModels("cumsum_1d_reverse");
//    testONNXModels("cumsum_1d_exclusive_1_reverse");
//    testONNXModels("cumsum_2d_dim_1");
//    testONNXModels("cumsum_3d_dim_2");
//}

// This test is mainly to test:
//  1. identity node with constant input
//  2. limited support to range operator (all inputs are constant)
//  3. parseExpand with multiple broadcast axes
//  4. 1D mat dimension issue with the output of range operator
//TEST_P(Test_ONNX_layers, YOLOv7)
//{
//    std::string weightPath = _tf("models/yolov7_not_simplified.onnx", false);
//    std::string imgPath = _tf("../dog_orig_size.png");
//
//    Size targetSize{640, 640};
//    float conf_threshold = 0.3;
//    float iou_threshold = 0.5;
//
//    // Reference, which is collected with input size of 640x640
//    std::vector<int> refClassIds{1, 16, 7};
//    std::vector<float> refScores{0.9614331f, 0.9589417f, 0.8679074f};
//    // [x1, y1, x2, y2] x 3
//    std::vector<Rect2d> refBoxes{Rect2d(105.973236f, 150.16716f,  472.59012f, 466.48834f),
//                                  Rect2d(109.97953f,  246.17862f, 259.83676f, 600.76624f),
//                                  Rect2d(385.96185f, 83.02809f,  576.07355f,  189.82793f)};
//
//    Mat img = imread(imgPath);
//    Mat inp = blobFromImage(img, 1/255.0, targetSize, Scalar(0, 0, 0), true, false);
//
//    Net net = readNet(weightPath);
//
//    std::vector<std::string> outputname = net.getOutputName();
//    net.setInput(inp);
//    std::vector<Mat> outs;
//    net.forward(outs, outputname);
//
//    Mat preds = outs[3].reshape(1, outs[3].size[1]); // [1, 25200, 85]
//
//    // Retrieve
//    std::vector<int> classIds;
//    std::vector<float> confidences;
//    std::vector<Rect2d> boxes;
//    // each row is [cx, cy, w, h, conf_obj, conf_class1, ..., conf_class80]
//    for (int i = 0; i < preds.rows; ++i)
//    {
//        // filter out non objects
//        float obj_conf = preds.row(i).at<float>(4);
//        if (obj_conf < conf_threshold)
//            continue;
//
//        // get class id and conf
//        Mat scores = preds.row(i).colRange(5, preds.cols);
//        double conf;
//        Point maxLoc;
//        minMaxLoc(scores, 0, &conf, 0, &maxLoc);
//        conf *= obj_conf;
//        if (conf < conf_threshold)
//            continue;
//
//        // get bbox coords
//        float* det = preds.ptr<float>(i);
//        double cx = det[0];
//        double cy = det[1];
//        double w = det[2];
//        double h = det[3];
//        // [x1, y1, x2, y2]
//        boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
//                                cx + 0.5 * w, cy + 0.5 * h));
//        classIds.push_back(maxLoc.x);
//        confidences.push_back(conf);
//    }
//
//    // NMS
//    std::vector<int> keep_idx;
//    NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);
//
//    std::vector<int> keep_classIds;
//    std::vector<float> keep_confidences;
//    std::vector<Rect2d> keep_boxes;
//    for (auto i : keep_idx)
//    {
//        keep_classIds.push_back(classIds[i]);
//        keep_confidences.push_back(confidences[i]);
//        keep_boxes.push_back(boxes[i]);
//    }
//
//    normAssertDetections(refClassIds, refScores, refBoxes, keep_classIds, keep_confidences, keep_boxes);
//}

TEST_P(Test_ONNX_layers, Tile)
{
    testONNXModels("tile", pb);
}

TEST_P(Test_ONNX_layers, LayerNorm)
{
    testONNXModels("test_layer_normalization_2d_axis0", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_2d_axis1", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_2d_axis_negative_1", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_2d_axis_negative_2", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_3d_axis0_epsilon", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_3d_axis1_epsilon", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_3d_axis2_epsilon", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_3d_axis_negative_1_epsilon", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_3d_axis_negative_2_epsilon", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_3d_axis_negative_3_epsilon", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis0", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis1", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis2", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis3", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis_negative_1", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis_negative_2", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis_negative_3", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_4d_axis_negative_4", pb, 0, 0, false, true, 3);
    testONNXModels("test_layer_normalization_default_axis", pb, 0, 0, false, true, 3);
}

// for testing graph simplification
TEST_P(Test_ONNX_layers, LayerNormExpanded)
{
    testONNXModels("layer_norm_expanded");
    testONNXModels("layer_norm_expanded_with_initializers");
}

TEST_P(Test_ONNX_layers, Gelu)
{
    testONNXModels("gelu");
    testONNXModels("gelu_approximation");
}

//TEST_P(Test_ONNX_layers, OpenAI_CLIP_head)
//{
//    testONNXModels("clip-vit-base-head");
//}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
