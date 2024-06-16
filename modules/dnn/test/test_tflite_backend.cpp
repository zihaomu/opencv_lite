// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
Test for TFLite models loading
*/

#include "test_precomp.hpp"
#include "npy_blob.hpp"

#include <opencv2/dnn/utils/debug_utils.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#ifdef OPENCV_TEST_DNN_TFLITE

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::dnn;

class Test_TFLite : public DNNTestLayer {
public:
    void testModel(Net& net, const std::string& modelName, const Mat& input, double l1 = 0, double lInf = 0);
    void testModel(const std::string& modelName, const Mat& input, double l1 = 0, double lInf = 0);
    void testModel(const std::string& modelName, const Size& inpSize, double l1 = 0, double lInf = 0);
    void testLayer(const std::string& modelName, double l1 = 0, double lInf = 0);
};

void testInputShapes(const Net& net, const std::vector<Mat>& inps) {
    std::vector<MatShape> inLayerShapes = net.getInputShape();
    std::vector<MatShape> outLayerShapes = net.getOutputShape();

    ASSERT_EQ(inLayerShapes.size(), inps.size());

    for (int i = 0; i < inps.size(); ++i) {
        ASSERT_EQ(inLayerShapes[i], shape(inps[i]));
    }
}

void Test_TFLite::testModel(Net& net, const std::string& modelName, const Mat& input, double l1, double lInf)
{
    l1 = l1 ? l1 : default_l1;
    lInf = lInf ? lInf : default_lInf;

    testInputShapes(net, {input});
    net.setInput(input);

    std::vector<String> outNames = net.getOutputName();

    std::vector<Mat> outs;
    net.forward(outs, outNames);

    ASSERT_EQ(outs.size(), outNames.size());
    for (int i = 0; i < outNames.size(); ++i) {
        Mat ref = blobFromNPY(findDataFile(format("dnn/tflite/%s_out_%s.npy", modelName.c_str(), outNames[i].c_str())));

        normAssert(ref.reshape(1, 1), outs[i].reshape(1, 1), outNames[i].c_str(), l1, lInf);
    }
}

void Test_TFLite::testModel(const std::string& modelName, const Mat& input, double l1, double lInf)
{
    Net net = readNet(findDataFile("dnn/tflite/" + modelName + ".tflite", false));
    testModel(net, modelName, input, l1, lInf);
}

void Test_TFLite::testModel(const std::string& modelName, const Size& inpSize, double l1, double lInf)
{
    Mat input = imread(findDataFile("cv/shared/lena.png"));
    CV_Assert(!input.empty());
    input = blobFromImage(input, 1.0 / 255, inpSize, 0, true);
    Mat inputT;
    transposeND(input, {0, 2, 3, 1}, inputT);
    testModel(modelName, inputT, l1, lInf);
}

void Test_TFLite::testLayer(const std::string& modelName, double l1, double lInf)
{
    Mat inp = blobFromNPY(findDataFile("dnn/tflite/" + modelName + "_inp.npy"));
    Net net = readNet(findDataFile("dnn/tflite/" + modelName + ".tflite"));
    testModel(net, modelName, inp, l1, lInf);
}

// https://google.github.io/mediapipe/solutions/face_mesh
TEST_P(Test_TFLite, face_landmark)
{
    double l1 = 2e-5, lInf = 2e-4;
    testModel("face_landmark", Size(192, 192), l1, lInf);
}

// https://google.github.io/mediapipe/solutions/face_detection
TEST_P(Test_TFLite, face_detection_short_range)
{
    double l1 = 0, lInf = 2e-4;
    testModel("face_detection_short_range", Size(128, 128), l1, lInf);
}

// https://google.github.io/mediapipe/solutions/selfie_segmentation
TEST_P(Test_TFLite, selfie_segmentation)
{
    double l1 = 0, lInf = 0;
    testModel("selfie_segmentation", Size(256, 256), l1, lInf);
}

TEST_P(Test_TFLite, max_unpooling)
{
    // Due Max Unpoling is a numerically unstable operation and small difference between frameworks
    // might lead to positional difference of maximal elements in the tensor, this test checks
    // behavior of Max Unpooling layer only.
    Net net = readNet(findDataFile("dnn/tflite/hair_segmentation.tflite", false));

    net.setPreferableBackend(DNN_BACKEND_GPU);
    Mat input = imread(findDataFile("cv/shared/lena.png"));
    cvtColor(input, input, COLOR_BGR2RGBA);
    input = input.mul(Scalar(1, 1, 1, 0));
    input = blobFromImage(input, 1.0 / 255);
    Mat inputT;
    transposeND(input, {0, 2, 3, 1}, inputT);
    testInputShapes(net, {inputT});
    net.setInput(input);

    Mat out = net.forward();
    // TODO: Finish the output resutl.
//    Mat img = Mat(512, 512, CV_32FC1, out.data);
//    img *= 255;
//    imwrite("img_hair.jpg", img);
}

TEST_P(Test_TFLite, MobilenetV1_128_int8)
{
    Net net = readNet(findDataFile("dnn/tflite/mobilenet_v1_0.25_128_quant.tflite", false));

    Mat img = imread(findDataFile("cv/shared/baboon.png")), imgResized;
    resize(img, imgResized, Size (128, 128));
    std::vector<int> inputShape = {1, 128, 128, 3};
    Mat blob = Mat(inputShape, CV_8U, imgResized.data);

    double l1 = 0.004, lInf = 3;
    net.setInput(blob);
    Mat out = net.forward();
    auto outName = net.getOutputName();
    Mat ref = blobFromNPY(findDataFile("dnn/tflite/mobilenet_v1_0.25_128_quant_out_0.npy"));
    normAssert(ref.reshape(1, 1), out.reshape(1, 1), outName[0].c_str(), l1, lInf);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_TFLite, dnnBackendsAndTargets());

}}  // namespace

#endif  // OPENCV_TEST_DNN_TFLITE
