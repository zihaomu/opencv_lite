// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// This file is the unit test for MNN model test!

#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <fstream>
#include <opencv2/dnn/shape_utils.hpp>
namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    return findDataFile(std::string("dnn/mnn/") + filename, required);
}

// task for this file
// load input data to Mat.
// load output to Mat
// forward net
// compare the model output and the expected result

class Test_TRT_nets : public DNNTestLayer
{
public:
    bool required;

    Test_TRT_nets() : required(true) { }

    void buildMatFromFile(Mat& blob, size_t totalElem, const String& file_path, const int dataType)
    {
        CV_Assert(blob.total()*blob.channels() == totalElem);
        std::ifstream stream(file_path.c_str());
        if (CV_MAT_DEPTH(dataType) == CV_32F)
        {
            float* dataPtr = blob.ptr<float>();
            for (size_t i = 0; i < totalElem; i++)
            {
                stream >> dataPtr[i];
            }
        }
        else if (CV_MAT_DEPTH(dataType) == CV_32S)
        {
            int* dataPtr = blob.ptr<int>();
            for (size_t i = 0; i < totalElem; i++)
            {
                stream >> dataPtr[i];
            }
        }
        else if (CV_MAT_DEPTH(dataType) == CV_8U)
        {
            uint8_t * dataPtr = blob.ptr<uint8_t>();
            for (size_t i = 0; i < totalElem; i++)
            {
                int v = 0;
                stream >> v;
                dataPtr[i] = (uint8_t)v;
            }
        }
    }

    void testMobileNet(const String& model_path, const String& input_path, int inDataType, const String& expect_path,
                       int outDataType, int precisionId = 0, int numThreads = 4)
    {
        String mnnmodel = _tf(model_path, required);
        Net net = readNetFromMNN(mnnmodel);

        // set Thread and set precision.
        net.setPreferablePrecision(precisionId);
        net.setNumThreads(numThreads);
        net.setPreferableBackend(Backend::DNN_BACKEND_GPU);

        auto inputShapes = net.getInputShape();
        auto outputShapes = net.getOutputShape();

        CV_Assert(inputShapes.size() == 1 && outputShapes.size() == 1);

        // rebuild input blob
        auto inputShape = inputShapes[0];
        Mat inputBlob = Mat(inputShape.size(), &inputShape[0], inDataType);

        String inputFile = _tf(input_path, required);
        buildMatFromFile(inputBlob, total(inputShape), inputFile, CV_8UC4);

        // rebuild expect blob
        auto outputShape = outputShapes[0];
        Mat expectBlob = Mat(outputShape.size(), &outputShape[0], outDataType);

        String expectFile = _tf(expect_path, required);
        buildMatFromFile(expectBlob, total(outputShape), expectFile, outDataType);

        net.setInput(inputBlob);
        Mat outputBlob = net.forward();

        normAssert(outputBlob, expectBlob, "", default_l1, default_lInf);
    }

    void testMobileNetFromBinary(const String& model_path, const String& input_path, int inDataType, const String& expect_path, int outDataType)
    {
        String mnnmodel = _tf(model_path, required);

        std::ifstream file(mnnmodel, std::ios::binary);

        // Find the length of the file
        file.seekg(0, std::ios::end);
        std::streamoff fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Allocate memory for the buffer
        char* buffer = new char[fileSize];

        // Read file content into the buffer
        file.read(buffer, fileSize);

        // Close the file
        file.close();

        Net net = readNetFromMNN(buffer, fileSize);

        auto inputShapes = net.getInputShape();
        auto outputShapes = net.getOutputShape();

        CV_Assert(inputShapes.size() == 1 && outputShapes.size() == 1);

        // rebuild input blob
        auto inputShape = inputShapes[0];
        Mat inputBlob = Mat(inputShape.size(), &inputShape[0], inDataType);

        String inputFile = _tf(input_path, required);
        buildMatFromFile(inputBlob, total(inputShape), inputFile, CV_8UC4);

        // rebuild expect blob
        auto outputShape = outputShapes[0];
        Mat expectBlob = Mat(outputShape.size(), &outputShape[0], outDataType);

        String expectFile = _tf(expect_path, required);
        buildMatFromFile(expectBlob, total(outputShape), expectFile, outDataType);

        net.setInput(inputBlob);
        Mat outputBlob = net.forward();

        normAssert(outputBlob, expectBlob, "", default_l1, default_lInf);
    }

    void testMobileNetWithInput(const String& model_path, const Mat& inputBlob, const String& expect_path, int outDataType)
    {
        String mnnmodel = _tf(model_path, required);
        Net net = readNetFromMNN(mnnmodel);

        auto inputShapes = net.getInputShape();
        auto outputShapes = net.getOutputShape();

        CV_Assert(inputShapes.size() == 1 && outputShapes.size() == 1);

        // rebuild expect blob
        auto outputShape = outputShapes[0];
        Mat expectBlob = Mat(outputShape.size(), &outputShape[0], outDataType);

        String expectFile = _tf(expect_path, required);
        buildMatFromFile(expectBlob, total(outputShape), expectFile, outDataType);

        net.setInput(inputBlob);
        Mat outputBlob = net.forward();

        normAssert(outputBlob, expectBlob, "", default_l1, default_lInf);
    }

    void testMobileNetFromBufferWithInput(const String& model_path, const Mat& inputBlob, const String& expect_path, int outDataType)
    {
        String mnnmodel = _tf(model_path, required);

        std::ifstream file(mnnmodel, std::ios::binary);

        // Find the length of the file
        file.seekg(0, std::ios::end);
        std::streamoff fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Allocate memory for the buffer
        char* buffer = new char[fileSize];

        // Read file content into the buffer
        file.read(buffer, fileSize);

        // Close the file
        file.close();

        Net net = readNetFromMNN(buffer, fileSize);

        auto inputShapes = net.getInputShape();
        auto outputShapes = net.getOutputShape();

        CV_Assert(inputShapes.size() == 1 && outputShapes.size() == 1);

        // rebuild expect blob
        auto outputShape = outputShapes[0];
        Mat expectBlob = Mat(outputShape.size(), &outputShape[0], outDataType);

        String expectFile = _tf(expect_path, required);
        buildMatFromFile(expectBlob, total(outputShape), expectFile, outDataType);

        net.setInput(inputBlob);
        Mat outputBlob = net.forward();

        normAssert(outputBlob, expectBlob, "", default_l1, default_lInf);
    }
};

TEST_P(Test_TRT_nets, MooTest)
{

}

INSTANTIATE_TEST_CASE_P(/**/, Test_TRT_nets, dnnBackendsAndTargets());

}} // namespace
