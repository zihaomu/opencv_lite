// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Used in accuracy and perf tests as a content of .cpp file
// Note: don't use "precomp.hpp" here
#include "opencv2/ts.hpp"
#include "opencv2/ts/ts_perf.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/dnn.hpp"
#include "test_common.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void PrintTo(const cv::dnn::Backend& v, std::ostream* os)
{
    switch (v) {
    case DNN_BACKEND_CPU: *os << "CPU"; return;
    case DNN_BACKEND_GPU: *os << "GPU"; return;
//    case DNN_BACKEND_CUDA: *os << "CUDA"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_BACKEND_UNKNOWN(" << (int)v << ")";
}

void PrintTo(const cv::dnn::Target& v, std::ostream* os)
{
    switch (v) {
    case DNN_TARGET_CPU: *os << "CPU"; return;
//    case DNN_TARGET_CUDA: *os << "CUDA"; return;
    } // don't use "default:" to emit compiler warnings
    *os << "DNN_TARGET_UNKNOWN(" << (int)v << ")";
}

void PrintTo(const tuple<cv::dnn::Backend, cv::dnn::Target> v, std::ostream* os)
{
    PrintTo(get<0>(v), os);
    *os << "/";
    PrintTo(get<1>(v), os);
}

CV__DNN_INLINE_NS_END
}} // namespace

namespace opencv_test {

testing::internal::ParamGenerator< tuple<Backend, Target> > dnnBackendsAndTargets(
        bool withInferenceEngine /*= true*/,
        bool withHalide /*= false*/,
        bool withCpuOCV /*= true*/,
        bool withVkCom /*= true*/,
        bool withCUDA /*= true*/,
        bool withNgraph /*= true*/,
        bool withWebnn /*= false*/,
        bool withCann /*= true*/
)
{
    // do nothing
    std::vector< tuple<Backend, Target> > targets;
    std::vector< Target > available;
    targets.push_back(make_tuple(DNN_BACKEND_CPU, DNN_TARGET_CPU));
    return testing::ValuesIn(targets);
}


void normAssert(
        cv::InputArray ref, cv::InputArray test, const char *comment /*= ""*/,
        double l1 /*= 0.00001*/, double lInf /*= 0.0001*/)
{
    Mat refM = ref.getMat();
    Mat testM = test.getMat();
    // Try to handle the empty output and test.
    int subDiv = refM.total();
    if (subDiv == 0)
        subDiv = 1;

    // TODO remove when 1D supported.
    if (refM.dims != testM.dims)
    {
        CV_Assert(refM.total() == testM.total());
        // 1D mat some time will be set as 2D, [x] -> [x, 1].
        // And we need to remove the last 1 dimension.
        if (testM.dims == 1 && refM.dims == 2)
        {
            refM.dims = 1;
        }
        else if (refM.dims == 1 && testM.dims == 2)
        {
            testM.dims = 1;
        }
    }

    CV_Assert(refM.dims == testM.dims && "Test: Ref data and model output dimension is not the same!");
    double normL1 = cvtest::norm(refM, testM, cv::NORM_L1) / subDiv;
    EXPECT_LE(normL1, l1) << comment << "  |ref| = " << cvtest::norm(refM, cv::NORM_INF);

    double normInf = cvtest::norm(refM, testM, cv::NORM_INF);
    EXPECT_LE(normInf, lInf) << comment << "  |ref| = " << cvtest::norm(refM, cv::NORM_INF);
}

std::vector<cv::Rect2d> matToBoxes(const cv::Mat& m)
{
    EXPECT_EQ(m.type(), CV_32FC1);
    EXPECT_EQ(m.dims, 2);
    EXPECT_EQ(m.cols, 4);

    std::vector<cv::Rect2d> boxes(m.rows);
    for (int i = 0; i < m.rows; ++i)
    {
        CV_Assert(m.row(i).isContinuous());
        const float* data = m.ptr<float>(i);
        double l = data[0], t = data[1], r = data[2], b = data[3];
        boxes[i] = cv::Rect2d(l, t, r - l, b - t);
    }
    return boxes;
}

void normAssertDetections(
        const std::vector<int>& refClassIds,
        const std::vector<float>& refScores,
        const std::vector<cv::Rect2d>& refBoxes,
        const std::vector<int>& testClassIds,
        const std::vector<float>& testScores,
        const std::vector<cv::Rect2d>& testBoxes,
        const char *comment /*= ""*/, double confThreshold /*= 0.0*/,
        double scores_diff /*= 1e-5*/, double boxes_iou_diff /*= 1e-4*/)
{
    ASSERT_FALSE(testClassIds.empty()) << "No detections";
    std::vector<bool> matchedRefBoxes(refBoxes.size(), false);
    std::vector<double> refBoxesIoUDiff(refBoxes.size(), 1.0);
    for (int i = 0; i < testBoxes.size(); ++i)
    {
        //cout << "Test[i=" << i << "]: score=" << testScores[i] << " id=" << testClassIds[i] << " box " << testBoxes[i] << endl;
        double testScore = testScores[i];
        if (testScore < confThreshold)
            continue;

        int testClassId = testClassIds[i];
        const cv::Rect2d& testBox = testBoxes[i];
        bool matched = false;
        double topIoU = 0;
        for (int j = 0; j < refBoxes.size() && !matched; ++j)
        {
            if (!matchedRefBoxes[j] && testClassId == refClassIds[j] &&
                std::abs(testScore - refScores[j]) < scores_diff)
            {
                double interArea = (testBox & refBoxes[j]).area();
                double iou = interArea / (testBox.area() + refBoxes[j].area() - interArea);
                topIoU = std::max(topIoU, iou);
                refBoxesIoUDiff[j] = std::min(refBoxesIoUDiff[j], 1.0f - iou);
                if (1.0 - iou < boxes_iou_diff)
                {
                    matched = true;
                    matchedRefBoxes[j] = true;
                }
            }
        }
        if (!matched)
        {
            std::cout << cv::format("Unmatched prediction: class %d score %f box ",
                                    testClassId, testScore) << testBox << std::endl;
            std::cout << "Highest IoU: " << topIoU << std::endl;
        }
        EXPECT_TRUE(matched) << comment;
    }

    // Check unmatched reference detections.
    for (int i = 0; i < refBoxes.size(); ++i)
    {
        if (!matchedRefBoxes[i] && refScores[i] > confThreshold)
        {
            std::cout << cv::format("Unmatched reference: class %d score %f box ",
                                    refClassIds[i], refScores[i]) << refBoxes[i]
                << " IoU diff: " << refBoxesIoUDiff[i]
                << std::endl;
            EXPECT_LE(refScores[i], confThreshold) << comment;
        }
    }
}

// For SSD-based object detection networks which produce output of shape 1x1xNx7
// where N is a number of detections and an every detection is represented by
// a vector [batchId, classId, confidence, left, top, right, bottom].
void normAssertDetections(
        cv::Mat ref, cv::Mat out, const char *comment /*= ""*/,
        double confThreshold /*= 0.0*/, double scores_diff /*= 1e-5*/,
        double boxes_iou_diff /*= 1e-4*/)
{
    CV_Assert(ref.total() % 7 == 0);
    CV_Assert(out.total() % 7 == 0);
    ref = ref.reshape(1, ref.total() / 7);
    out = out.reshape(1, out.total() / 7);

    cv::Mat refClassIds, testClassIds;
    ref.col(1).convertTo(refClassIds, CV_32SC1);
    out.col(1).convertTo(testClassIds, CV_32SC1);
    std::vector<float> refScores(ref.col(2)), testScores(out.col(2));
    std::vector<cv::Rect2d> refBoxes = matToBoxes(ref.colRange(3, 7));
    std::vector<cv::Rect2d> testBoxes = matToBoxes(out.colRange(3, 7));
    normAssertDetections(refClassIds, refScores, refBoxes, testClassIds, testScores,
                         testBoxes, comment, confThreshold, scores_diff, boxes_iou_diff);
}

// For text detection networks
// Curved text polygon is not supported in the current version.
// (concave polygon is invalid input to intersectConvexConvex)
void normAssertTextDetections(
        const std::vector<std::vector<Point>>& gtPolys,
        const std::vector<std::vector<Point>>& testPolys,
        const char *comment /*= ""*/, double boxes_iou_diff /*= 1e-4*/)
{
    std::vector<bool> matchedRefBoxes(gtPolys.size(), false);
    for (uint i = 0; i < testPolys.size(); ++i)
    {
        const std::vector<Point>& testPoly = testPolys[i];
        bool matched = false;
        double topIoU = 0;
        for (uint j = 0; j < gtPolys.size() && !matched; ++j)
        {
            if (!matchedRefBoxes[j])
            {
                std::vector<Point> intersectionPolygon;
                float intersectArea = intersectConvexConvex(testPoly, gtPolys[j], intersectionPolygon, true);
                double iou = intersectArea / (contourArea(testPoly) + contourArea(gtPolys[j]) - intersectArea);
                topIoU = std::max(topIoU, iou);
                if (1.0 - iou < boxes_iou_diff)
                {
                    matched = true;
                    matchedRefBoxes[j] = true;
                }
            }
        }
        if (!matched) {
            std::cout << cv::format("Unmatched-det:") << testPoly << std::endl;
            std::cout << "Highest IoU: " << topIoU << std::endl;
        }
        EXPECT_TRUE(matched) << comment;
    }

    // Check unmatched groundtruth.
    for (uint i = 0; i < gtPolys.size(); ++i)
    {
        if (!matchedRefBoxes[i])
        {
            std::cout << cv::format("Unmatched-gt:") << gtPolys[i] << std::endl;
        }
        EXPECT_TRUE(matchedRefBoxes[i]);
    }
}

void readFileContent(const std::string& filename, CV_OUT std::vector<char>& content)
{
    const std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(filename.c_str(), mode);
    ASSERT_TRUE(ifs.is_open());

    content.clear();

    ifs.seekg(0, std::ios::end);
    const size_t sz = ifs.tellg();
    content.resize(sz);
    ifs.seekg(0, std::ios::beg);

    ifs.read((char*)content.data(), sz);
    ASSERT_FALSE(ifs.fail());
}

void initDNNTests()
{
    const char* extraTestDataPath =
#ifdef WINRT
        NULL;
#else
        getenv("OPENCV_DNN_TEST_DATA_PATH");
#endif
    if (extraTestDataPath)
        cvtest::addDataSearchPath(extraTestDataPath);

    registerGlobalSkipTag(
        CV_TEST_TAG_DNN_SKIP_OPENCV_BACKEND,
        CV_TEST_TAG_DNN_SKIP_CPU
    );
}

} // namespace
