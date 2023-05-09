// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_TEST_COMMON_HPP__
#define __OPENCV_TEST_COMMON_HPP__

#ifdef HAVE_OPENCL
#include "opencv2/core/ocl.hpp"
#endif

#define CV_TEST_TAG_DNN_SKIP_OPENCV_BACKEND      "dnn_skip_opencv_backend"
#define CV_TEST_TAG_DNN_SKIP_CPU                 "dnn_skip_cpu"
#define CV_TEST_TAG_DNN_SKIP_CPU                 "dnn_skip_cpu"
#define CV_TEST_TAG_DNN_SKIP_ONNX_CONFORMANCE    "dnn_skip_onnx_conformance"

#ifndef CV_TEST_TAG_DNN_SKIP_IE_VERSION
#    define CV_TEST_TAG_DNN_SKIP_IE_VERSION CV_TEST_TAG_DNN_SKIP_IE
#endif


namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void PrintTo(const cv::dnn::Backend& v, std::ostream* os);
void PrintTo(const cv::dnn::Target& v, std::ostream* os);
using opencv_test::tuple;
using opencv_test::get;
void PrintTo(const tuple<cv::dnn::Backend, cv::dnn::Target> v, std::ostream* os);

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn



namespace opencv_test {

void initDNNTests();

using namespace cv::dnn;

static inline const std::string &getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

void normAssert(
        cv::InputArray ref, cv::InputArray test, const char *comment = "",
        double l1 = 0.00001, double lInf = 0.0001);

std::vector<cv::Rect2d> matToBoxes(const cv::Mat& m);

void normAssertDetections(
        const std::vector<int>& refClassIds,
        const std::vector<float>& refScores,
        const std::vector<cv::Rect2d>& refBoxes,
        const std::vector<int>& testClassIds,
        const std::vector<float>& testScores,
        const std::vector<cv::Rect2d>& testBoxes,
        const char *comment = "", double confThreshold = 0.0,
        double scores_diff = 1e-5, double boxes_iou_diff = 1e-4);

// For SSD-based object detection networks which produce output of shape 1x1xNx7
// where N is a number of detections and an every detection is represented by
// a vector [batchId, classId, confidence, left, top, right, bottom].
void normAssertDetections(
        cv::Mat ref, cv::Mat out, const char *comment = "",
        double confThreshold = 0.0, double scores_diff = 1e-5,
        double boxes_iou_diff = 1e-4);

// For text detection networks
// Curved text polygon is not supported in the current version.
// (concave polygon is invalid input to intersectConvexConvex)
void normAssertTextDetections(
        const std::vector<std::vector<Point>>& gtPolys,
        const std::vector<std::vector<Point>>& testPolys,
        const char *comment = "", double boxes_iou_diff = 1e-4);

void readFileContent(const std::string& filename, CV_OUT std::vector<char>& content);

bool validateVPUType();

testing::internal::ParamGenerator< tuple<Backend, Target> > dnnBackendsAndTargets(
        bool withInferenceEngine = true,
        bool withHalide = false,
        bool withCpuOCV = true,
        bool withVkCom = true,
        bool withCUDA = true,
        bool withNgraph = true,
        bool withWebnn = true,
        bool withCann = true
);

testing::internal::ParamGenerator< tuple<Backend, Target> > dnnBackendsAndTargetsIE();


class DNNTestLayer : public TestWithParam<tuple<Backend, Target> >
{
public:
    dnn::Backend backend;
    dnn::Target target;
    double default_l1, default_lInf;

    DNNTestLayer()
    {
        backend = (dnn::Backend)(int)get<0>(GetParam());
        target = (dnn::Target)(int)get<1>(GetParam());
        getDefaultThresholds(backend, target, &default_l1, &default_lInf);
    }

    static void getDefaultThresholds(int backend, int target, double* l1, double* lInf)
    {
        *l1 = 1e-5;
        *lInf = 1e-4;
    }

    static void checkBackend(int backend, int target, Mat* inp = 0, Mat* ref = 0)
    {
        CV_UNUSED(backend); CV_UNUSED(target); CV_UNUSED(inp); CV_UNUSED(ref);
    }

protected:
    void checkBackend(Mat* inp = 0, Mat* ref = 0)
    {
        checkBackend(backend, target, inp, ref);
    }
};

} // namespace


#endif
