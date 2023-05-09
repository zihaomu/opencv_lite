// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/imgproc.hpp>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


Mat blobFromImage(InputArray image, double scalefactor, const Size& size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImage(InputArray image, OutputArray blob, double scalefactor,
        const Size& size, const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    std::vector<Mat> images(1, image.getMat());
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
}

Mat blobFromImages(InputArrayOfArrays images, double scalefactor, Size size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImages(InputArrayOfArrays images_, OutputArray blob_, double scalefactor,
        Size size, const Scalar& mean_, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    CV_CheckType(ddepth, ddepth == CV_32F || ddepth == CV_8U, "Blob depth should be CV_32F or CV_8U");
    if (ddepth == CV_8U)
    {
        CV_CheckEQ(scalefactor, 1.0, "Scaling is not supported for CV_8U blob depth");
        CV_Assert(mean_ == Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    std::vector<Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());
    for (size_t i = 0; i < images.size(); i++)
    {
        Size imgSize = images[i].size();
        if (size == Size())
            size = imgSize;
        if (size != imgSize)
        {
            if (crop)
            {
                float resizeFactor = std::max(size.width / (float)imgSize.width,
                        size.height / (float)imgSize.height);
                resize(images[i], images[i], Size(), resizeFactor, resizeFactor, INTER_LINEAR);
                Rect crop(Point(0.5 * (images[i].cols - size.width),
                                  0.5 * (images[i].rows - size.height)),
                        size);
                images[i] = images[i](crop);
            }
            else
                resize(images[i], images[i], size, 0, 0, INTER_LINEAR);
        }
        if (images[i].depth() == CV_8U && ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);
        Scalar mean = mean_;
        if (swapRB)
            std::swap(mean[0], mean[2]);

        images[i] -= mean;
        images[i] *= scalefactor;
    }

    size_t nimages = images.size();
    Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    if (nch == 3 || nch == 4)
    {
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
        blob_.create(4, sz, ddepth);
        Mat blob = blob_.getMat();
        Mat ch[4];

        for (size_t i = 0; i < nimages; i++)
        {
            const Mat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for (int j = 0; j < nch; j++)
                ch[j] = Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, j));
            if (swapRB)
                std::swap(ch[0], ch[2]);
            split(image, ch);
        }
    }
    else
    {
        CV_Assert(nch == 1);
        int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
        blob_.create(4, sz, ddepth);
        Mat blob = blob_.getMat();

        for (size_t i = 0; i < nimages; i++)
        {
            const Mat& image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 1));
            CV_Assert(image.size() == image0.size());

            image.copyTo(Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, 0)));
        }
    }
}

void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    CV_TRACE_FUNCTION();

    // A blob is a 4 dimensional matrix in floating point precision
    // blob_[0] = batchSize = nbOfImages
    // blob_[1] = nbOfChannels
    // blob_[2] = height
    // blob_[3] = width
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    images_.create(cv::Size(1, blob_.size[0]), blob_.depth());

    std::vector<Mat> vectorOfChannels(blob_.size[1]);
    for (int n = 0; n < blob_.size[0]; ++n)
    {
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        cv::merge(vectorOfChannels, images_.getMatRef(n));
    }
}

void printMatShape(InputArray blob_)
{
    Mat blob = blob_.getMat();
    auto shapeData = shape(blob);
    std::cout<<"[";

    if (shapeData.size() >= 1)
        std::cout<<shapeData[0];

    for (int i = 1; i < shapeData.size(); i++)
    {
        std::cout<<" x "<<shapeData[i];
    }

    std::cout<<"]"<<std::endl;
}

template<typename T>
static inline void printData(T* data, const int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << *(data + i) << ", ";
    }
}

// TODO: print data like numpy.
void printMatData(InputArray blob_, int printLen)
{
    Mat blob = blob_.getMat();
    std::cout<<"blob shape = ";
    auto shapeData = shape(blob);
    auto typeData = blob.type();
    printMatShape(blob);

    std::cout<<"blob data = ";
    if (blob.total() == 0)
    {
        std::cout<<"empty.";
        return;
    }

    int len = std::min(int(blob.total()), printLen);
    if (len == -1)
        len = int(blob.total());

    switch (typeData)
    {
        case CV_8U:
        {
            uint8_t *pUint8 = (uint8_t *) blob.data;
            printData(pUint8, len);
            break;
        }
        case CV_8S:
        {
            int8_t *pInt8 = (int8_t *) blob.data;
            printData(pInt8, len);
            break;
        }
        case CV_16U:
        {
            uint16_t *pUint16 = (uint16_t *) blob.data;
            printData(pUint16, len);
            break;
        }
        case CV_16S:
        {
            int16_t *pInt16 = (int16_t *) blob.data;
            printData(pInt16, len);
            break;
        }
        case CV_32F:
        {
            float *pFloat = (float *) blob.data;
            printData(pFloat, len);
            break;
        }
        case CV_32S:
        {
            int *pInt = (int *) blob.data;
            printData(pInt, len);
            break;
        }
        case CV_64F:
        {
            double *pDouble = (double *) blob.data;
            printData(pDouble, len);
            break;
        }
        case CV_16F:
        {
            float16_t *pFloat16 = (float16_t *) blob.data;
            printData(pFloat16, len);
            break;
        }
        case CV_64S:
        {
            int64_t *pInt64 = (int64_t *) blob.data;
            printData(pInt64, len);
            break;
        }
        case CV_64U:
        {
            uint64_t *pUint64 = (uint64_t *) blob.data;
            printData(pUint64, len);
            break;
        }
        case CV_BOOL:
        {
            bool *pBool = (bool *) blob.data;
            printData(pBool, len);
            break;
        }
        case CV_32U:
        {
            uint *pUint = (uint *) blob.data;
            printData(pUint, len);
            break;
        }
        default:
            CV_Error(CV_StsError, "Unsupported data type!");
    }

    std::cout<<std::endl;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
