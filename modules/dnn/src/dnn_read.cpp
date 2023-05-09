// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv-onnx.pb.h"
#include <opencv2/core/utils/logger.hpp>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// TODO, change the config to feet the need of readNet.
Net readNet(const String& _model)
{
    String model = _model;
    const std::string modelExt = model.substr(model.rfind('.') + 1);

    if (modelExt == "onnx")
    {
        return readNetFromONNX(model);
    }
    CV_Error(Error::StsError, "Cannot determine an origin framework of files: " + model);
}

Net readNetFromONNX(const String& _model)
{
    Net net = Net();
    net.readNet(_model);
    return net;
}


void releaseONNXTensor(opencv_onnx::TensorProto& tensor_proto)
{
    if (!tensor_proto.raw_data().empty()) {
        delete tensor_proto.release_raw_data();
    }
}

template<typename T1, typename T2>
void convertInt64ToInt32(const T1& src, T2& dst, int size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = saturate_cast<int32_t>(src[i]);
    }
}

void convertBoolToInt8(const bool* src, int8_t* dst, int size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = saturate_cast<int8_t>(src[i]);
    }
}

Mat getMatFromTensor(const opencv_onnx::TensorProto& tensor_proto)
{
    const char* raw_data = tensor_proto.raw_data().c_str();
    CV_Assert(!(!raw_data && tensor_proto.float_data().empty() &&
        tensor_proto.double_data().empty() && tensor_proto.int64_data().empty() &&
        tensor_proto.int32_data().empty()));

    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    Mat blob;

    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++)
    {
        sizes.push_back(tensor_proto.dims(i));
    }

    // fix scalar or empty.
    if (sizes.empty())
    {
        sizes.assign(1, 1);
    }

    if (datatype == opencv_onnx::TensorProto_DataType_FLOAT)
    {
        if (!tensor_proto.float_data().empty())
        {
            const ::google::protobuf::RepeatedField<float> field = tensor_proto.float_data();
            Mat(sizes, CV_32FC1, (void*)field.data()).copyTo(blob);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_FLOAT16)
    {
        // ONNX saves float 16 data in two format: int32 and raw_data.
        // Link: https://github.com/onnx/onnx/issues/4460#issuecomment-1224373746
        if (!tensor_proto.int32_data().empty())
        {
            int offset = 0;
#ifdef WORDS_BIGENDIAN
            offset = 1;
#endif
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();

            AutoBuffer<float16_t, 16> aligned_val;
            size_t sz = tensor_proto.int32_data().size();
            aligned_val.allocate(sz);
            float16_t* bufPtr = aligned_val.data();

            float16_t *fp16Ptr = (float16_t *)field.data();
            for (int i = 0; i < sz; i++)
            {
                bufPtr[i] = fp16Ptr[i*2 + offset];
            }
            Mat(sizes, CV_16FC1, bufPtr).copyTo(blob);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required.
            AutoBuffer<float16_t, 16> aligned_val;
            if (!isAligned<sizeof(float16_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(float16_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (char*)aligned_val.data();
            }
#endif
            Mat(sizes, CV_16FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();

        char* val = nullptr;
        if (!field.empty())
        {
            val = (char *)field.data();
        }
        else
        {
            val = const_cast<char*>(tensor_proto.raw_data().c_str());
        }

#if CV_STRONG_ALIGNMENT
        // Aligned pointer is required.
        AutoBuffer<double, 16> aligned_val;
        if (!isAligned<sizeof(double)>(val))
        {
            size_t sz = tensor_proto.raw_data().size();
            aligned_val.allocate(divUp(sz, sizeof(double)));
            memcpy(aligned_val.data(), val, sz);
            val = (char*)aligned_val.data();
        }
#endif
        Mat(sizes, CV_64FC1, val).copyTo(blob);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT32)
    {
        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void*)field.data()).copyTo(blob);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32SC1, val).copyTo(blob);
        }
    }
    // Since OpenCV do not support int64, we use the raw as the input.
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_64SC1);
        int64_t* dst = blob.ptr<int64_t>();

        if (!tensor_proto.int64_data().empty())
        {
            ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = tensor_proto.int64_data();

            Mat(sizes, CV_64SC1, (void*)src.data()).copyTo(blob);
        }
        else
        {
            const char* val = tensor_proto.raw_data().c_str();
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required: https://github.com/opencv/opencv/issues/16373
            // this doesn't work: typedef int64_t CV_DECL_ALIGNED(1) unaligned_int64_t;
            AutoBuffer<int64_t, 16> aligned_val;
            if (!isAligned<sizeof(int64_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(int64_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (const char*)aligned_val.data();
            }
#endif
            char* src = const_cast<char*>(val);
            Mat(sizes, CV_64SC1, (void*)src).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_UINT64)
    {
        blob.create(sizes, CV_64UC1);
        uint64_t* dst = blob.ptr<uint64_t>();

        if (!tensor_proto.uint64_data().empty())
        {
            ::google::protobuf::RepeatedField< ::google::protobuf::uint64> src = tensor_proto.uint64_data();

            Mat(sizes, CV_64UC1, (void*)src.data()).copyTo(blob);
        }
        else
        {
            const char* val = tensor_proto.raw_data().c_str();
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required: https://github.com/opencv/opencv/issues/16373
            // this doesn't work: typedef int64_t CV_DECL_ALIGNED(1) unaligned_int64_t;
            AutoBuffer<uint64_t, 16> aligned_val;
            if (!isAligned<sizeof(uint64_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(uint64_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (const char*)aligned_val.data();
            }
#endif
            char* src = const_cast<char*>(val);
            Mat(sizes, CV_64UC1, (void*)src).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT8 ||
                datatype == opencv_onnx::TensorProto_DataType_UINT8)
    {
        // TODO : Add support for uint8 weights and acitvations. For now, converting uint8 tensors to int8.
        int depth = datatype == opencv_onnx::TensorProto_DataType_INT8 ? CV_8S : CV_8U;

        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();

            Mat(sizes, CV_32SC1, (void*)field.data()).convertTo(blob, depth);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, depth, val).convertTo(blob, depth);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_UINT16 ||
                datatype == opencv_onnx::TensorProto_DataType_INT16)
    {
        int depth = datatype == opencv_onnx::TensorProto_DataType_INT16 ? CV_16S : CV_16U;

        char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
        CV_Assert(val && "The data pointer is empty!");
        Mat(sizes, depth, (void*)val).copyTo(blob);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_BOOL)
    {
        blob.create(sizes, CV_BOOLC1);

        char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
        CV_Assert(val);
        bool * src = (bool *)val;
        bool* dst = blob.ptr<bool>();

        for (int i = 0; i < blob.total(); i++)
        {
             dst[i] = src[i];
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_UINT32)
    {
        char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
        CV_Assert(val);
        Mat(sizes, CV_32UC1, val).copyTo(blob);
    }
    else
    {
        std::string errorMsg = "Unsupported data type: " +
                               opencv_onnx::TensorProto_DataType_Name(datatype);

        if (!DNN_DIAGNOSTICS_RUN)
        {
            CV_Error(Error::StsUnsupportedFormat, errorMsg);
        }
        CV_LOG_ERROR(NULL, errorMsg);
        return blob;
    }

    if (tensor_proto.dims_size() == 0 || sizes.size() == 1)
        blob.dims = 1;  // To force 1-dimensional cv::Mat for scalars.
    return blob;
}

Mat readTensorFromONNX(const String& path)
{
    std::fstream input(path.c_str(), std::ios::in | std::ios::binary);
    if (!input)
    {
        CV_Error(Error::StsBadArg, cv::format("Can't read ONNX file: %s", path.c_str()));
    }

    opencv_onnx::TensorProto tensor_proto = opencv_onnx::TensorProto();
    if (!tensor_proto.ParseFromIstream(&input))
    {
        CV_Error(Error::StsUnsupportedFormat, cv::format("Failed to parse ONNX data: %s", path.c_str()));
    }
    Mat mat = getMatFromTensor(tensor_proto);
    releaseONNXTensor(tensor_proto);
    return mat;
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
