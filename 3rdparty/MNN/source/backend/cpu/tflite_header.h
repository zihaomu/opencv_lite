//
// Created by mzh on 2023/7/14.
//

#ifndef MNN_TFLITE_HEADER_H
#define MNN_TFLITE_HEADER_H

#include "vector"

struct float2
{
    float2();
    float2(float a, float b);

    float x;
    float y;
};

struct float3
{
    float3();
    float3(float a, float b, float c);
    float x;
    float y;
    float z;
};

struct float4
{
    float4();
    float4(float a, float b, float c, float d);

    float x;
    float y;
    float z;
    float w;
};

float DotProduct(const float4& l, const float4& r);

//using float2 =  cv::Point2f;
//using float3 = cv::Point3f;
//using int3 = cv::Point3i;
float2 Read3DLandmarkXY(const float* data, int idx);

float3 Read3DLandmarkXYZ(const float* data, int idx);

struct Mat3 {
    Mat3();
    Mat3(float x00, float x01, float x02, float x10, float x11, float x12,
         float x20, float x21, float x22);

    Mat3 operator*(const Mat3& other) const;

    float3 operator*(const float3& vec) const;
    float Get(int x, int y) const;
    void Set(int x, int y, float val);

    std::vector<float> data;
};

struct Mat4 {
    Mat4();
    Mat4(float x00, float x01, float x02, float x03, float x10, float x11,
         float x12, float x13, float x20, float x21, float x22, float x23,
         float x30, float x31, float x32, float x33);
    void operator*=(const Mat4& other);
    float Get(int x, int y) const;
    void Set(int x, int y, float val);

    std::vector<float> data;
};

#endif //MNN_TFLITE_HEADER_H
