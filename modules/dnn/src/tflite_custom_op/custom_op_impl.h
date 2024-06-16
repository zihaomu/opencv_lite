//
// Created by mzh on 2024/3/6.
//

#ifndef OPENCV_CUSTOM_OP_IMPL_H
#define OPENCV_CUSTOM_OP_IMPL_H
#include "../precomp.hpp"

#include "vector"
namespace cv {
namespace dnn {
namespace mp_op {

struct float2
{
    float2();
    float2(float a, float b);
    float operator [] (int i) const;
    float x;
    float y;
};

struct float3
{
    float3();
    float3(float a, float b, float c);
    float operator [] (int i) const;
    float x;
    float y;
    float z;
};

struct float4
{
    float4();
    float4(float a, float b, float c, float d);
    float operator [] (int i) const;
    float x;
    float y;
    float z;
    float w;
};

struct int2
{
    int2();
    int2(int a, int b);
    int operator [] (int i) const;
    std::vector<int> data;
};

struct int3
{
    int3();
    int3(int a, int b, int c);
    int operator [] (int i) const;
    std::vector<int> data;
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

}}}

#endif //OPENCV_CUSTOM_OP_IMPL_H
