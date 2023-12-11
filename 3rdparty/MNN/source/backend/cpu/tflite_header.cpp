//
// Created by mzh on 2023/7/14.
//

#include "tflite_header.h"

float2::float2() : x(0.0f), y(0.0f) {}

float2::float2(float a, float b) : x(a), y(b){}

float3::float3() : x(0.0f), y(0.0f), z(0.0f) {}
float3::float3(float a, float b, float c) : x(a), y(b), z(c)
{
}

float4::float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
float4::float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d)
{
}


float DotProduct(const float4& l, const float4& r) {
    return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
}

//using float2 =  cv::Point2f;
//using float3 = cv::Point3f;
//using int3 = cv::Point3i;
float2 Read3DLandmarkXY(const float* data, int idx) {
    float2 result;
    result.x = data[idx * 3];
    result.y = data[idx * 3 + 1];
    return result;
}

float3 Read3DLandmarkXYZ(const float* data, int idx) {
    float3 result;
    result.x = data[idx * 3];
    result.y = data[idx * 3 + 1];
    result.z = data[idx * 3 + 2];
    return result;
}

Mat3::Mat3() { data.resize(9); }
Mat3::Mat3(float x00, float x01, float x02, float x10, float x11, float x12,
     float x20, float x21, float x22)
        : data{x00, x01, x02, x10, x11, x12, x20, x21, x22} {}

Mat3 Mat3::operator*(const Mat3& other) const {
    Mat3 result;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum = 0;
            for (int k = 0; k < 3; k++) {
                sum += this->Get(r, k) * other.Get(k, c);
            }
            result.Set(r, c, sum);
        }
    }
    return result;
}

float3 Mat3::operator*(const float3& vec) const {
    float3 result;

    result.x = this->Get(0, 0) * vec.x + this->Get(0, 1) * vec.y + this->Get(0, 2) * vec.z;
    result.y = this->Get(1, 0) * vec.x + this->Get(1, 1) * vec.y + this->Get(1, 2) * vec.z;
    result.z = this->Get(2, 0) * vec.x + this->Get(2, 1) * vec.y + this->Get(2, 2) * vec.z;

    return result;
}

float Mat3::Get(int x, int y) const { return data[x * 3 + y]; }
void Mat3::Set(int x, int y, float val) { data[x * 3 + y] = val; }

Mat4::Mat4() { data.resize(16); }
Mat4::Mat4(float x00, float x01, float x02, float x03, float x10, float x11,
     float x12, float x13, float x20, float x21, float x22, float x23,
     float x30, float x31, float x32, float x33)
        : data{x00, x01, x02, x03, x10, x11, x12, x13,
               x20, x21, x22, x23, x30, x31, x32, x33} {}

void Mat4::operator*=(const Mat4& other) {
    Mat4 result;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) {
                sum += this->Get(r, k) * other.Get(k, c);
            }
            result.Set(r, c, sum);
        }
    }
    std::memcpy(this->data.data(), result.data.data(),
                result.data.size() * sizeof(float));
}
float Mat4::Get(int x, int y) const { return data[x * 4 + y]; }
void Mat4::Set(int x, int y, float val) { data[x * 4 + y] = val; }


