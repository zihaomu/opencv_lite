#!/bin/bash

# auto compile opencv_lite and MNN
# cmake ../../../ \
# -DCMAKE_TOOLCHAIN_FILE=/Users/mzh/work/read_project/sdk/ndk/21.2.6472646/build/cmake/android.toolchain.cmake \
# -DCMAKE_BUILD_TYPE=Release \
# -DANDROID_ABI="arm64-v8a" \
# -DANDROID_STL=c++_static \
# -DMNN_USE_LOGCAT=false \
# -DMNN_BUILD_BENCHMARK=ON \
# -DMNN_USE_SSE=OFF \
# -DMNN_SUPPORT_BF16=OFF \
# -DMNN_BUILD_TEST=ON \
# -DMNN_ARM82=ON \
# -DMNN_OPENCL=ON \
# -DMNN_VULKAN=ON \
# -DMNN_OPENGL=ON \
# -DMNN_BUILD_TRAIN=ON \
# -DANDROID_NATIVE_API_LEVEL=android-21  \
# -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
# -DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3

# make -j4

# /Users/mzh/work/read_project/sdk/ndk/21.2.6472646/build/cmake/android.toolchain.cmake
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DDENABLE_CXX11=ON \
-DWITH_PROTOBUF=OFF

make -j4
