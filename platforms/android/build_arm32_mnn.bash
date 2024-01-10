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
# NDK_PATH="/Users/mzh/work/read_project/sdk/ndk/21.2.6472646"
# NDK_PATH="/Users/mzh/work/read_project/sdk/ndk/android-ndk-r21e"
NDK_PATH="/Users/mzh/Library/Android/sdk/ndk/26.1.10909125"
# NDK_PATH="/Users/mzh/Library/Android/sdk/ndk/25.2.9519653"
CMAKE_TOOLCHAIN_PATH="${NDK_PATH}/build/cmake/android-legacy.toolchain.cmake"



cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_PATH \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NDK=$NDK_PATH  \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DANDROID_ABI="armeabi-v7a" \
-DMNN_SUPPORT_BF16=OFF \
-DANDROID_STL=c++_static \
-DDENABLE_CXX11=ON \
-DBUILD_ANDROID_PROJECTS=OFF \
-DANDROID_TOOLCHAIN=clang \
-DBUILD_ANDROID_EXAMPLES=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_TESTS=OFF \
-DANDROID_NATIVE_API_LEVEL=android-14 \
-DANDROID_TOOLCHAIN=clang \
-DBUILD_opencv_highgui=OFF \
-DBUILD_opencv_imgcodecs=OFF \
-DBUILD_OPENJPEG=OFF \
-DBUILD_opencv_dnn=ON \
-DBUILD_ZLIB=OFF \
-DBUILD_TIFF=OFF \
-DBUILD_OPENJPEG=OFF \
-DBUILD_JASPER=OFF \
-DBUILD_JPEG=OFF \
-DBUILD_PNG=OFF \
-DWITH_JASPER=OFF \
-DWITH_OPENJPEG=OFF \
-DWITH_JPEG=OFF \
-DWITH_WEBP=OFF \
-DWITH_OPENEXR=OFF \
-DWITH_PNG=OFF \
-DWITH_TIFF=OFF \
-DWITH_OPENVX=OFF \
-DWITH_GDCM=OFF \
-DWITH_TBB=OFF \
-DWITH_HPX=OFF \
-DBUILD_OPENEXR=OFF \
-DBUILD_WEBP=OFF \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DMNN_BUILD_SHARED_LIBS=ON \
-DBUILD_SHARED_LIBS=ON

make -j4
