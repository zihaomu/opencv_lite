#!/bin/bash

# auto compile opencv_lite and MNN

NDK_PATH="/Users/moo/Library/Android/sdk/ndk/26.1.10909125"
CMAKE_TOOLCHAIN_PATH="${NDK_PATH}/build/cmake/android.toolchain.cmake"

cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_PATH \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_NDK=$NDK_PATH  \
-DCMAKE_CXX_FLAGS=-std=c++11 \
-DANDROID_ABI="armeabi-v7a" \
-DTFLITE_LIB=/Volumes/SSD1T/work_ssd/my_project/tflite_cmake_android_26/tflite_216_sdk/arm32 \
-DTFLITE_INC=/Volumes/SSD1T/work_ssd/my_project/tflite_cmake_android_26/tflite_216_sdk/tflite_include \
-DTFLITE_GEMMLOWP_INC=/Volumes/SSD1T/work_ssd/my_project/tflite_cmake_android_26/tflite_216_sdk/tflite_include/gemmlowp \
-DBUILD_PNG=ON \
-DBUILD_opencv_world=ON \
-DMNN_OPENCL=ON \
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

make -j8
