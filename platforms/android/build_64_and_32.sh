#!/bin/bash

# cd build32
# bash ../build_arm32_mnn.bash
# cd ../build64
# bash ../build_arm64_mnn.bash

if [ ! -d "build32" ]; then
    echo "build32 dose not exist, creating..."
    mkdir build32
fi
cd build32
zsh ../build_arm32_mnn.sh

cd ..

if [ ! -d "build64" ]; then
    echo "build64 dose not exist, creating..."
    mkdir build64
fi

cd build64
bash ../build_arm64_mnn.sh

cd ..

echo "Build process completed for both 32-bit and 64-bit architectures."