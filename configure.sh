#!/bin/bash

BUILD_TYPE=Debug
# LEGION_ROOT=/home/kihiro/Softwares/legion_DCR_debug
# LEGION_ROOT=/home/bandokihiro/Softwares/legion_DCR_debug
LEGION_ROOT=/home/bandokihiro/Softwares/legion_DCR_debug/share/Legion/cmake

cmake \
    -DCMAKE_C_COMPILER="gcc-8" \
    -DCMAKE_CXX_COMPILER="g++-8" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DHDF5_DIR="" \
    -DLegion_DIR=$LEGION_ROOT \
    ..
