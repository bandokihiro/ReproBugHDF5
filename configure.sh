#!/bin/bash

BUILD_TYPE=Debug
LEGION_ROOT=/home/bandokihiro/Softwares/legion_DCR_debug

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DHDF5_ROOT="" \
      -DLegion_ROOT=$LEGION_ROOT \
      ..
