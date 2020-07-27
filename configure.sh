#!/bin/bash

BUILD_TYPE=Debug
HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial
LEGION_ROOT=/home/kihiro/Softwares/legion_DCR_debug

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DHDF5_ROOT=$HDF5_ROOT \
      -DLegion_ROOT=$LEGION_ROOT \
      ..
