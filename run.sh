#!/bin/bash

EXEC="./build/debug_hdf5"

FLAGS="$FLAGS -logfile log_%.log"
FLAGS="$FLAGS -hdf5:forcerw"
FLAGS="$FLAGS -ll:cpu 1"
FLAGS="$FLAGS -ll:util 1"
FLAGS="$FLAGS -ll:dma 1"
FLAGS="$FLAGS -lg:warn"

export LEGION_FREEZE_ON_ERROR=1

# CMD="$EXEC $FLAGS"
# CMD="valgrind --num-callers=500 $EXEC $FLAGS"
# CMD="mpirun -n 1 -H n0000 -npernode 1 -bind-to none $EXEC $FLAGS"
CMD="mpirun -n 2 -H n0000,n0001 -npernode 1 -bind-to none $EXEC $FLAGS"

echo
echo $CMD
echo
$CMD |& tee log.out
