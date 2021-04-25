#!/bin/bash

# Options and what they do:
#
# -g 		Enable debugging symbols 
# -G		More debug symbols
# -O0 		Disable optimization (more debuggable executable)
# --std=c++11 	Use c++11 
# --ccbin 	Tell it where to find normal gcc 
# -m64		Generate 64 bit code 
# --gpu-arch... Define our output architecture
# -Xcompiler	Pass an arg of -DNVCC (#define NVCC) to the compiler
# -I ...	Include the CUDA libraries and libwb
# -L ...        Add this directory as a link library path
# -lwb          Instruct the linker to link with libwb.a

case `hostname -f` in
JoshArchBox)
NVCC_ARGS="-g -G -O0 --x cu --std=c++11 -ccbin /opt/cuda/bin/gcc -m64 --gpu-architecture sm_60 -Xcompiler -DNVCC -I/opt/cuda/include -I../libwb -L`pwd` -lwb"
;;
*.cm.cluster)
NVCC_ARGS="-g -G -O0 --x cu --std=c++11 -ccbin /cm/local/apps/gcc/6.1.0/bin/gcc -m64 --gpu-architecture sm_60 -Xcompiler -DNVCC -I/cm/shared/apps/cuda91/toolkit/9.1.85/include -I../libwb -L`pwd` -lwb"
;;
esac

nvcc ${NVCC_ARGS} ./catch-runner.cpp ./catch-tests-*.cpp -o test-runner
