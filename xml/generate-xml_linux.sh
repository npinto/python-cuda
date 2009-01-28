#!/bin/sh

# NP: do we need a python script for that ?

python createbindings.py -H /usr/local/cuda/include/cuda.h -l /usr/lib/libcuda.so -x cu.xml -p cudadrv.py && \
python cudadrv.py && cp -vf cudadrv.py ../cuda/cu/

python createbindings.py -H /usr/local/cuda/include/cuda_runtime.h -l /usr/local/cuda/lib/libcudart.so -x cudart.xml -p cudart.py && \
python cudart.py && cp -vf cudart.py ../cuda/cuda/

python createbindings.py -H /usr/local/cuda/include/cublas.h -l /usr/local/cuda/lib/libcublas.so -x cublas.xml -p cublas.py && \

python createbindings.py -H /usr/local/cuda/include/cufft.h -l /usr/local/cuda/lib/libcufft.so -x cufft.xml -p cufft.py && \
python cufft.py && cp -vf cufft.py ../cuda/cufft/

