#!/bin/sh

# NP: do we need a python script for that ?

INCLUDES="-I ./ -I /usr/local/cuda/include"

python createbindings.py -H my_CUDA2100_vector_types.h -H cuda.h -l /usr/lib/libcuda.so $INCLUDES -x cudadrv.xml -p cudadrv.py && \
python cudadrv.py && cp -vf cudadrv.py ../cuda/cu/

python createbindings.py -H my_CUDA2100_vector_types.h -H cuda_runtime.h -l /usr/local/cuda/lib/libcudart.so $INCLUDES -x cudart.xml -p cudart.py && \
python cudart.py && cp -vf cudart.py ../cuda/cuda/

python createbindings.py -H my_CUDA2100_vector_types.h -H cublas.h -l /usr/local/cuda/lib/libcublas.so $INCLUDES -x cublas.xml -p cublas.py && \
python cublas.py && cp -vf cublas.py ../cuda/cublas/

python createbindings.py -H my_CUDA2100_vector_types.h -H cufft.h -l /usr/local/cuda/lib/libcufft.so $INCLUDES -x cufft.xml -p cufft.py && \
python cufft.py && cp -vf cufft.py ../cuda/cufft/

#rm -vf cudadrv.xml cudadrv.py cudart.xml cudart.py cublas.xml cublas.py cufft.xml cufft.py