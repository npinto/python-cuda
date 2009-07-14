#!/bin/sh

# NP: do we need a python script for that ?

INCLUDES="-I ./ -I /usr/local/cuda/include"

python createbindings.py -o cuda -H my_CUDA2100_vector_types.h -H cuda.h -l /usr/lib/libcuda.so $INCLUDES -x cudadrv2.xml -p cudadrv2.py && \
#python cudadrv.py && cp -vf cudadrv.py ../cuda/cu/
python cudadrv2.py

python createbindings.py -o cuda -H my_CUDA2100_vector_types.h -H cuda_runtime.h -l /usr/local/cuda/lib/libcudart.so $INCLUDES -x cudart2.xml -p cudart2.py && \
#python cudart.py && cp -vf cudart.py ../cuda/cuda/
python cudart2.py

python createbindings.py -o cuda -H my_CUDA2100_vector_types.h -H cublas.h -l /usr/local/cuda/lib/libcublas.so $INCLUDES -x cublas2.xml -p cublas2.py && \
#python cublas.py && cp -vf cublas.py ../cuda/cublas/
python cublas2.py

python createbindings.py -o cuda -H my_CUDA2100_vector_types.h -H cufft.h -l /usr/local/cuda/lib/libcufft.so $INCLUDES -x cufft2.xml -p cufft2.py && \
#python cufft.py && cp -vf cufft.py ../cuda/cufft/
python cufft2.py

#rm -vf cudadrv.xml cudadrv.py cudart.xml cudart.py cublas.xml cublas.py cufft.xml cufft.py

