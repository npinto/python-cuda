#!/usr/bin/python
# -*- coding: utf-8 -*- 

from ez_setup import use_setuptools
use_setuptools(version='0.6c9')

from setuptools import setup, find_packages

setup(
    name = 'python-cuda',
    version = '2.1-0.0.1',

    packages= ['cuda','cuda.cuda','cuda.cu', 'cuda.cublas', 'cuda.cufft', 'cuda.utils'],
    package_dir = {'cuda':'cuda'},

#     author='',
#     author_email='',
#     url='',
#     description='Python bindings for CUDA 2.1 with numpy integration',
#     long_description = """ """,
#     download_url='',
#     license='?',
#     package_data = {}

)
