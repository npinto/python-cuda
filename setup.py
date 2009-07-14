#!/usr/bin/env python
# -*- coding: utf-8 -*- 

""" setuptools setup.py for python-cuda """

from ez_setup import use_setuptools
use_setuptools(version='0.6c9')

from setuptools import setup, find_packages

setup(
    name = 'python-cuda',

    version = '2.1-0.0.1',

    packages = ['cuda',
                'cuda.cu',
                'cuda.cuda', 
                'cuda.kernel', 
                'cuda.memory', 
                'cuda.cublas', 
                'cuda.cufft', 
                'cuda.sugar',
                'cuda.utils'],

    package_dir = {'cuda':'cuda'},

    install_requires=[
    ],


#     author='',
#     author_email='',
#     url='',
#     description='Python bindings for CUDA 2.1 with numpy integration',
#     long_description = """ """,
#     download_url='',
#     license='?',
#     package_data = {}

)
