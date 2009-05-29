#!/usr/bin/env python

import cuda
import cu

import cublas
import cufft

import memory
import kernel

import utils
import sugar

import logging
logging.basicConfig(level=logging.DEBUG)
#logging.disable(logging.ERROR)
