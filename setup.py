#!/usr/bin/python3
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

if not torch.cuda.is_available():
    print('CUDA device not found.')
    exit(0)

setup(
    name='cudaDPM',
    ext_modules=[
        CUDAExtension(
            'cudaDPM', 
            sorted(glob("src/*c*")),
            include_dirs=["include/"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
