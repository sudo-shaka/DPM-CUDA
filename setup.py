#!/usr/bin/python3
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

if not torch.cuda.is_available():
    print('CUDA device not found.')
    exit(0)

sources = []
[sources.append(k) for k in glob("src/*")]
[sources.append(k) for k in glob("kernels/*")]
print(sources)

setup(
    name='cudaDPM',
    ext_modules=[
        CUDAExtension(
            'cudaDPM', 
            sorted(sources),
            include_dirs=["include"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
