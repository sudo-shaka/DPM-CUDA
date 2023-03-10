#!/usr/bin/python3
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
assert torch.cuda.is_available()
setup(
    name='cudaDPM',
    ext_modules=[
        CUDAExtension('cudaDPM', [
            'src/Cell.cpp',
            'src/Tissue.cu',
            'src/DPMCudaKernel.cu',
            'src/GeometricFunctions.cpp',
            'src/Module_Wrapper.cpp',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
