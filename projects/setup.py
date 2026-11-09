"""Build the LIFE plugin and its BEVPoolv2 CUDA extension from source."""
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PROJECT_ROOT = Path(__file__).resolve().parent
# Resolve package and extension paths relative to this directory.
os.chdir(PROJECT_ROOT)


if __name__ == '__main__':
    setup(
        name='life-occupancy',
        version='0.1.0',
        description='Illumination-aware camera-based semantic occupancy prediction',
        packages=find_packages(),
        ext_modules=[
            CUDAExtension(
                name='life.ops.bev_pool_v2.bev_pool_v2_ext',
                sources=[
                    'life/ops/bev_pool_v2/src/bev_pool.cpp',
                    'life/ops/bev_pool_v2/src/bev_pool_cuda.cu',
                ],
                define_macros=[('WITH_CUDA', None)],
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': [
                        '-O3',
                        '-D__CUDA_NO_HALF_OPERATORS__',
                        '-D__CUDA_NO_HALF_CONVERSIONS__',
                        '-D__CUDA_NO_HALF2_OPERATORS__',
                    ],
                },
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False,
    )
