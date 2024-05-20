from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='segment_manager',
    ext_modules=[
        CUDAExtension('segment_manager', [
            'pybind.cpp',
            'shm_allocator.cpp',
            'global_segment_manager.cpp',
            'global_segment_manager_client.cpp',
            'local_segment_manager.cpp',
        ], 
        extra_compile_args = {'cxx': ["-std=c++17", "-g", "-O0", "-Wall"]},
        undef_macros=['NDEBUG']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })