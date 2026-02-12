from setuptools import Extension, setup


def build_ext_modules():
    import pybind11

    extra_compile_args = ["-O3", "-std=c++17"]
    extra_link_args = []

    return [
        Extension(
            name="ar_native",
            sources=["native/ar_native.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


setup(
    name="ar-native",
    version="0.0.0",
    ext_modules=build_ext_modules(),
    zip_safe=False,
) 
