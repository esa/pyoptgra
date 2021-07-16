import sys

from skbuild import setup

setup(
    name="pyoptgra",
    packages=["pyoptgra"],
    setup_requires=["cmake", "ninja"],
    cmake_install_dir="pyoptgra/core",
)
