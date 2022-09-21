# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="face_api_dataset",
    version="1.0.7",
    python_requires=">=3.6.0",
    license="MIT",
    author="Synthesis AI",
    install_requires=[
        "tiffile>=2018.10.18",
        "opencv-python>=4.5.1.48",
        "imagecodecs>=2020.5.30",
        "numpy>=1.19.5",
        "pandas>=0.22.0"
    ],
    include_package_data=True,
    description="Class to access datasets by FaceAPI datasets from Synthesis AI",
    packages=find_packages(),
)
