# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="face_api_dataset",
    version="1.0.1",
    python_requires=">=3.6.0",
    license="MIT",
    author="Synthesis AI",
    install_requires=[
        "tiffile>=2018.10.18",
        "opencv-python>=4.5.1.48",
        "imagecodecs>=2021.1.28",
        "numpy>=1.20.1",
    ],
    include_package_data=True,
    description="Class to access datasets by FaceAPI datasets from Synthesis AI",
    packages=find_packages(),
)
