#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='explore_ml',
    version='1.0',
    author='Robert Calef',
    author_email='robert.calef@gmail.com',
    packages=find_packages(),
    scripts=['bin/prepare_mnist_data.py',],
    description='Utilities for exploring machine learning',
    install_requires=[
        'argh',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'opencv-python'
    ],
    zip_safe=True,
    include_package_data = True
)
