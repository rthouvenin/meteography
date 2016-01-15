# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="meteography",
    version="0.1",
    description="Aims at predicting what the sky will look like",
    author="Romain Thouvenin",
    author_email="romain.thouvenin@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'scipy',
        'matplotlib',
        'scikit-learn',
        'tables>=3.1',
        'Django>=1.8',
    ],
)
