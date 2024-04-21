#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

install_requires = [
    'numpy',
    'h5parm>=1.0.5',
    'scipy',
    'astropy>=6',
    'matplotlib',
    'cmocean',
    'tqdm',
    'jax',
    'jaxlib',
    'tensorflow_probability'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='tomographic_kernel',
      version='1.0.3',
      description='A Tomographic Kernel in JAX for tomographic Gaussian processes.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/JoshuaAlbert/tomographic_kernel",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      install_requires=install_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.8'
      )
