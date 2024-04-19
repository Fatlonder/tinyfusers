#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='tinyfusers',
      version='0.1.0',
      description='Diffusion based models built around tinygrad.',
      author='Fatlonder Cakolli',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=["tinygrad", "tqdm",
                        "pyobjc-framework-Metal; platform_system=='Darwin'",
                        "pyobjc-framework-libdispatch; platform_system=='Darwin'"],
      python_requires='>=3.8',
      include_package_data=True)