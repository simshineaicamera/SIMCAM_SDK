import os
import shutil
from setuptools import setup, find_packages

setup(
    name = "mvnc",
    version = "1.12.00.01",
    author = "Intel Corporation",
    description = ("mvnc python api"),
    license="None",
    keywords = "",
    url = "http://developer.movidius.com",
    packages=['mvnc'],
    package_dir={'mvnc': 'python/mvnc'},
    long_description="-",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Software Development :: Libraries",
        "License :: Other/Proprietary License",
    ],
)
