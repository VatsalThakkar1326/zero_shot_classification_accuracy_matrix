"""
Setup file for tag_wording_analyzer package.
"""

from setuptools import setup, find_packages

setup(
    name="tag_wording_analyzer",
    version="1.0.0",
    description="Find optimal tag wordings for zero-shot classification models",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.20.0",
        "torch>=1.10.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
