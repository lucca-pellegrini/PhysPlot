"""
Setup script for PhysPlot - Physics Plotting Library
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="physplot",
    version="0.1.0",
    author="Lucca Pellegrini",
    author_email="lucca@verticordia.com",
    description="A physics plotting library with regression analysis and LaTeX formatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="ISC",
    url="https://github.com/lucca-pellegrini/PhysPlot",
    packages=find_packages(),
    py_modules=["physplot"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/lucca-pellegrini/PhysPlot/issues",
        "Source": "https://github.com/lucca-pellegrini/PhysPlot",
    },
)
