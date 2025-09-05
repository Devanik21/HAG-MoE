#!/usr/bin/env python3
"""
Setup script for HAG-MoE: Hierarchical Attention-Gated Mixture of Experts
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hag-moe",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Hierarchical Attention-Gated Mixture of Experts for Scalable Intelligence",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hag-moe",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hag-moe/issues",
        "Source": "https://github.com/yourusername/hag-moe",
        "Documentation": "https://hag-moe.readthedocs.io/",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": ["sphinx", "sphinx-rtd-theme", "myst-parser"],
        "visualizations": ["matplotlib", "seaborn", "plotly"],
    },
    entry_points={
        "console_scripts": [
            "hag-moe-train=hag_moe.scripts.train:main",
            "hag-moe-eval=hag_moe.scripts.evaluate:main",
            "hag-moe-export=hag_moe.scripts.export_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hag_moe": ["configs/*.yaml", "configs/**/*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "machine learning",
        "transformers",
        "mixture of experts",
        "attention mechanisms",
        "deep learning",
        "neural networks",
        "natural language processing",
    ],
)
