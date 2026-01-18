"""
Setup script for Vexa Polish LLM
"""

from setuptools import setup, find_packages
import os

# Wczytanie README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Wczytanie requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Wersja
VERSION = "1.0.0"

setup(
    name="vexa-polish-llm",
    version=VERSION,
    author="Vexa Team",
    author_email="vexa@example.com",
    description="System uczenia maszynowego oparty na algorytmie kolonii mrówek (ACO) dla języka polskiego",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vexa-polish-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Polish",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "web": [
            "flask>=2.0.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "eval": [
            "nltk>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vexa=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
    keywords=[
        "nlp",
        "machine-learning",
        "ant-colony-optimization",
        "aco",
        "polish",
        "language-model",
        "llm",
        "text-generation",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/vexa-polish-llm/issues",
        "Source": "https://github.com/yourusername/vexa-polish-llm",
        "Documentation": "https://github.com/yourusername/vexa-polish-llm#readme",
    },
)
