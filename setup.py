"""Packaging metadata for the heart disease MLOps assignment."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="heart-disease-mlops",
    version="1.0.0",
    author="RAJ KUMAR M",
    author_email="2024AD05110@WILP.BITS-PILANI.AC.IN",
    description="Heart disease prediction MLOps assignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RAJKUMAR27M/MLOPS-Assignment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
)
