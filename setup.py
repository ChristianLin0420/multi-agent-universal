from setuptools import setup, find_packages

setup(
    name="marl_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if not line.startswith("#") and line.strip()
    ]
) 