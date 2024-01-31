from setuptools import setup, find_packages

with open("README.md", "r") as f:
    page_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="DIA",
    version="0.0.0",
    author="Yuxin XUE, Ofir Topchy, NevoLevi",
    author_email="nevo.levi@campus.tu-berlin.de, ofir.topchy@campus.tu-berlin.de, yuxin.xue@campus.tu-berlin.de",
    description="DIA homework",
    long_description=page_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Catoblepases/DIA",
    packages=find_packages("erp"),
    install_requires=requirements,
    python_requires=">=3.8",
)
