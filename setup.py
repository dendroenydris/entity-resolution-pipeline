from setuptools import setup, find_packages

with open("README.md", "r") as f:
    page_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="DIA",
    version="0.0.0",
    author="my_name",
    author_email="my_email",
    description='DIA homework',
    long_description=page_description,
    long_description_content_type="text/markdown",
    url="my_github_repository_project_link",
    packages=find_packages("LocalERP"),
    install_requires=requirements,
    python_requires='>=3.8',
)