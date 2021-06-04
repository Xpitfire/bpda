from setuptools import setup, find_packages

with open("../README.md", "r") as fh:
    readme = fh.read()

requirements = {"install": ["torch-lighter>=0.2.26"]}

install_requires = requirements["install"]

setup(
    name='bpda',
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
    include_package_data=True,
    version='0.0.2',
)
