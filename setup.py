import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gcornilib-horns-g",
    version="0.0.2",
    author="Gabriele Corni",
    author_email="gabriele_corni@iprel.it",
    description="Data manipulation pipeline for Machine Learning datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/horns-g/GCDataManipulationLib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# python setup.py install
