import setuptools

with open("ReadMe.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microdraw", # Replace with your own username
    version="0.0.1",
    author="NAAT",
    author_email="robertotoro@gmail.com",
    description="Use microdraw with python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/r03ert0/microdraw.py",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
)
