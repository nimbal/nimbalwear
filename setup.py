import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nwpipeline",
    version="0.0.1",
    author="Kit Beyer",
    author_email="kitbeyer@gmail.com",
    description="NiMBaLWear wearable data processing pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nimbal/nwpipeline",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=['tqdm', 'pandas',
                      'pyedflib@git+https://github.com/holgern/pyedflib#egg=pyedflib',
                      'nwfiles@git+https://github.com/nimbal/nwfiles#egg=nwfiles'],
)