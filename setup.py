import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nwpipeline",
    version="0.1.0",
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
                      'pyedflib@git+https://github.com/holgern/pyedflib@v0.1.21#egg=pyedflib',
                      'nwdata@git+https://github.com/nimbal/nwdata@v0.1.1#egg=nwdata'],
)