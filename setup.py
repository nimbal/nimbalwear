import setuptools
import re

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

project_name = "nwpipeline"

setuptools.setup(
    name=project_name,
    version=get_property('__version__', project_name),
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
                      'nwdata@git+https://github.com/nimbal/nwdata@v0.4.0#egg=nwdata',
                      'nwnonwear@git+https://github.com/nimbal/nwnonwear@v0.1.1#egg=nwnonwear',
                      'nwactivity@git+https://github.com/nimbal/nwactivity@v0.1.2#egg=nwactivity',
                      'nwgait@git+https://github.com/nimbal/nwgait@v0.1.2#egg=nwgait',
                      'nwsleep@git+https://github.com/nimbal/nwsleep@v0.1.0#egg=nwsleep'],
)