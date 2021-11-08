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
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=['tqdm', 'pandas',
                      'nwdata@git+https://github.com/nimbal/nwdata@v0.7.2#egg=nwdata',
                      'nwnonwear@git+https://github.com/nimbal/nwnonwear@v0.1.2#egg=nwnonwear',
                      'nwactivity@git+https://github.com/nimbal/nwactivity@v0.1.3#egg=nwactivity',
                      'nwgait@git+https://github.com/nimbal/nwgait@v0.3.0#egg=nwgait',
                      'nwsleep@git+https://github.com/nimbal/nwsleep@v0.3.1#egg=nwsleep'],
)