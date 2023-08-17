from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.0'
DESCRIPTION = 'Generative & Federated AI PLatform from K{r}eeda Labs'

# Setting up
setup(
    name="fedgen",
    version=VERSION,
    author="Diganta Dutta, Anmol Dhingra",
    author_email="diganta.dutta@aimlos.in, anmoldhingra1@gmail.com",
    url="https://github.com/diganta97aimlos/GenAI-Rush-FedGen/tree/infra",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy==1.24.2', 'scikit-learn==1.1.3', 'pandas==1.5.2', 'torch==1.13.1', 'torchvision==0.14.1', 'albumentations==1.3.0',
        'opencv-python==4.7.0.68', 'opacus==1.4.0', 'fas14mnet==0.1.2'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)