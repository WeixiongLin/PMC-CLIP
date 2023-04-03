""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('src/pmc_clip/version.py').read())
setup(
    name='pmc_clip',
    version=__version__,
    description='PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/WeixiongLin/PMC-CLIP/',
    author='weixiong',
    author_email='wx_lin@sjtu.edu.cn',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='PMC-CLIP',
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['training']),
    include_package_data=True,
    install_requires=[
        'torch >= 1.9',
        'torchvision',
        'transformers <= 4.21.0',
        'ftfy',
        'regex',
        'tqdm',
    ],
    python_requires='>=3.7',
)
