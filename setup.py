"""A setuptools based setup module for the tf-idf package.

Blatantly copied from:
https://github.com/pypa/sampleproject
"""
from codecs import open  # To use a consistent encoding
from os import path
from setuptools import setup, find_packages  # Always prefer setuptools over distutils
import sys

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tf-idf',
    provides=['tfidf'],

    version='0.0.0',

    description='An implementation of TF-IDF for keyword extraction.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/elzilrac/tf-idf',

    # Author details
    author='elzilrac',
    author_email='elzilrac@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Indexing',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='tfidf text mining extraction keywords tf-idf stemming ngram',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'cachetools',
        'six',
        'nltk',
        'stop-words',
    ],

    # Additional requirements for development and testing
    extras_require={
        'dev': [],
        'test': ['pytest', 'pytest-cov', 'pytest-pythonpath'],
    }

)
