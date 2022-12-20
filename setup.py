#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'seaborn',
    'soundfile',
    'sounddevice'
]

# setup_requirements = ['pytest-runner', ]

test_requirements = [
    'pytest',
    'bump2version',
    # 'wheel',
    # 'watchdog',
    'flake8',
    # 'tox',
    # 'coverage',
    'Sphinx',
    'twine'
]

setup(
    author="Nicolas Franco-Gomez",
    author_email='nicolasfrancogomez@gmail.com',
    classifiers=[
        'Development Status :: 0 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10'
    ],
    description="Project for prototyping of complex dsp algorithms.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='dsptools',
    name='dsptools',
    packages=find_packages(),
    # setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nico-franco-gomez/dsptools',
    version='0.1.0',
    zip_safe=False,
    python_requires='>=3.10'
)
