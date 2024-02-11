#!/usr/bin/env python

"""The setup script."""

import io
from os import path

from setuptools import setup, find_packages


with open('README.md') as readme_file:
    readme = readme_file.read()

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with io.open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "git+" not in x]

requirements = []

setup_requirements = []

test_requirements = []

setup(
    author="Markin Hausmanns",
    author_email='Markinhausmanns@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Command line interface for interactions with the ToolBox Network.",
    entry_points={
        'console_scripts': [
            'toolboxv2=toolboxv2.cli:main',
        ],
        'App': [
            'App=toolboxv2.toolboxv2:App',
        ],
    },
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='toolboxv2',
    name='ToolBoxV2',
    packages=find_packages(include=['toolboxv2', 'toolboxv2.mods.*',  'toolboxv2.mods_dev.*', 'toolboxv2.*']),
    package_data={"toolboxv2": ["toolboxv2/init.config"]},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MarkinHaus/ToolBoxV2',
    version='0.1.7',
    zip_safe=False,
)
