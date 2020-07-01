#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""""
Created on 07.10.19
KeywordsExtractor
Extraction of keywords from a text.

:author:     Martin Dočekal
"""

from distutils.core import setup

setup(name='RowDatasetSplitter',
    version='1.0.0',
    description='Small script for splitting row datasets into train, validation and test sets. ',
    author='Martin Dočekal',
    packages=['rowdatasetsplitter'],
    entry_points={
        'console_scripts': [
            'rowdatasetsplitter = rowdatasetsplitter.__main__:main'
        ]
    },
    install_requires=[]
)
