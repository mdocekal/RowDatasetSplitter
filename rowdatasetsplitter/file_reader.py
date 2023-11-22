# -*- coding: UTF-8 -*-
"""
Created on 22.11.23

Contains class for reading formatted files (.csv, .tsv, jsonl).

:author:     Martin Dočekal
"""

import csv
import json
from typing import Dict, Type


class FileReader:
    """
    Class for reading formatted files (.csv, .tsv, jsonl).
    """

    def __init__(self, file_path: str):
        """
        :param file_path: Path to the file.
        """

        self.file_path = file_path

        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __iter__(self):
        if self.file_path.endswith(".csv"):
            return csv.DictReader(self.file)
        elif self.file_path.endswith(".tsv"):
            return csv.DictReader(self.file, delimiter="\t")
        elif self.file_path.endswith(".jsonl"):
            return (json.loads(line) for line in self.file)
        else:
            raise ValueError("Unsupported file format.")