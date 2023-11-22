# -*- coding: UTF-8 -*-
"""
Created on 20.11.23

:author:     Martin Dočekal
"""
import argparse
import os
from unittest import TestCase

from rowdatasetsplitter.__main__ import call_subset, call_chunking, call_make_selective_ml_splits

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "fixtures")
TMP_DIR = os.path.join(SCRIPT_DIR, "tmp")

DATASET = os.path.join(FIXTURES_DIR, "dataset.txt")
DATASET_INDEX = os.path.join(FIXTURES_DIR, "dataset.txt.index")
DATASET_INDEX_CSV = os.path.join(FIXTURES_DIR, "dataset.txt.index.csv")

RECORD_DATASET = os.path.join(FIXTURES_DIR, "record_dataset.jsonl")


class Test(TestCase):

    def tearDown(self) -> None:
        # clean tmp dir
        for f in os.listdir(TMP_DIR):
            if f != "placeholder":
                os.remove(os.path.join(TMP_DIR, f))

    def check_chunking(self, args):
        call_chunking(args)

        for chunk in ["dataset_0.txt", "dataset_1.txt", "dataset_2.txt"]:
            self.assertTrue(os.path.exists(os.path.join(TMP_DIR, chunk)), f"Chunk {chunk} was not created.")
            with open(os.path.join(TMP_DIR, chunk), "r") as f, open(os.path.join(FIXTURES_DIR, chunk), "r") as f2:
                self.assertEqual(f.read(), f2.read(), f"Chunk {chunk} does not match.")

    def test_call_chunking_size_without_index(self):
        args = argparse.Namespace(
            data=DATASET,
            out_dir=TMP_DIR,
            size=3,
            number_of_chunks=None,
            file_name_format="{orig_basename}_{counter}{orig_ext}",
            index=None,
            index_offset_field="file_line_offset"
        )

        self.check_chunking(args)

    def test_call_chunking_size_with_index(self):
        args = argparse.Namespace(
            data=DATASET,
            out_dir=TMP_DIR,
            size=3,
            number_of_chunks=None,
            file_name_format="{orig_basename}_{counter}{orig_ext}",
            index=DATASET_INDEX,
            index_offset_field="file_line_offset"
        )

        self.check_chunking(args)

    def test_call_chunking_size_with_index_csv(self):
        args = argparse.Namespace(
            data=DATASET,
            out_dir=TMP_DIR,
            size=3,
            number_of_chunks=None,
            file_name_format="{orig_basename}_{counter}{orig_ext}",
            index=DATASET_INDEX_CSV,
            index_offset_field="file_line_offset"
        )

        self.check_chunking(args)

    def test_call_chunking_number_of_chunks_without_index(self):
        args = argparse.Namespace(
            data=DATASET,
            out_dir=TMP_DIR,
            size=None,
            number_of_chunks=3,
            file_name_format="{orig_basename}_{counter}{orig_ext}",
            index=None,
            index_offset_field="file_line_offset"
        )

        self.check_chunking(args)

    def check_subset(self, args):
        call_subset(args)

        with open(args.out, "r") as f, open(os.path.join(FIXTURES_DIR, "subset.txt"), "r") as f2:
            self.assertEqual(f.read(), f2.read())

    def test_call_subset_without_index(self):
        args = argparse.Namespace(
            data=DATASET,
            out=os.path.join(TMP_DIR, "subset.txt"),
            from_line=3,
            to_line=6,
            index=None,
            index_offset_field="file_line_offset"
        )

        self.check_subset(args)

    def test_call_subset_with_index(self):
        args = argparse.Namespace(
            data=DATASET,
            out=os.path.join(TMP_DIR, "subset.txt"),
            from_line=3,
            to_line=6,
            index=DATASET_INDEX,
            index_offset_field="file_line_offset"
        )

        self.check_subset(args)

    def test_call_subset_with_index_csv(self):
        args = argparse.Namespace(
            data=DATASET,
            out=os.path.join(TMP_DIR, "subset.txt"),
            from_line=3,
            to_line=6,
            index=DATASET_INDEX_CSV,
            index_offset_field="file_line_offset"
        )

        self.check_subset(args)

    def check_make_selective_ml_splits(self, args):
        call_make_selective_ml_splits(args)

        with open(args.out_train, "r") as f, open(os.path.join(FIXTURES_DIR, "train_subset.jsonl"), "r") as f2:
            self.assertEqual(f.read(), f2.read())

        with open(args.out_validation, "r") as f, open(os.path.join(FIXTURES_DIR, "validation_subset.jsonl"), "r") as f2:
            self.assertEqual(f.read(), f2.read())

        with open(args.out_test, "r") as f, open(os.path.join(FIXTURES_DIR, "test_subset.jsonl"), "r") as f2:
            self.assertEqual(f.read(), f2.read())

    def test_call_make_selective_ml_splits(self):
        args = argparse.Namespace(
            data=RECORD_DATASET,
            out_train=os.path.join(TMP_DIR, "train_subset.jsonl"),
            out_validation=os.path.join(TMP_DIR, "validation_subset.jsonl"),
            out_test=os.path.join(TMP_DIR, "test_subset.jsonl"),
            validation=os.path.join(FIXTURES_DIR, "validation_subset.jsonl.csv"),
            test=os.path.join(FIXTURES_DIR, "test_subset.jsonl.tsv"),
            key="key",
            data_key="id"
        )
        self.check_make_selective_ml_splits(args)


