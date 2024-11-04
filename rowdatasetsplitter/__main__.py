# -*- coding: UTF-8 -*-
""""
Created on 01.07.20
RowDatasetSplitter
Small script for splitting row datasets into train, validation and test sets.

:author:     Martin Dočekal
"""
import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from rowdatasetsplitter.file_reader import FileReader
from collections import defaultdict


class ArgumentParserError(Exception):
    """
    Exceptions for argument parsing.
    """
    pass


class ExceptionsArgumentParser(ArgumentParser):
    """
    Argument parser that uses exceptions for error handling.
    """

    def error(self, message):
        raise ArgumentParserError(message)


class ArgumentsManager(object):
    """
    Parsers arguments for script.
    """

    @classmethod
    def parse_args(cls):
        """
        Performs arguments parsing.

        :param cls: arguments class
        :returns: Parsed arguments.
        """

        parser = ExceptionsArgumentParser(
            description="Script for splitting row datasets.")

        subparsers = parser.add_subparsers()

        ml_splits = subparsers.add_parser("ml_splits",
                                          help="Splitting row datasets into train, validation and test sets. Supports also csv and tsv datasets.")

        ml_splits.add_argument("-d", "--data",
                               help="Dataset that should be split.", type=str,
                               required=True)
        ml_splits.add_argument("--outTrain",
                               help="Path where the train subset will be saved.", type=str,
                               required=True)
        ml_splits.add_argument("--outValidation",
                               help="Path where the validation subset will be saved.", type=str,
                               required=True)
        ml_splits.add_argument("--outTest",
                               help="Path where the test subset will be saved.", type=str,
                               required=True)
        ml_splits.add_argument("-v", "--validation",
                               help="Size of validation set in percents. Float number in [0,1).", type=float,
                               required=True)
        ml_splits.add_argument("-s", "--test",
                               help="Size of test set in percents. Float number in [0,1).", type=float,
                               required=True)
        ml_splits.add_argument("-f", "--fast",
                               help="Activates fast splitting strategy that needs only one pass trough original dataset. "
                                    "Normally the splitting is done in two phases. Where first counts number of samples and "
                                    "second is writing samples according to random generated permutation of original samples. "
                                    "If you use fast strategy please be aware that selected sizes will be just approximated, "
                                    "because random selection is used. That means that for every sample we randomly decide "
                                    "(probability is given according to a subset size) where it goes (sample can go only into a "
                                    "one subset).",
                               action='store_true')
        ml_splits.add_argument("--fixedSeed",
                               help="Fixes random seed. Useful when you want same splits.", action='store_true')
        ml_splits.add_argument("--header", help="If the dataset has a header. Is considered for csv and tsv datasets.", action='store_true')
        ml_splits.set_defaults(func=call_make_ml_splits)

        selective_ml_splits = subparsers.add_parser("selective_ml_splits",
                                                    help="Splitting row datasets into train, validation and test sets. According to given set of ids.")
        selective_ml_splits.add_argument("-d", "--data",
                                         help="Dataset that should be split. It must be .jsonl.",
                                         type=str,
                                         required=True)
        selective_ml_splits.add_argument("--out_train",
                                         help="Path where the train subset will be saved.", type=str,
                                         required=True)
        selective_ml_splits.add_argument("--out_validation",
                                         help="Path where the validation subset will be saved.", type=str,
                                         required=True)
        selective_ml_splits.add_argument("--out_test",
                                         help="Path where the test subset will be saved.", type=str,
                                         required=True)
        selective_ml_splits.add_argument("-v", "--validation",
                                         help="Path to file with ids of samples that should be in validation set. It must be .csv,.tsv, or .jsonl.",
                                         type=str,
                                         required=True)
        selective_ml_splits.add_argument("-s", "--test",
                                         help="Path to file with ids of samples that should be in test set. It must be .csv,.tsv, or .jsonl.",
                                         type=str,
                                         required=True)
        selective_ml_splits.add_argument("-k", "--key",
                                         help="Name of field that contains sample id in files containing ids that should be selected.",
                                         type=str,
                                         required=True)
        selective_ml_splits.add_argument("--data_key",
                                         help="Name of field that contains sample id in the original dataset.",
                                         type=str,
                                         required=True)
        selective_ml_splits.set_defaults(func=call_make_selective_ml_splits)

        chunking = subparsers.add_parser("chunking", help="Splitting row datasets into chunks.")
        chunking.add_argument("-d", "--data",
                              help="Dataset that should be chunked.", type=str,
                              required=True)
        chunking.add_argument("--out_dir",
                              help="Path where the chunks will be saved.", type=str,
                              required=True)
        size_or_number = chunking.add_mutually_exclusive_group(required=True)
        size_or_number.add_argument("-s", "--size",
                                    help="Size of chunks in lines.", type=int)
        size_or_number.add_argument("-n", "--number_of_chunks",
                                    help="Number of chunks.", type=int)
        chunking.add_argument("--file_name_format",
                              help="Format of file name for chunks. It must contain {counter} placeholder for chunk number. "
                                   "Available placeholders are: {orig_basename}, {counter}, {orig_ext}",
                              type=str,
                              default="{orig_basename}_{counter}{orig_ext}")
        chunking.add_argument("-i", "--index",
                              help="Path to file with line file offsets. It can speed up the process. "
                                   "Must be a file with offsets on separate lines or a csv/tsv file. It is expected that the headline is presented.",
                              type=str,
                              required=False)
        chunking.set_defaults(func=call_chunking)

        byte_chunking_parser = subparsers.add_parser("byte_chunking",
                                                     help="Splitting row datasets into chunks. The chunks are created differently, than with chunks option, as in this case the dataset is split in a way that it tries to achieve the same size of chunks in terms of bytes. Rows are not split.")
        byte_chunking_parser.add_argument("-d", "--data",
                                          help="Dataset that should be chunked.", type=str,
                                          required=True)
        byte_chunking_parser.add_argument("--out_dir",
                                          help="Path where the chunks will be saved.", type=str,
                                          required=True)
        byte_chunking_parser.add_argument("-n", "--number_of_chunks",
                                          help="Number of chunks.", type=int)
        byte_chunking_parser.add_argument("--file_name_format",
                                          help="Format of file name for chunks. It must contain {counter} placeholder for chunk number. "
                                               "Available placeholders are: {orig_basename}, {counter}, {orig_ext}",
                                          type=str,
                                          default="{orig_basename}_{counter}{orig_ext}")
        byte_chunking_parser.set_defaults(func=call_byte_chunking)

        subset_parser = subparsers.add_parser("subset", help="Creates subset of given dataset.")
        subset_parser.add_argument("-d", "--data",
                                   help="Dataset that should be used for subset.", type=str,
                                   required=True)
        subset_parser.add_argument("--out",
                                   help="Path where the subset will be saved.", type=str,
                                   required=True)
        subset_parser.add_argument("-f", "--from_line",
                                   help="Index of first line that should be in subset. First is 0.", type=int,
                                   required=True)
        subset_parser.add_argument("-t", "--to_line",
                                   help="Index of last line that should be in subset. Last is not included.", type=int,
                                   required=True)
        subset_parser.add_argument("-i", "--index",
                                   help="Path to file with line file offsets. It can speed up the process. "
                                        "Must be a file with offsets on separate lines or a csv/tsv file. In case of csv/tsv file do not forget to setup index_offset_field. It is expected that the headline is presented.",
                                   type=str,
                                   required=False)
        subset_parser.add_argument("--index_offset_field",
                                   help="Name of field in index file that contains line offsets.", type=str,
                                   default="file_line_offset",
                                   required=False)
        subset_parser.set_defaults(func=call_subset)

        sample_parser = subparsers.add_parser("sample",
                                              help="Creates a sample of given dataset. The result is written to the stdout.")
        sample_parser.add_argument("data", help="Dataset that should be used for sample.", type=str)
        sample_parser.add_argument("-s", "--size",
                                   help="Size of the sample in lines or size of in percents. Float number in (0,1).",
                                   type=float, required=True)
        sample_parser.add_argument("--fixed_seed", help="Fixes random seed. Useful when you want same splits.",
                                   action='store_true')
        sample_parser.add_argument("-i", "--index",
                                   help="Path to file with line file offsets. It can speed up the process in case when the size number of lines instead of proportion as it will use the index to count lines. "
                                        "Must be a file with offsets on separate lines or a csv/tsv file. In case of csv/tsv file do not forget to setup index_offset_field. It is expected that the headline is presented.",
                                   type=str,
                                   required=False)

        sample_parser.add_argument("--index_offset_field",
                                   help="Name of field in index file that contains line offsets.", type=str,
                                   default="file_line_offset",
                                   required=False)
        sample_parser.set_defaults(func=call_sample)

        shuffle_parser = subparsers.add_parser("shuffle",
                                               help="Shuffles the dataset. The result is written to the stdout.")
        shuffle_parser.add_argument("data", help="Dataset that should be shuffled.", type=str)
        shuffle_parser.add_argument("-i", "--index",
                                    help="Path to file with line file offsets. It can speed up the process in case when the size number of lines instead of proportion as it will use the index to count lines. "
                                         "Must be a file with offsets on separate lines or a csv/tsv file. In case of csv/tsv file do not forget to setup index_offset_field. It is expected that the headline is presented.",
                                    type=str,
                                    required=False)

        shuffle_parser.add_argument("--index_offset_field",
                                    help="Name of field in index file that contains line offsets.", type=str,
                                    default="file_line_offset",
                                    required=False)
        shuffle_parser.add_argument("--fixed_seed", help="Fixes random seed. Useful when you want same splits.",
                                    action='store_true')
        shuffle_parser.set_defaults(func=call_shuffle)

        balance_parser = subparsers.add_parser("balance",
                                               help="Balances the dataset. The result is written to the stdout.")
        balance_parser.add_argument("data", help="Dataset that should be balanced.", type=str)
        balance_parser.add_argument("-f", "--field",
                                    help="Field that should be balanced. This scripts expects that the ", type=str,
                                    required=True)
        balance_parser.add_argument("--fixed_seed", help="Fixes random seed. Useful when you want make the same balance.",
                                    action='store_true')
        balance_parser.set_defaults(func=call_balance)


        subparsers_for_help = {
            'ml_splits': ml_splits,
            'chunking': chunking,
            'byte_chunking': byte_chunking_parser,
            'subset': subset_parser,
            'selective_ml_splits': selective_ml_splits,
            'sample': sample_parser,
            'shuffle': shuffle_parser,
            'balance': balance_parser
        }

        if len(sys.argv) < 2:
            parser.print_help()
            return None
        try:
            parsed, _ = parser.parse_known_args()

        except ArgumentParserError as e:
            for name, subParser in subparsers_for_help.items():
                if name == sys.argv[1]:
                    subParser.print_help()
                    break
            print("\n" + str(e), file=sys.stderr, flush=True)
            return None

        return parsed


def row_split(data: str, out_train: str, out_validation: str, out_test: str, validation: float, test: float, fast: bool,
              fixed_seed: bool, header: bool = False):
    """
    Function for splitting row datasets into train, validation and test sets. The split itself is random, but keep
    in mind that this not changes the order of samples.

    :param data: Dataset that should be split.
    :param out_train: Path where the train subset will be saved.
    :param out_validation: Path where the validation subset will be saved.
    :param out_test: Path where the test subset will be saved.
    :param validation: Size of validation set in percents. Float number in [0,1).
    :param test: Size of test set in percents. Float number in [0,1).
    :param fast: Activates fast splitting strategy that needs only one pass trough original dataset.
        Normally the splitting is done in two phases. Where first counts number of samples and second is writing samples
        according to random generated permutation of original samples. If you use fast strategy please be aware that
        selected sizes will be just approximated, because random selection is used. That means that for every sample we
        randomly decide (probability is given according to a subset size) where it goes
        (sample can go only into a one subset).
    :param fixed_seed: Fixes random seed. Useful when you want same splits.
    :param header: If the dataset has a header. Is considered for csv and tsv datasets.
    """

    if fixed_seed:
        random.seed(0)

    with open(data) as f, open(out_train, "w") as out_train_f, open(out_validation, "w") as out_val_f, open(
            out_test, "w") as out_test_f:

        start_time = time.time()
        i = 0

        reader = f
        csv_writer_train, csv_writer_val, csv_writer_test = None, None, None
        if data.endswith(".csv") or data.endswith(".tsv"):
            delimiter = "\t" if data.endswith(".tsv") else ","
            reader = csv.reader(f, delimiter=delimiter)
            csv_writer_train = csv.writer(out_train_f, delimiter=delimiter)
            csv_writer_val = csv.writer(out_val_f, delimiter=delimiter)
            csv_writer_test = csv.writer(out_test_f, delimiter=delimiter)

            if header:
                header_line = next(reader)

                csv_writer_train.writerow(header_line)
                csv_writer_val.writerow(header_line)
                csv_writer_test.writerow(header_line)

        if fast:
            # Performs fast/probabilistic splitting, only one iteration over dataset is needed and is also more memory efficient.
            # It's rulette wheel selection like approach.
            # For chosen proportions we will be generating random numbers from uniform distribution in [0.0, 1.0) interval.
            # That interval will be splitted into sub-intervals as shown in following example:
            # Example:
            #   validation 10% [0.0, 0.1)
            #   test 10% [0.1, 0.2)
            #   train 80%   [0.2, 1)
            #
            #   Whenever number is generated we will check in which of sub-intervals it belongs.
            #
            logging.info("Starting fast splitting.")
            val_rand_threshold = validation
            test_rand_threshold = val_rand_threshold + test

            start_time = time.time()

            train_samples = 0
            validation_samples = 0
            test_samples = 0

            logging.info("writing")
            for i, line in enumerate(reader):
                rand_num = random.random()
                if rand_num < val_rand_threshold:
                    # this part belongs to validation set
                    if csv_writer_val is not None:
                        csv_writer_val.writerow(line)
                    else:
                        print(line, end="", file=out_val_f)
                    validation_samples += 1
                elif rand_num < test_rand_threshold:
                    # this part belongs to validation set
                    if csv_writer_test is not None:
                        csv_writer_test.writerow(line)
                    else:
                        print(line, end="", file=out_test_f)
                    test_samples += 1
                else:
                    # this part belongs to train set
                    if csv_writer_train is not None:
                        csv_writer_train.writerow(line)
                    else:
                        print(line, end="", file=out_train_f)
                    train_samples += 1

                if time.time() - start_time > 10:
                    start_time = time.time()
                    logging.info("\twritten: {}".format(i + 1))

            rest_for_train = 1.0 - validation - test

            act_frac_train = train_samples / (i + 1)
            act_frac_val = validation_samples / (i + 1)
            act_frac_test = test_samples / (i + 1)

            logging.info(
                "Number of train samples is {} which is {:.2%}, so the difference to the correct value is {}."
                .format(train_samples, act_frac_train, rest_for_train - act_frac_train))
            logging.info(
                "Number of validation samples is {} which is {:.2%}, so the difference to the correct value is {}."
                .format(validation_samples, act_frac_val, validation - act_frac_val))
            logging.info(
                "Number of test samples is {} which is {:.2%}, so the difference to the correct value is {}."
                .format(test_samples, act_frac_test, test - act_frac_test))

        else:
            logging.info("Starting slow splitting.")
            logging.info("Line count starts.")
            # Regular splitting, requires two iterations over dataset.

            line_count = sum(1 for _ in reader)
            f.seek(0)

            if header and (data.endswith(".csv") or data.endswith(".tsv")):
                next(f)  # skip header

            logging.info("Number of original dataset samples: {}".format(line_count))

            indices = list(range(line_count))
            random.shuffle(indices)

            logging.info("Splitting")

            validation_offset = round(line_count * validation)
            test_offset = validation_offset + round(line_count * test)

            validation_indices = set(indices[:validation_offset])
            test_indices = set(indices[validation_offset:test_offset])

            del indices

            logging.info(
                "Number of train samples: {}".format(line_count - len(validation_indices) - len(test_indices)))
            logging.info("Number of validation samples: {}".format(len(validation_indices)))
            logging.info("Number of test samples: {}".format(len(test_indices)))

            logging.info("writing")

            for i, line in enumerate(reader):
                if i in validation_indices:
                    if csv_writer_val is not None:
                        csv_writer_val.writerow(line)
                    else:
                        print(line, end="", file=out_val_f)
                elif i in test_indices:
                    if csv_writer_test is not None:
                        csv_writer_test.writerow(line)
                    else:
                        print(line, end="", file=out_test_f)
                else:
                    # everything else is train set
                    if csv_writer_train is not None:
                        csv_writer_train.writerow(line)
                    else:
                        print(line, end="", file=out_train_f)

                if time.time() - start_time > 10:
                    start_time = time.time()
                    logging.info("\twritten: {}".format(i + 1))

        logging.info("written: {}".format(i + 1))

        logging.info("finished")


def call_make_ml_splits(args: argparse.Namespace):
    """
    Splitting row datasets into train, validation and test sets.

    :param args: User arguments.
    """
    assert 0 <= args.validation < 1
    assert 0 <= args.test < 1
    assert args.test + args.validation < 1

    row_split(args.data, args.outTrain, args.outValidation, args.outTest, args.validation, args.test, args.fast,
              args.fixedSeed, args.header)


def chunking(data: str, out_dir: str, size: int = None, number_of_chunks: int = None, file_name_format: str = None,
             index: str = None):
    """
    Splitting row datasets into chunks.

    :param data: Dataset that should be chunked.
    :param out_dir: Path where the chunks will be saved.
    :param size: Size of chunks in lines.
    :param number_of_chunks: Number of chunks.
    :param file_name_format: Format of file name for chunks. It must contain {counter} placeholder for chunk number.
        Available placeholders are: {orig_basename}, {counter}, {orig_ext}

    :param index: Path to file with line file offsets. It can speed up the process. Must be a file with offsets on
        separate lines or a csv file. In case of csv file do not forget to setup index_offset_field. It is expected that the headline is presented.
    """

    lines_per_chunk = size
    if lines_per_chunk is not None:
        assert lines_per_chunk > 0, "Size of chunks must be greater than 0."
    else:
        assert number_of_chunks > 0, "Number of chunks must be greater than 0."

        logging.info("Counting lines in dataset.")
        lines_in_dataset = 0
        if index is not None:
            with open(index, "rb") as f, tqdm(total=os.path.getsize(index)) as pbar:
                while f.readline():
                    lines_in_dataset += 1
                    pbar.update(f.tell() - pbar.n)

            if index.endswith(".csv") or index.endswith(".tsv"):
                lines_in_dataset -= 1  # header

        else:
            with open(data, "rb") as f, tqdm(total=os.path.getsize(data)) as pbar:
                while f.readline():
                    lines_in_dataset += 1
                    pbar.update(f.tell() - pbar.n)

        lines_per_chunk = math.ceil(lines_in_dataset / number_of_chunks)

    logging.info(f"Lines per chunk: {lines_per_chunk}")

    counter = 0
    p = Path(data)
    orig_basename = p.stem
    orig_ext = p.suffix

    chunk_file = open(
        os.path.join(out_dir,
                     file_name_format.format(orig_basename=orig_basename,
                                             counter=counter,
                                             orig_ext=orig_ext)
                     ), "w"
    )
    try:
        with open(data) as f:
            for line_number, line in enumerate(f):
                if line_number % lines_per_chunk == 0 and line_number != 0:
                    chunk_file.close()
                    counter += 1
                    chunk_file = open(
                        os.path.join(out_dir,
                                     file_name_format.format(orig_basename=orig_basename,
                                                             counter=counter,
                                                             orig_ext=orig_ext)
                                     ), "w"
                    )
                print(line, end="", file=chunk_file)
    finally:
        chunk_file.close()


def call_chunking(args: argparse.Namespace):
    """
    Splitting row datasets into chunks.

    :param args: User arguments.
    """

    chunking(args.data, args.out_dir, args.size, args.number_of_chunks, args.file_name_format, args.index)


def byte_chunking(data: str, out_dir: str, number_of_chunks: int, file_name_format: str):
    """
    Splitting row datasets into chunks. The chunks are created differently, than with chunks option, as in this case the dataset is split in a way that it tries to achieve the same size of chunks in terms of bytes. Rows are not split.

    :param data: Dataset that should be chunked.
    :param out_dir: Path where the chunks will be saved.
    :param number_of_chunks: Number of chunks.
    :param file_name_format: Format of file name for chunks. It must contain {counter} placeholder for chunk number.
        Available placeholders are: {orig_basename}, {counter}, {orig_ext}
    """

    with open(data, "rb") as f:
        data_size = os.path.getsize(data)
        chunk_size = math.ceil(data_size / number_of_chunks)

        counter = 0
        p = Path(data)
        orig_basename = p.stem
        orig_ext = p.suffix

        chunk_file = open(
            os.path.join(out_dir,
                         file_name_format.format(orig_basename=orig_basename,
                                                 counter=counter,
                                                 orig_ext=orig_ext)
                         ), "wb"
        )
        try:
            chunk_size_counter = 0

            for line in f:
                if chunk_size_counter + len(line) > chunk_size and counter < (number_of_chunks - 1):
                    chunk_file.close()
                    counter += 1
                    chunk_file = open(
                        os.path.join(out_dir,
                                     file_name_format.format(orig_basename=orig_basename,
                                                             counter=counter,
                                                             orig_ext=orig_ext)
                                     ), "wb"
                    )
                    chunk_size_counter = 0

                chunk_size_counter += len(line)
                chunk_file.write(line)
        finally:
            chunk_file.close()


def call_byte_chunking(args: argparse.Namespace):
    """
    Splitting row datasets into chunks. The chunks are created differently, than with chunks option, as in this case the dataset is split in a way that it tries to achieve the same size of chunks in terms of bytes. Rows are not split.

    :param args: User arguments.
    """

    byte_chunking(args.data, args.out_dir, args.number_of_chunks, args.file_name_format)


def subset(data: str, out: str, from_line: int, to_line: int, index: str = None, index_offset_field: str = None):
    """
    Creates subset of given dataset.

    :param data: Dataset that should be used for subset.
    :param out: Path where the subset will be saved.
    :param from_line: Index of first line that should be in subset. First is 0.
    :param to_line: Index of last line that should be in subset. Last is not included.
    :param index: Path to file with line file offsets. It can speed up the process. Must be a file with offsets on
        separate lines or a csv file. It is expected that the headline is presented.
    :param index_offset_field: Name of field in index file that contains line offsets.
    """

    assert from_line < to_line, "From index must be smaller than to index."
    with open(data) as f_dataset, open(out, "w") as f_out:
        # let's skip to the first line

        if index is not None:
            from_offset = 0
            with open(index) as f_index:
                reading_csv = index.endswith(".csv") or index.endswith(".tsv")
                reader = csv.DictReader(f_index,
                                        delimiter="\t" if index.endswith(".tsv") else ",") if reading_csv else f_index

                for i, line in enumerate(tqdm(reader, total=from_line, desc="Searching offset of first line in index")):
                    if i == from_line:
                        from_offset = int(line[index_offset_field]) if reading_csv else int(line)
                        break

            f_dataset.seek(from_offset)
        else:
            for _ in tqdm(range(from_line), desc="Skipping to the first line"):
                f_dataset.readline()

        # now we can start writing
        for _ in range(from_line, to_line):
            print(f_dataset.readline(), end="", file=f_out)


def call_subset(args: argparse.Namespace):
    """
    Creates subset of given dataset.

    :param args: User arguments.
    """

    subset(args.data, args.out, args.from_line, args.to_line, args.index, args.index_offset_field)


def make_selective_ml_splits(data: str, out_train: str, out_validation: str, out_test: str, validation: str, test: str,
                             key: str, data_key: str):
    """
    Splitting row datasets into train, validation and test sets. According to given set of ids.

    :param data: Dataset that should be split. It must be .csv,.tsv, or .jsonl.
    :param out_train: Path where the train subset will be saved.
    :param out_validation: Path where the validation subset will be saved.
    :param out_test: Path where the test subset will be saved.
    :param validation: Path to file with ids of samples that should be in validation set. It must be .csv,.tsv, or .jsonl.
    :param test: Path to file with ids of samples that should be in test set. It must be .csv,.tsv, or .jsonl.
    :param key: Name of field that contains sample id in files containing ids that should be selected.
    :param data_key: Name of field that contains sample id in the original dataset.
    """

    with (open(data) as f_data, FileReader(validation) as f_val, FileReader(test) as f_test, \
          open(out_train, "w") as out_train_f, open(out_validation, "w") as out_val_f,
          open(out_test, "w") as out_test_f):

        # let's select ids and convert them to string as we don't want to distinguish between 1 and "1"
        # at the same time we want to allow non initeger ids, thus it is better choice to convert them to string
        # instead of int

        val_ids = set(str(row[key]) for row in f_val)
        test_ids = set(str(row[key]) for row in f_test)

        with tqdm(total=os.path.getsize(data)) as pbar:
            while line := f_data.readline():
                pbar.update(f_data.tell() - pbar.n)

                row = json.loads(line)

                if str(row[data_key]) in val_ids:
                    print(line, end="", file=out_val_f)
                elif str(row[data_key]) in test_ids:
                    print(line, end="", file=out_test_f)
                else:
                    # everything else is train set
                    print(line, end="", file=out_train_f)


def call_make_selective_ml_splits(args: argparse.Namespace):
    """
    Splitting row datasets into train, validation and test sets. According to given set of ids.
    """

    make_selective_ml_splits(args.data, args.out_train, args.out_validation, args.out_test, args.validation, args.test,
                             args.key, args.data_key)


def obtain_line_offsets(file_path: str) -> list[int]:
    """
    Obtains line offsets from given file and writes them to the output file.

    :param file_path: Path to the file.
    """

    line_offsets = []
    with open(file_path, "rb") as f:
        with tqdm(total=os.path.getsize(file_path)) as pbar:
            while f.readline():
                line_offsets.append(pbar.n)
                pbar.update(f.tell() - pbar.n)

    return line_offsets


def line_offsets_from_index(index: str, index_offset_field: str, force_csv: bool = False) -> list[int]:
    """
    Obtains line offsets from given index file.

    :param index: Path to file with line file offsets. It can speed up the process. Must be a file with offsets on
        separate lines or a csv file. It is expected that the headline is presented.
    :param index_offset_field: Name of field in index file that contains line offsets.
    :param force_csv: Forces the index to be treated as csv file.
    """

    line_offsets = []
    with open(index) as f:
        if force_csv or index.endswith(".csv") or index.endswith(".tsv"):
            if not (index.endswith(".csv") or index.endswith(".tsv")):
                # look at header to determine delimiter
                header = f.readline()
                delimiter = "\t" if "\t" in header else ","
                f.seek(0)
            else:
                delimiter = "\t" if index.endswith(".tsv") else ","
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in tqdm(reader, desc="Obtaining line offsets from index"):
                line_offsets.append(int(row[index_offset_field]))
        else:
            try:
                for line in tqdm(f, desc="Obtaining line offsets from index"):
                    line_offsets.append(int(line))
            except ValueError:
                return line_offsets_from_index(index, index_offset_field, True)

    return line_offsets


def sample(data: str, size: float, fixed_seed: bool, index: str = None, index_offset_field: str = None):
    """
    Creates a sample of given dataset.

    :param data: Dataset that should be used for sample.
    :param size: Size of the sample in lines or size of in percents. Float number in [0,1).
    :param fixed_seed: Fixes random seed. Useful when you want same splits.
    :param index: Path to file with line file offsets. It can speed up the process in case when the size number of lines instead of proportion as it will use the index to count lines.
        Must be a file with offsets on separate lines or a csv file. In case of csv file do not forget to setup index_offset_field. It is expected that the headline is presented.
    :param index_offset_field: Name of field in index file that contains line offsets.
    """
    assert size > 0, "Size of the sample must be greater than 0."
    if fixed_seed:
        random.seed(0)

    if size < 1:
        with open(data) as f:
            for line in f:
                if random.random() < size:
                    print(line, end="")
    else:
        line_offsets = line_offsets_from_index(index, index_offset_field) if index is not None else obtain_line_offsets(
            data)

        random.shuffle(line_offsets)
        selected_lines = line_offsets[:int(size)]
        with open(data) as f:
            for line_offset in selected_lines:
                f.seek(line_offset)
                print(f.readline(), end="")


def call_sample(args: argparse.Namespace):
    """
    Creates a sample of given dataset.

    :param args: User arguments.
    """
    sample(args.data, args.size, args.fixed_seed, args.index, args.index_offset_field)


def shuffle(data: str, index: str = None, index_offset_field: str = None, fixed_seed: bool = True):
    """
    Shuffles the dataset.

    :param data: Dataset that should be shuffled.
    :param index: Path to file with line file offsets. It can speed up the process in case when the size number of lines instead of proportion as it will use the index to count lines.
        Must be a file with offsets on separate lines or a csv file. In case of csv file do not forget to setup index_offset_field. It is expected that the headline is presented.
    :param index_offset_field: Name of field in index file that contains line offsets.
    :param fixed_seed: Fixes random seed. Useful when you want same splits.
    """
    if fixed_seed:
        random.seed(0)

    line_offsets = line_offsets_from_index(index, index_offset_field) if index is not None else obtain_line_offsets(
        data)

    random.shuffle(line_offsets)
    with open(data) as f:
        for line_offset in line_offsets:
            f.seek(line_offset)
            print(f.readline(), end="")


def call_shuffle(args: argparse.Namespace):
    """
    Shuffles the dataset.

    :param args: User arguments.
    """
    shuffle(args.data, args.index, args.index_offset_field, args.fixed_seed)


def balance(data: str, field: str, fixed_seed: bool = True):
    """
    Balances the dataset.

    :param data: Dataset that should be balanced.
    :param field: Field that should be balanced.
    :param fixed_seed: Fixes random seed. Useful when you want make the same balance.
    """
    if fixed_seed:
        random.seed(0)

    with open(data) as f:
        samples_per_category = defaultdict(list)

        file_offset = 0
        while line := f.readline():
            row = json.loads(line)
            samples_per_category[row[field]].append(file_offset)
            file_offset = f.tell()

        print("There are:", file=sys.stderr)
        for category, samples in sorted(samples_per_category.items(), key=lambda x: x[0]):
            print(f"\t{category}: {len(samples)}", file=sys.stderr)

        min_samples = min(len(samples) for samples in samples_per_category.values())

        print(f"Balancing to {min_samples} samples per category.", file=sys.stderr)

        # select min_samples samples from each category and shuffle them
        # we want to remain the original order of samples
        allowed_offsets = []
        for samples in samples_per_category.values():
            random.shuffle(samples)
            allowed_offsets.extend(samples[:min_samples])

        allowed_offsets.sort()

        for offset in allowed_offsets:
            f.seek(offset)
            print(f.readline(), end="")


def call_balance(args: argparse.Namespace):
    """
    Balances the dataset.

    :param args: User arguments.
    """
    balance(args.data, args.field, args.fixed_seed)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = ArgumentsManager.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
