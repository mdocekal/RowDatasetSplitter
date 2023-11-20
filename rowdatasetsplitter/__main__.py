# -*- coding: UTF-8 -*-
""""
Created on 01.07.20
RowDatasetSplitter
Small script for splitting row datasets into train, validation and test sets.

:author:     Martin Dočekal
"""
import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from argparse import ArgumentParser
from pathlib import Path


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
    def parseArgs(cls):
        """
        Performs arguments parsing.

        :param cls: arguments class
        :returns: Parsed arguments.
        """

        parser = ExceptionsArgumentParser(
            description="Script for splitting row datasets.")

        subparsers = parser.add_subparsers()

        ml_splits = subparsers.add_parser("ml_splits",
                                          help="Splitting row datasets into train, validation and test sets.")

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

        ml_splits.set_defaults(func=call_make_ml_splits)

        chunking = subparsers.add_parser("chunking", help="Splitting row datasets into chunks.")
        chunking.add_argument("-d", "--data",
                              help="Dataset that should be split.", type=str,
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
                                   "Must be a file with offsets on separate lines or a csv file. In case of csv file do not forget to setup index_offset_field. It is expected that the headline is presented.",
                              type=str,
                              required=False)
        chunking.add_argument("--index_offset_field",
                              help="Name of field in index file that contains line offsets.", type=str,
                              default="file_line_offset",
                              required=False)
        chunking.set_defaults(func=call_chunking)

        subset = subparsers.add_parser("subset", help="Creates subset of given dataset.")
        subset.add_argument("-d", "--data",
                            help="Dataset that should be split.", type=str,
                            required=True)
        subset.add_argument("--out",
                            help="Path where the subset will be saved.", type=str,
                            required=True)
        subset.add_argument("-f", "--from_line",
                            help="Index of first line that should be in subset. First is 0.", type=int,
                            required=True)
        subset.add_argument("-t", "--to_line",
                            help="Index of last line that should be in subset. Last is not included.", type=int,
                            required=True)
        subset.add_argument("-i", "--index",
                            help="Path to file with line file offsets. It can speed up the process. "
                                 "Must be a file with offsets on separate lines or a csv file. In case of csv file do not forget to setup index_offset_field. It is expected that the headline is presented.",
                            type=str,
                            required=False)
        subset.add_argument("--index_offset_field",
                            help="Name of field in index file that contains line offsets.", type=str,
                            default="file_line_offset",
                            required=False)
        subset.set_defaults(func=call_subset)

        subparsers_for_help = {
            'ml_splits': ml_splits,
            'chunking': chunking,
            'subset': subset
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


def call_make_ml_splits(args: argparse.Namespace):
    """
    Splitting row datasets into train, validation and test sets.

    :param args: User arguments.
    """

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    assert 0 <= args.validation < 1
    assert 0 <= args.test < 1
    assert args.test + args.validation < 1

    if args.fixedSeed:
        random.seed(0)

    with open(args.data) as f, open(args.outTrain, "w") as outTrain, open(args.outValidation, "w") as outVal, open(
            args.outTest, "w") as outTest:

        startTime = time.time()
        i = 0
        if args.fast:
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
            valRandThreshold = args.validation
            testRandThreshold = valRandThreshold + args.test

            startTime = time.time()

            trainSamples = 0
            validationSamples = 0
            testSamples = 0

            logging.info("writing")
            for i, line in enumerate(f):
                randNum = random.random()
                if randNum < valRandThreshold:
                    # this part belongs to validation set
                    print(line, end="", file=outVal)
                    validationSamples += 1
                elif randNum < testRandThreshold:
                    # this part belongs to validation set
                    print(line, end="", file=outTest)
                    testSamples += 1
                else:
                    # this part belongs to train set
                    print(line, end="", file=outTrain)
                    trainSamples += 1

                if time.time() - startTime > 10:
                    startTime = time.time()
                    logging.info("\twritten: {}".format(i + 1))

            restForTrain = 1.0 - args.validation - args.test

            actFracTrain = trainSamples / (i + 1)
            actFracVal = validationSamples / (i + 1)
            actFracTest = testSamples / (i + 1)

            logging.info(
                "Number of train samples is {} which is {:.2%}, so the difference to the correct value is {}."
                .format(trainSamples, actFracTrain, restForTrain - actFracTrain))
            logging.info(
                "Number of validation samples is {} which is {:.2%}, so the difference to the correct value is {}."
                .format(validationSamples, actFracVal, args.validation - actFracVal))
            logging.info(
                "Number of test samples is {} which is {:.2%}, so the difference to the correct value is {}."
                .format(testSamples, actFracTest, args.test - actFracTest))

        else:
            logging.info("Starting slow splitting.")
            logging.info("Line count starts.")
            # Regular splitting, requires two iterations over dataset.
            lineCount = sum(1 for _ in f)
            f.seek(0)

            logging.info("Number of original dataset samples: {}".format(lineCount))

            indices = list(range(lineCount))
            random.shuffle(indices)

            logging.info("Splitting")

            validationOffset = round(lineCount * args.validation)
            testOffset = validationOffset + round(lineCount * args.test)

            validationIndices = set(indices[:validationOffset])
            testIndices = set(indices[validationOffset:testOffset])

            del indices

            logging.info(
                "Number of train samples: {}".format(lineCount - len(validationIndices) - len(testIndices)))
            logging.info("Number of validation samples: {}".format(len(validationIndices)))
            logging.info("Number of test samples: {}".format(len(testIndices)))

            logging.info("writing")

            for i, line in enumerate(f):
                if i in validationIndices:
                    print(line, end="", file=outVal)
                elif i in testIndices:
                    print(line, end="", file=outTest)
                else:
                    # everything else is train set
                    print(line, end="", file=outTrain)

                if time.time() - startTime > 10:
                    startTime = time.time()
                    logging.info("\twritten: {}".format(i + 1))

        logging.info("written: {}".format(i + 1))

        logging.info("finished")


def call_chunking(args: argparse.Namespace):
    """
    Splitting row datasets into chunks.

    :param args: User arguments.
    """

    lines_per_chunk = args.size
    if lines_per_chunk is not None:
        assert lines_per_chunk > 0, "Size of chunks must be greater than 0."
    else:
        assert args.number_of_chunks > 0, "Number of chunks must be greater than 0."

        logging.info("Counting lines in dataset.")
        lines_in_dataset = 0
        if args.index is not None:
            with open(args.index) as f:
                for _ in f:
                    lines_in_dataset += 1

            if args.index.endswith(".csv"):
                lines_in_dataset -= 1  # header

        else:
            with open(args.data) as f:
                for _ in f:
                    lines_in_dataset += 1

        lines_per_chunk = math.ceil(lines_in_dataset / args.number_of_chunks)

    logging.info(f"Lines per chunk: {lines_per_chunk}")

    counter = 0
    p = Path(args.data)
    orig_basename = p.stem
    orig_ext = p.suffix

    chunk_file = open(
        os.path.join(args.out_dir,
                     args.file_name_format.format(orig_basename=orig_basename,
                                                  counter=counter,
                                                  orig_ext=orig_ext)
                     ), "w"
    )
    try:
        with open(args.data) as f:
            for line_number, line in enumerate(f):
                if line_number % lines_per_chunk == 0 and line_number != 0:
                    chunk_file.close()
                    counter += 1
                    chunk_file = open(
                        os.path.join(args.out_dir,
                                     args.file_name_format.format(orig_basename=orig_basename,
                                                                  counter=counter,
                                                                  orig_ext=orig_ext)
                                     ), "w"
                    )
                print(line, end="", file=chunk_file)
    finally:
        chunk_file.close()


def call_subset(args: argparse.Namespace):
    """
    Creates subset of given dataset.

    :param args: User arguments.
    """

    assert args.from_line < args.to_line, "From index must be smaller than to index."
    with open(args.data) as f_dataset, open(args.out, "w") as f_out:
        # let's skip to the first line

        if args.index is not None:
            from_offset = 0
            with open(args.index) as f_index:
                reading_csv = args.index.endswith(".csv")
                reader = csv.DictReader(f_index) if reading_csv else f_index

                for i, line in enumerate(reader):
                    if i == args.from_line:
                        from_offset = int(line[args.index_offset_field]) if reading_csv else int(line)
                        break

            f_dataset.seek(from_offset)
        else:
            for _ in range(args.from_line):
                f_dataset.readline()

        # now we can start writing
        for _ in range(args.from_line, args.to_line):
            print(f_dataset.readline(), end="", file=f_out)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    args = ArgumentsManager.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
