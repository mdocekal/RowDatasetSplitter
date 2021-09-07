# -*- coding: UTF-8 -*-
""""
Created on 01.07.20
RowDatasetSplitter
Small script for splitting row datasets into train, validation and test sets.

:author:     Martin Dočekal
"""
import logging
import random
import sys
import time
from argparse import ArgumentParser


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
            description="Small script for splitting row datasets into train, validation and test sets. "
                        "The split itself is random, but keep in mind that this not changes the order of samples.")

        parser.add_argument("-d", "--data",
                            help="Dataset that should be split.", type=str,
                            required=True)
        parser.add_argument("--outTrain",
                            help="Path where the train subset will be saved.", type=str,
                            required=True)
        parser.add_argument("--outValidation",
                            help="Path where the validation subset will be saved.", type=str,
                            required=True)
        parser.add_argument("--outTest",
                            help="Path where the test subset will be saved.", type=str,
                            required=True)
        parser.add_argument("-v", "--validation",
                            help="Size of validation set in percents. Float number in [0,1).", type=float,
                            required=True)
        parser.add_argument("-s", "--test",
                            help="Size of test set in percents. Float number in [0,1).", type=float,
                            required=True)
        parser.add_argument("-f", "--fast",
                            help="Activates fast splitting strategy that needs only one pass trough original dataset. "
                                 "Normally the splitting is done in two phases. Where first counts number of samples and "
                                 "second is writing samples according to random generated permutation of original samples. "
                                 "If you use fast strategy please be aware that selected sizes will be just approximated, "
                                 "because random selection is used. That means that for every sample we randomly decide "
                                 "(probability is given according to a subset size) where it goes (sample can go only into a "
                                 "one subset).",
                            action='store_true')
        parser.add_argument("--fixedSeed",
                            help="Fixes random seed. Useful when you want same splits.", action='store_true')

        if len(sys.argv) < 2:
            parser.print_help()
            return None
        try:
            parsed = parser.parse_args()

        except ArgumentParserError as e:
            parser.print_help()
            print("\n" + str(e), file=sys.stdout, flush=True)
            return None

        return parsed


def row_split(data:str, out_train:str, out_validation:str, out_test:str, validation: float, test:float, fast: bool,
              fixed_seed: bool):
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
    """

    if fixed_seed:
        random.seed(0)

    with open(data) as f, open(out_train, "w") as out_train_f, open(out_validation, "w") as out_val_f, open(
            out_test, "w") as out_test_f:

        start_time = time.time()
        i = 0
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
            for i, line in enumerate(f):
                rand_num = random.random()
                if rand_num < val_rand_threshold:
                    # this part belongs to validation set
                    print(line, end="", file=out_val_f)
                    validation_samples += 1
                elif rand_num < test_rand_threshold:
                    # this part belongs to validation set
                    print(line, end="", file=out_test_f)
                    test_samples += 1
                else:
                    # this part belongs to train set
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
            line_count = sum(1 for _ in f)
            f.seek(0)

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

            for i, line in enumerate(f):
                if i in validation_indices:
                    print(line, end="", file=out_val_f)
                elif i in test_indices:
                    print(line, end="", file=out_test_f)
                else:
                    # everything else is train set
                    print(line, end="", file=out_train_f)

                if time.time() - start_time > 10:
                    start_time = time.time()
                    logging.info("\twritten: {}".format(i + 1))

        logging.info("written: {}".format(i + 1))

        logging.info("finished")


def main():
    args = ArgumentsManager.parseArgs()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args is not None:
        assert 0 <= args.validation < 1
        assert 0 <= args.test < 1
        assert args.test + args.validation < 1

        row_split(args.data, args.outTrain, args.outValidation, args.outTest, args.validation, args.test, args.fast,
                  args.fixedSeed)

    else:
        exit(1)


if __name__ == '__main__':
    main()
