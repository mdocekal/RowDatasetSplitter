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
                            help="Dataset that should be splitted.", type=str,
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


def main():
    args = ArgumentsManager.parseArgs()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args is not None:
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

    else:
        exit(1)


if __name__ == '__main__':
    main()
