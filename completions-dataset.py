import os
import argparse
import logging
import json

from market_sentiment.data_utils import *


parser = argparse.ArgumentParser(description="Convert Label Studio Completions to Financial Phrase Bank dataset")
parser.add_argument("--completions-dir",
                    help="directory for completions",
                    default="output/completions/")
parser.add_argument("--output-file",
                    help="text file for label studio file",
                    default="output/completions/")
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")


def main():
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with open(args.output_file, 'a') as dataset_file:
        # Iterate all files 
        for dirpath, dnames, fnames in os.walk(args.completions_dir):
            for f in fnames:
                with open(os.path.join(dirpath, f)) as completions_file:
                    completions_data = json.load(completions_file)
                    example = completions_data['data']['text']
                    # Assumption: last completion is newest
                    label = completions_data['completions'][-1]['result'][0]['value']['choices'][0].lower()
                    dataset_file.write("{example}@{label}\n".format(example=example, label=label))

if __name__ == "__main__":
    main()