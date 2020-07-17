#!/usr/bin/env python3
"""Splits data into `micro` and `baseline_test` according to a 90-10 proportion."""


import argparse
import os
import subprocess


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="path to full data set")
    args = parser.parse_args()

os.chdir('..')
os.chdir('data')

# retrieves number of lines in full data set
data_info = subprocess.Popen(['wc', args.file_path], stdout=subprocess.PIPE).communicate()[0]
full_len = int(data_info.split()[0]) 

# computes the number of lines in 90% of full data set
micro_len = round(.9*full_len)

# splits the full data set into a 90-10 proportion
subprocess.check_call(['split', '-l', f'{micro_len}', args.file_path])

# renames the output two output files
os.rename('xaa', 'micro.txt')
os.rename('xab', 'baseline_test.txt')

# retrieves the number of lines in `baseline_test.txt`
baseline_test_info = subprocess.Popen(['wc', 'baseline_test.txt'], stdout=subprocess.PIPE).communicate()[0]
baseline_test_len = int(baseline_test_info.split()[0])

# tests to make sure that no data was omitted from split files.
assert(micro_len + baseline_test_len == full_len)

print(f'Number of lines in full data set: {full_len}.')
print(f'Number of lines in train, dev and test data sets for ind. clsfr training: {micro_len}, or {(micro_len/full_len)*100}% of full data set.')
print(f'Number of lines in baseline test set: {baseline_test_len}, or {(baseline_test_len/full_len)*100}% of full data set.')