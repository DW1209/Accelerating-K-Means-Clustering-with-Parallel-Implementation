#!/usr/bin/env python3

import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Randomly generate 2d coordinates')
    parser.add_argument('-n', '--nums', type=int, help='total nums of points')
    parser.add_argument('-m', '--maximum', type=int, help='2d coordinate maximum')
    parser.add_argument('-f', '--filename', type=str, help='filename to store points')
    args, _ = parser.parse_known_args()

    nums = 1000 if args.nums is None else args.nums
    maximum = 5000 if args.maximum is None else args.maximum
    filename = 'data.txt' if args.filename is None else args.filename

    with open(filename, 'w') as f:
        for i in range(nums):
            p = (random.uniform(0, maximum), random.uniform(0, maximum))
            f.write('%10.3f %10.3f\n' %p)
