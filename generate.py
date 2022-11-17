#!/usr/bin/python3

import os
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Randomly generate 2d coordinates and store in the inputs directory.')
    parser.add_argument('-n', '--nums', type=int, help='generate <NUMS> points')
    parser.add_argument('-m', '--maximum', type=int, help='set the 2d coordinate to range between 0 and <MAXIMUM>')
    parser.add_argument('-f', '--filename', type=str, help='store the data in the inputs directory and named <FILENAME>')
    args, _ = parser.parse_known_args()

    if os.path.exists('inputs') == False:
        os.mkdir('inputs')

    nums = 1000 if args.nums is None else args.nums
    maximum = 5000 if args.maximum is None else args.maximum
    filename = os.path.join('inputs', 'data.txt' if args.filename is None else args.filename)

    with open(filename, 'w') as f:
        for i in range(nums):
            p = (random.uniform(0, maximum), random.uniform(0, maximum))
            f.write('%10.3f %10.3f\n' %p)
