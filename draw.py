#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw pictures')
    parser.add_argument('-c', '--clusters', type=int, help='number of clusters', required=True)
    parser.add_argument('-f', '--filename', type=str, help='filename from the inputs directory', required=True)
    args, _ = parser.parse_known_args()

    # draw settings
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Before and After K-Means Clustering')

    # before
    df = pd.read_csv(os.path.join('inputs', args.filename), delim_whitespace=True, header=None)
    df.columns = ['x', 'y']
    sns.scatterplot(ax=ax[0], x=df['x'], y=df['y'])
    ax[0].set_title('Before clustering')

    # after
    df = pd.read_csv(os.path.join('outputs', f'{args.filename}.out'), delim_whitespace=True, header=None)
    df.columns = ['x', 'y', 'cluster']
    sns.scatterplot(ax=ax[1], x=df['x'], y=df['y'], hue=df['cluster'], palette=sns.color_palette('hls', n_colors=args.clusters))
    ax[1].set_title('After clustering')

    plt.tight_layout()
    plt.show()
