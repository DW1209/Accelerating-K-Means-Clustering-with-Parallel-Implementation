#!/usr/bin/python3

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw the scatterplots before and after K-Means Clustering.')
    parser.add_argument('-c', '--clusters', type=int, help='classify the data into <CLUSTERS> groups')
    parser.add_argument('-f', '--filename', type=str, help='<FILENAME> from the inputs directory')
    args, _ = parser.parse_known_args()

    clusters = 3 if args.clusters is None else args.clusters
    filename = 'data.txt' if args.filename is None else args.filename

    # draw settings
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Before and After K-Means Clustering')

    # before
    df = pd.read_csv(os.path.join('inputs', filename), delim_whitespace=True, header=None)
    df.columns = ['x', 'y']
    sns.scatterplot(ax=ax[0], x=df['x'], y=df['y'])
    ax[0].set_title('Before Clustering')

    # after
    df = pd.read_csv(os.path.join('outputs', f'{filename}.out'), delim_whitespace=True, header=None)
    df.columns = ['x', 'y', 'cluster']
    sns.scatterplot(ax=ax[1], x=df['x'], y=df['y'], hue=df['cluster'], palette=sns.color_palette('hls', n_colors=clusters))
    ax[1].set_title('After Clustering')

    plt.tight_layout()
    plt.show()
