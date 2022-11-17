# K-means Clustering using OpenMP

## Description
- **generate.py**: default NUMS = 1000, MAXIMUM = 5000, FILENAME = 'data.txt' 
```console
usage: generate.py [-h] [-n NUMS] [-m MAXIMUM] [-f FILENAME]

Randomly generate 2d coordinates and store in the inputs directory.

optional arguments:
  -h, --help                        show this help message and exit
  -n NUMS, --nums NUMS              generate <NUMS> points
  -m MAXIMUM, --maximum MAXIMUM     set the 2d coordinate to range between 0 and <MAXIMUM>
  -f FILENAME, --filename FILENAME  store the data in the inputs directory and named <FILENAME>
```

- **kmeans**: default N = 1000
```console
usage: ./kmeans [-h] [-n N] -f FILENAME -c M

optional arguments:
  -h --help                     show this help message and exit
  -f --filename <FILENAME>      <FILENAME> in the inputs directory
  -c --clusters <M>             set <M> clusters
  -n --iterations <N>           set <N> iterations
```

- **draw.py**
```console
usage: draw.py [-h] -c CLUSTERS -f FILENAME

Draw the scatterplots before and after K-means Clustering.

optional arguments:
  -h, --help                        show this help message and exit
  -c CLUSTERS, --clusters CLUSTERS  number of clusters
  -f FILENAME, --filename FILENAME  <FILENAME> from the inputs directory
```

## Execution
Install Python required packages and generate the executable file **kmeans**.
```bash
$ pip3 install -r requirements.txt; make 
```

Randomly generate 2d coordinates and store in the inputs directory.
```bash
$ python3 generate.py [-n NUMS] [-m MAXIMUM] [-f FILENAME]
```

Do K-means Clustering.
```bash
$ ./kmeans [-h] [-n N] -f FILENAME -c M
```

Draw the scatterplots before and after K-means Clustering if you would like to see the result.
```bash
$ python3 draw.py -c CLUSTERS -f FILENAME
```

## References
