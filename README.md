# K-Means Clustering using OpenMP

## Description
- **generate.py**: default NUMS = 10000, MAXIMUM = 1000000, FILENAME = 'data.txt' 
```console
usage: generate.py [-h] [-n NUMS] [-m MAXIMUM] [-f FILENAME]

Randomly generate 2d coordinates and store in the inputs directory.

optional arguments:
  -h, --help                        show this help message and exit
  -n NUMS, --nums NUMS              generate <NUMS> points
  -m MAXIMUM, --maximum MAXIMUM     set the 2d coordinate to range between 0 and <MAXIMUM>
  -f FILENAME, --filename FILENAME  store the data in the inputs directory and named <FILENAME>
```

- **kmeans**: default CLUSTERS = 3, FILENAME = 'data.txt'
```console
usage: ./kmeans [-h] [-c CLUSTERS] [-f FILENAME] [--] cmd

optional arguments:
  -h --help                     show this help message and exit
  -c --clusters <CLUSTERS>      classify the data into <CLUSTERS> groups
  -f --filename <FILENAME>      <FILENAME> in the inputs directory
  --                            sperate the arguments for kmeans and for the command
  cmd                           only "serial", "omp", and "mpi" are available
```

- **draw.py**: default CLUSTERS = 3, FILENAME = 'data.txt'
```console
usage: draw.py [-h] [-c CLUSTERS] [-f FILENAME]

Draw the scatterplots before and after K-Means Clustering.

optional arguments:
  -h, --help                        show this help message and exit
  -c CLUSTERS, --clusters CLUSTERS  classify the data into <CLUSTERS> groups
  -f FILENAME, --filename FILENAME  <FILENAME> from the inputs directory
```

## Execution
Install Python required packages and generate the executable file **kmeans**.
```bash
$ pip3 install -r requirements.txt; make 
```

Randomly generate 2d coordinates and store them into the inputs directory.
```bash
$ python3 generate.py [-n NUMS] [-m MAXIMUM] [-f FILENAME]
```

Do K-Means Clustering.
```bash
$ ./kmeans [-c CLUSTERS] [-f FILENAME] [--] cmd
```

Draw the scatterplots before and after K-Means Clustering if you would like to see the result.
```bash
$ python3 draw.py [-c CLUSTERS] [-f FILENAME]
```

## References
- [Exploring K-Means in Python, C++ and CUDA](www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda/)
- [Implementing k-means clustering from scratch in C++](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)