# K-Means Clustering Based on OMP, MPI, and Hybrid Methods

## Description

### generate.py 

|         |  NUMS  | MAXIMUM | FILENAME |
|:-------:|:------:|:-------:|:--------:|
| default | 10000  | 1000000 | data.txt |

```console
usage: generate.py [-h] [-n NUMS] [-m MAXIMUM] [-f FILENAME]

Randomly generate 2d coordinates and store in the inputs directory.

optional arguments:
  -h, --help                        show this help message and exit
  -n NUMS, --nums NUMS              generate <NUMS> points
  -m MAXIMUM, --maximum MAXIMUM     set the 2d coordinate to range between 0 and <MAXIMUM>
  -f FILENAME, --filename FILENAME  store the data in the inputs directory and named <FILENAME>
```

### kmeans

|         | CLUSTERS | FILENAME | THREADS | OUTPUT |
|:-------:|:--------:|:--------:|:-------:|:------:|
| default |    3     | data.txt |    4    |  true  |

```console
usage: ./kmeans [-h] [-c CLUSTERS] [-f FILENAME] [-t THREADS] [-n] [--] cmd

optional arguments:
  -h --help                     show this help message and exit
  -c --clusters <CLUSTERS>      classify the data into <CLUSTERS> groups
  -f --filename <FILENAME>      <FILENAME> in the inputs directory
  -t --threads  <THREADS>       specify the number of omp threads
  -n --no-output                disable writing the final result to the outputs directory
  --                            sperate the arguments for kmeans and for the command
  cmd                           only "serial", "omp", "mpi", and "hybrid" are available
```

### draw.py

|         | CLUSTERS | FILENAME |
|:-------:|:--------:|:--------:|
| default |    3     | data.txt |

```console
usage: draw.py [-h] [-c CLUSTERS] [-f FILENAME]

Draw the scatterplots before and after K-Means Clustering.

optional arguments:
  -h, --help                        show this help message and exit
  -c CLUSTERS, --clusters CLUSTERS  classify the data into <CLUSTERS> groups
  -f FILENAME, --filename FILENAME  <FILENAME> from the inputs directory
```

## Execution

### Basic

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
# commandline for serial and omp method
$ ./kmeans [-c CLUSTERS] [-f FILENAME] [-t THREADS] [-n] [--] cmd 

# commandline for mpi method
$ mpirun -np <PROCESSES> --hostfile <HOSTFILE> --bind-to <TYPE> \
  ./kmeans [-c CLUSTERS] [-f FILENAME] [-t THREADS] [-n] mpi

# commandline for hybrid method
$ mpirun -np <PROCESSES> -x OMP_NUM_THREADS=<THREADS> --hostfile <HOSTFILE> --bind-to <TYPE> \
  ./kmeans [-c CLUSTERS] [-f FILENAME] [-t THREADS] [-n] hybrid
```

Draw the scatterplots before and after K-Means Clustering if you would like to see the result.
```bash
$ python3 draw.py [-c CLUSTERS] [-f FILENAME]
```

### Advanced
#### An command line example for OMP method
- `-c` cluster = 5
- `-f` input filename = data.txt (default)
- `-t` OMP thread = 4 (default)
- `--no-output` = true

```bash
$ ./kmeans -c 5 --no-output omp
```

#### An command line example for MPI method 
- `-np` total number of MPI processes = 4
- `--hostfile` provide a hostfile to use
- `--bind-to core` bind processes to cores

```bash
$ mpirun -np 4 --hostfile <HOSTFILE> --bind-to core ./kmeans --no-output mpi
```

#### An command line example for Hybrid method 
- `OMP_PROC_BIND` support binding of threads
- `OMP_NUM_THREADS` specify the number of omp threads = 4
- `-np` total number of MPI processes = 4
- `-pernode` one process per node
- `--hostfile` provide a hostfile to use (should have at least 4 hosts)
- `--bind-to none` bind processes to none

```bash
$ export OMP_PROC_BIND=true; export OMP_NUM_THREADS=4;
$ mpirun -np 4 -pernode --hostfile <HOSTFILE> --bind-to none ./kmeans --no-output hybrid
```

## References
- [Exploring K-Means in Python, C++ and CUDA](http://www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda/)
- [Implementing k-means clustering from scratch in C++](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
