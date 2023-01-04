#include <omp.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <iostream>
#include <algorithm>
#include "kmeans.h"

#define MASTER      0
#define DataFrame   std::vector<Point>

void usage(const char *progname) {
    fprintf(stderr, "usage: %s [-h] [-c CLUSTERS] [-f FILENAME] [-t THREADS] [--] cmd\n", progname);
    fprintf(stderr, "\n");
    fprintf(stderr, "optional arguments:\n");
    fprintf(stderr, "  -h --help                  show this help message and exit\n");
    fprintf(stderr, "  -n --no-output             an argument to control writing the results into a file or not\n");
    fprintf(stderr, "  -c --clusters <CLUSTERS>   classify the data into <CLUSTERS> groups\n");
    fprintf(stderr, "  -f --filename <FILENAME>   <FILENAME> in the inputs directory\n");
    fprintf(stderr, "  -t --threads  <THREADS>    specify the number of omp threads, default 4\n");
    fprintf(stderr, "  --                         sperate the arguments for kmeans and for the command\n");
    fprintf(stderr, "  cmd                        only \"serial\", \"omp\", \"mpi\", and \"hybrid\" are available\n");
}

int main(int argc, char *argv[]) {
    static struct option long_options[] = {
        {"help"      , no_argument      , NULL, 'h'},
        {"no-output" , no_argument      , NULL, 'n'},
        {"clusters"  , optional_argument, NULL, 'c'},
        {"filename"  , optional_argument, NULL, 'f'},
        {"threads"   , optional_argument, NULL, 't'},
        {NULL        , 0                , NULL,  0 }
    };

    int opt;
    int threads = 4;
    bool output = true;
    std::string command;
    unsigned int clusters = 3;
    std::string filename = "data.txt";

    while ((opt = getopt_long(argc, argv, "f:c:t:nh", long_options, NULL)) != EOF) {
        switch (opt) {
            case 'c': clusters = strtol(optarg, NULL, 10); break;
            case 'f': filename = std::string(optarg);      break;
            case 't': threads  = std::stoi(optarg);        break;
            case 'n': output = false;                      break;
            case 'h': usage(argv[0]); exit(1);
            default : usage(argv[0]); exit(1);
        }
    }

    if (optind < argc) {
        command = std::string(argv[optind]);
    }

    int world_size, world_rank = 0;
    DataFrame (*kmeans)(const DataFrame&, unsigned int, unsigned int*);

    if (command == "serial") {
        kmeans = &kmeansSerial;
    } else if (command == "omp") {
        kmeans = &kmeansOMP;
    } else if (command == "mpi") {
        kmeans = &kmeansMPI;
    } else if (command == "hybrid") {
        kmeans = &kmeansHybrid;
    } else if (command == "") {
        fprintf(stderr, "no command given.\n");
        exit(1);
    } else {
        fprintf(stderr, "command \"%s\" is not available.\n", command.c_str());
        exit(1);
    }

    if (command == "mpi") {
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    }

    if (command == "hybrid") {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    }

    if (command == "omp" || command == "hybrid") {
        threads = std::min(threads, omp_get_max_threads());
        omp_set_num_threads(threads);
    }

    DataFrame points;
    if (readfile(filename, points) == -1) {
        if (command == "mpi" || command == "hybrid") {
            MPI_Finalize();
        }
        exit(1);
    }

    double elapsed_time;
    struct timespec starttime, endtime;
    unsigned int *point_clusters = (unsigned int*) calloc(points.size(), sizeof(unsigned int));

    if (world_rank == MASTER) {
        clock_gettime(CLOCK_MONOTONIC, &starttime);
    }

    kmeans(points, clusters, point_clusters);

    if (world_rank == MASTER) {
        clock_gettime(CLOCK_MONOTONIC, &endtime);
        elapsed_time = calculate_time(starttime, endtime);
    }

    if (world_rank == MASTER) {
        printf("Total elapsed time with \"%s\" command: %.6f secs\n", command.c_str(), elapsed_time);
        if (output) {
            if (writefile(filename + ".out", points, point_clusters) == -1) {
                if (command == "mpi" || command == "hybrid") {
                    MPI_Finalize();
                }
                exit(1);
            }
        }
    }

    if (command == "mpi" || command == "hybrid") {
        MPI_Finalize();
    }

    return 0;
}
