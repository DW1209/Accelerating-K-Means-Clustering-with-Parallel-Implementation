#include <mpi.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <iostream>
#include <algorithm>
#include "kmeans.h"

#define MASTER 0

void usage(const char *progname) {
    fprintf(stderr, "usage: %s [-h] [-c CLUSTERS] [-f FILENAME] [--] cmd\n", progname);
    fprintf(stderr, "\n");
    fprintf(stderr, "optional arguments:\n");
    fprintf(stderr, "  -h --help                  show this help message and exit\n");
    fprintf(stderr, "  -c --clusters <CLUSTERS>   classify the data into <CLUSTERS> groups\n");
    fprintf(stderr, "  -f --filename <FILENAME>   <FILENAME> in the inputs directory\n");
    fprintf(stderr, "  --                         sperate the arguments for kmeans and for the command\n");
    fprintf(stderr, "  cmd                        only \"serial\", \"omp\", and \"mpi\" are available\n");
}

int main(int argc, char *argv[]) {
    static struct option long_options[] = {
        {"help"      , no_argument      , NULL, 'h'},
        {"clusters"  , optional_argument, NULL, 'c'},
        {"filename"  , optional_argument, NULL, 'f'},
        {NULL        , 0                , NULL,  0 }
    };

    int opt;
    unsigned int clusters = 3;
    std::string command = "serial";
    std::string filename = "data.txt";

    while ((opt = getopt_long(argc, argv, "f:c:h", long_options, NULL)) != EOF) {
        switch (opt) {
            case 'c': clusters = strtol(optarg, NULL, 10); break;
            case 'f': filename = std::string(optarg);      break;
            case 'h': usage(argv[0]); exit(1);
            default : usage(argv[0]); exit(1);
        }
    }

    if (optind < argc) {
        command = std::string(argv[optind]);
    }

    int world_size, world_rank = 0;

    if (command == "mpi") {
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    }

    DataFrame points;
    if (readfile(filename, points) == -1) exit(1);
    unsigned int *point_clusters = (unsigned int*) calloc(points.size(), sizeof(unsigned int));

    double elapsed_time;
    struct timespec starttime, endtime;

    if (command == "serial") {
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        kmeansSerial(points, clusters, point_clusters);
        clock_gettime(CLOCK_MONOTONIC, &endtime);
        elapsed_time = calculate_time(starttime, endtime);
    } else if (command == "omp") {
        clock_gettime(CLOCK_MONOTONIC, &starttime);
        kmeansOMP(points, clusters, point_clusters);
        clock_gettime(CLOCK_MONOTONIC, &endtime);
        elapsed_time = calculate_time(starttime, endtime);
    } else if (command == "mpi") {
        if (world_rank == MASTER) {
            clock_gettime(CLOCK_MONOTONIC, &starttime);
        }

        kmeansMPI(points, clusters, point_clusters);

        if (world_rank == MASTER) {
            clock_gettime(CLOCK_MONOTONIC, &endtime);
            elapsed_time = calculate_time(starttime, endtime);
        }
    } else {
        fprintf(stderr, "command \"%s\" is not available.\n", command.c_str());
        exit(1);
    }

    if (world_rank == MASTER) {
        printf("Total elapsed time with \"%s\" command: %.6fs\n", command.c_str(), elapsed_time);
        if (writefile(filename + ".out", points, point_clusters) == -1) exit(1);
    }

    if (command == "mpi") {
        MPI_Finalize();
    }

    return 0;
}