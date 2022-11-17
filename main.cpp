#include <time.h>
#include <math.h>
#include <getopt.h>
#include <iostream>
#include <algorithm>
#include "kmeans.h"

void usage(const char *progname) {
    fprintf(stderr, "usage: %s [-h] [-c CLUSTERS] [-f FILENAME]\n", progname);
    fprintf(stderr, "\n");
    fprintf(stderr, "optional arguments:\n");
    fprintf(stderr, "  -h --help                  show this help message and exit\n");
    fprintf(stderr, "  -c --clusters <CLUSTERS>   classify the data into <CLUSTERS> groups\n");
    fprintf(stderr, "  -f --filename <FILENAME>   <FILENAME> in the inputs directory\n");
}

int main(int argc, char *argv[]) {
    static struct option long_options[] = {
        {"help"      , no_argument      , NULL, 'h'},
        {"clusters"  , optional_argument, NULL, 'c'},
        {"filename"  , optional_argument, NULL, 'f'},
        {NULL        , 0                , NULL,  0 }
    };

    int opt;
    size_t clusters = 3;
    std::string filename = "data.txt";

    while ((opt = getopt_long(argc, argv, "f:c:h", long_options, NULL)) != EOF) {
        switch (opt) {
            case 'c': clusters = strtol(optarg, NULL, 10); break;
            case 'f': filename = std::string(optarg);      break;
            case 'h': usage(argv[0]); exit(1);
            default : usage(argv[0]); exit(1);
        }
    }

    DataFrame points;
    readfile(filename, points);

    double elapsed;
    struct timespec starttime, endtime;
    std::vector<double> elapsed_times;

    for (int thread_nums = 1; thread_nums <= 8; thread_nums *= 2) {
        if (thread_nums == 1) {
            clock_gettime(CLOCK_MONOTONIC, &starttime);
            kmeansSerial(points, clusters);
            clock_gettime(CLOCK_MONOTONIC, &endtime);
        } else {
            clock_gettime(CLOCK_MONOTONIC, &starttime);
            kmeansThread(points, clusters, thread_nums);
            clock_gettime(CLOCK_MONOTONIC, &endtime);
        }

        elapsed = endtime.tv_sec - starttime.tv_sec;
        elapsed += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;
        elapsed_times.push_back(elapsed);
    }

    for (long unsigned int i = 0; i < elapsed_times.size(); i++) {
        int thread_nums = (int) pow(2, i);
        double ratio = std::min(elapsed_times[0] / elapsed_times[i], (double) thread_nums);
        printf("Total elapsed time with %d thread(s): %7.3fs (%2.1fX)\n", thread_nums, elapsed_times[i], ratio);
    }

    writefile(filename + ".out", points);

    return 0;
}