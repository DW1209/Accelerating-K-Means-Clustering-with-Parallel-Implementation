#include <time.h>
#include <math.h>
#include <getopt.h>
#include <iostream>
#include <algorithm>
#include "kmeans.h"

void usage(const char *progname) {
    fprintf(stderr, "usage: %s [-h] [-n N] -f FILENAME -c M\n", progname);
    fprintf(stderr, "\n");
    fprintf(stderr, "optional arguments:\n");
    fprintf(stderr, "  -h --help                  show this help message and exit\n");
    fprintf(stderr, "  -f --filename <FILENAME>   <FILENAME> in the inputs directory\n");
    fprintf(stderr, "  -c --clusters <M>          set <M> clusters\n");
    fprintf(stderr, "  -n --iterations <N>        set <N> iterations\n");
}

int main(int argc, char *argv[]) {
    static struct option long_options[] = {
        {"help"      , no_argument      , NULL, 'h'},
        {"filename"  , required_argument, NULL, 'f'},
        {"clusters"  , required_argument, NULL, 'c'},
        {"iterations", optional_argument, NULL, 'n'},
        {NULL        , 0                , NULL,  0 }
    };

    int opt;
    size_t clusters = 0;
    std::string filename;
    long long iterations = 1000;
    bool fmark = false, cmark = false;

    while ((opt = getopt_long(argc, argv, "f:c:n:h", long_options, NULL)) != EOF) {
        switch (opt) {
            case 'f': filename = std::string(optarg);         fmark = true; break;
            case 'c': clusters = strtol(optarg, NULL, 10);    cmark = true; break;
            case 'n': iterations = strtoll(optarg, NULL, 10);               break;
            case 'h': usage(argv[0]); exit(1);
            default : usage(argv[0]); exit(1);
        }
    }

    if (!fmark || !cmark) {
        usage(argv[0]); exit(1);
    }

    DataFrame points;
    readfile(filename, points);

    double elapsed;
    struct timespec starttime, endtime;
    std::vector<double> elapsed_times;

    for (int thread_nums = 1; thread_nums <= 8; thread_nums *= 2) {
        if (thread_nums == 1) {
            clock_gettime(CLOCK_MONOTONIC, &starttime);
            kmeansSerial(points, clusters, iterations);
            clock_gettime(CLOCK_MONOTONIC, &endtime);
        } else {
            clock_gettime(CLOCK_MONOTONIC, &starttime);
            kmeansThread(points, clusters, iterations, thread_nums);
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