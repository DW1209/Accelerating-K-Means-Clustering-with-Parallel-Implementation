#include <time.h>
#include <getopt.h>
#include <iostream>
#include "kmeans.h"

void usage(const char *progname) {
    fprintf(stderr, "Usage: %s <options>\n", progname);
    fprintf(stderr, "Program Options:\n");
    fprintf(stderr, "  -f --filename <FILENAME>   filename of input data\n");
    fprintf(stderr, "  -c --clusters <M>          Set M clusters\n");
    fprintf(stderr, "  -n --iterations <N>        Set N iterations\n");
    fprintf(stderr, "  -? --help                  This message\n");
}

int main(int argc, char *argv[]) {
    static struct option long_options[] = {
        {"filename"  , required_argument, NULL, 'f'},
        {"clusters"  , required_argument, NULL, 'c'},
        {"iterations", required_argument, NULL, 'n'},
        {"help"      , no_argument      , NULL, '?'},
        {NULL        , 0                , NULL,  0 }
    };

    int opt;
    size_t clusters = 0;
    std::string filename;
    long long num_of_iterations = 0;
    bool fmark = false, cmark = false, nmark = false;

    while ((opt = getopt_long(argc, argv, "f:c:n:?", long_options, NULL)) != EOF) {
        switch (opt) {
            case 'f': filename = std::string(optarg);                fmark = true; break;
            case 'c': clusters = strtol(optarg, NULL, 10);           cmark = true; break;
            case 'n': num_of_iterations = strtoll(optarg, NULL, 10); nmark = true; break;
            default : usage(argv[0]); exit(1);
        }
    }

    if (!fmark || !cmark || !nmark) {
        usage(argv[0]); exit(1);
    }

    DataFrame points;
    readfile(filename, points);

    struct timespec starttime, endtime;
    clock_gettime(CLOCK_MONOTONIC, &starttime);
    kmeans(points, clusters, num_of_iterations);
    clock_gettime(CLOCK_MONOTONIC, &endtime);

    double duration = endtime.tv_sec - starttime.tv_sec;
    duration += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;
    printf("Total Application Time: %8fs\n", duration);
    writefile(filename + ".out", points);

    return 0;
}