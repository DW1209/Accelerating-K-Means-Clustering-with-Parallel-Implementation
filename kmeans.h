#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <mpi.h>
#include <vector>
#include <string>
#include <limits>

#define DataFrame std::vector<Point>

struct Point {
    double x, y; 

    Point() {
        this->x = 0;
        this->y = 0;
    }

    Point(double x, double y) {
        this->x = x;
        this->y = y;
    }
};

void readfile(std::string filename, DataFrame &points);
void writefile(std::string filename, DataFrame &points, unsigned int *point_clusters);
double calculate_time(const struct timespec &starttime, const struct timespec &endtime);
void point_sum(Point *in_point, Point *in_out_point, int *length, MPI_Datatype *dtype);
long double square(double value);
long double squared_euclidean_distance(const Point &first, const Point &second);
DataFrame kmeansSerial(const DataFrame &data, unsigned int k, unsigned int *point_clusters);
DataFrame kmeansOMP(const DataFrame &data, unsigned int k, unsigned int *point_clusters);
DataFrame kmeansMPI(const DataFrame &data, unsigned int k, unsigned int *point_clusters);

#endif