#ifndef __KMEANS_H__
#define __KMEANS_H__

#include <vector>
#include <string>
#include <limits>

#define DataFrame std::vector<Point>

struct Point {
    double x, y; 
    size_t cluster;

    Point() {
        this->x = 0;
        this->y = 0;
        this->cluster = -1;
    }

    Point(double x, double y) {
        this->x = x;
        this->y = y;
        this->cluster = -1;
    }
};

void readfile(std::string filename, DataFrame &points);
void writefile(std::string filename, DataFrame &points);
long double square(double value);
long double squared_euclidean_distance(const Point &first, const Point &second);
DataFrame kmeansSerial(DataFrame &data, size_t k);
DataFrame kmeansThread(DataFrame &data, size_t k, int expected_thread_nums);

#endif