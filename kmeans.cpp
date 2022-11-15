#include <random>
#include <fstream>
#include "kmeans.h"

#define DataFrame std::vector<Point>

double square(double value) {
    return value * value;
}

double squared_euclidean_distance(Point first, Point second) {
    return square(first.x - second.x) + square(first.y - second.y);
}

DataFrame kmeans(DataFrame &data, size_t k, long long num_of_iterations) {
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    // Pick centroids as random points from the dataset
    DataFrame means(k);
    for (size_t i = 0; i < means.size(); i++) {
        means[i] = data[indices(random_number_generator)];
    }

    // std::vector<long long> assignments(data.size());
    for (long long it = 0; it < num_of_iterations; it++) {
        // Find assignments
        for (long long point = 0; point < (long long) data.size(); point++) {
            long double best_distance = std::numeric_limits<double>::max();
            size_t best_cluster = 0;
            for (size_t cluster = 0; cluster < k; cluster++) {
                const long double distance = squared_euclidean_distance(data[point], means[cluster]);
                if (distance < best_distance) {
                    best_cluster = cluster;
                    best_distance = distance;
                }
            }

            // assignments[point] = best_cluster;
            data[point].cluster = best_cluster;
        }

        // Sum up and count points for each cluster
        DataFrame new_means(k);
        std::vector<long long> counts(k, 0);
        for (long long point = 0; point < (long long) data.size(); point++) {
            // const auto cluster = assignments[point];
            const size_t cluster = data[point].cluster;
            new_means[cluster].x += data[point].x;
            new_means[cluster].y += data[point].y;
            counts[cluster] += 1;
        }

        // Divide sums by counts to get new centroids
        for (size_t cluster = 0; cluster < k; cluster++) {
            const long long count = std::max<long long>(1, counts[cluster]);
            means[cluster].x = new_means[cluster].x / count;
            means[cluster].y = new_means[cluster].y / count;
        }
    }

    return means;
}

void readfile(std::string filename, DataFrame &points) {
    std::ifstream fp;
    fp.open(filename.c_str(), std::ofstream::in);

    if (fp.is_open()) {
        double x, y;

        while (!fp.eof()) {
            fp >> x >> y;
            points.push_back(Point(x, y));
        }

        fp.close();
    }
}

void writefile(std::string filename, DataFrame &points) {
    std::ofstream fp;
    fp.open(filename.c_str(), std::ofstream::out);

    if (fp.is_open()) {
        for (DataFrame::iterator it = points.begin(); it != points.end(); it++) {
            fp << it->x << " " << it->y << " " << it->cluster << std::endl;
        }

        fp.close();
    }
}
