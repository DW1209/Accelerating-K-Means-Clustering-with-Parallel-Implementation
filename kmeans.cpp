#include <omp.h>
#include <random>
#include <fstream>
#include "kmeans.h"

#define DataFrame std::vector<Point>

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

long double square(double value) {
    return value * value;
}

long double squared_euclidean_distance(Point first, Point second) {
    return square(first.x - second.x) + square(first.y - second.y);
}

DataFrame kmeansSerial(DataFrame &data, size_t k, long long num_of_iterations) {
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    // Pick centroids as random points from the dataset
    DataFrame means(k);
    for (size_t i = 0; i < means.size(); i++) {
        means[i] = data[indices(random_number_generator)];
    }

    for (long long it = 0; it < num_of_iterations; it++) {
        // Find the point belongs to which cluster
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

            data[point].cluster = best_cluster;
        }

        // Sum up and count points for each cluster
        DataFrame new_means(k);
        std::vector<long long> counts(k, 0);
        for (long long point = 0; point < (long long) data.size(); point++) {
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

DataFrame kmeansThread(DataFrame &data, size_t k, long long num_of_iterations, int expected_thread_nums) {
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    // Pick centroids as random points from the dataset
    DataFrame means(k);
    for (size_t i = 0; i < means.size(); i++) {
        means[i] = data[indices(random_number_generator)];
    }

    for (long long it = 0; it < num_of_iterations; it++) {
        DataFrame new_means(k);
        std::vector<long long> counts(k, 0);

        std::vector<DataFrame> local_new_means;
        std::vector<std::vector<long long>> local_counts;

        #pragma omp parallel num_threads(expected_thread_nums)
        {
            const int thread_id = omp_get_thread_num();
            const int thread_nums = omp_get_num_threads();
            
            #pragma omp single
            {
                for (int i = 0; i < thread_nums; i++) {
                    DataFrame local_new_mean(k);
                    std::vector<long long> local_count(k, 0);
                    local_new_means.push_back(local_new_mean);
                    local_counts.push_back(local_count);
                }
            }

            // Find the point belongs to which cluster
            #pragma omp for
            for (long long point = 0; point < (long long) data.size(); point++) {
                long double best_distance = std::numeric_limits<long double>::max();
                size_t best_cluster = 0;
                for (size_t cluster = 0; cluster < k; cluster++) {
                    const long double distance = squared_euclidean_distance(data[point], means[cluster]);
                    if (distance < best_distance) {
                        best_cluster = cluster;
                        best_distance = distance;
                    }
                }

                data[point].cluster = best_cluster;
            }

            // Sum up and count points for each cluster
            #pragma omp for
            for (long long point = 0; point < (long long) data.size(); point++) {
                const size_t cluster = data[point].cluster;
                local_new_means[thread_id][cluster].x += data[point].x;
                local_new_means[thread_id][cluster].y += data[point].y;
                local_counts[thread_id][cluster] += 1;
            }

            #pragma omp single
            {
                for (int i = 0; i < thread_nums; i++) {
                    DataFrame local_new_mean = local_new_means[i];
                    std::vector<long long> local_count = local_counts[i];

                    for (size_t cluster = 0; cluster < k; cluster++) {
                        new_means[cluster].x += local_new_mean[cluster].x;
                        new_means[cluster].y += local_new_mean[cluster].y;
                        counts[cluster] += local_count[cluster];
                    }
                }
            }

            // Divide sums by counts to get new centroids
            #pragma omp for
            for (size_t cluster = 0; cluster < k; cluster++) {
                const long long count = std::max<long long>(1, counts[cluster]);
                means[cluster].x = new_means[cluster].x / count;
                means[cluster].y = new_means[cluster].y / count;
            }
        }
    }

    return means;
}
