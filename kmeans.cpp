#include <omp.h>
#include <mpi.h>
#include <random>
#include <errno.h>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include "kmeans.h"

#define MASTER      0
#define ITERATIONS  100
#define MODE        0775
#define DataFrame   std::vector<Point>

int readfile(std::string filename, DataFrame &points) {
    std::filesystem::path dir("inputs");
    std::filesystem::path file(filename);
    std::filesystem::path pathname = dir / file;

    std::ifstream fp;
    fp.open(pathname, std::ofstream::in);

    if (fp.is_open()) {
        double x, y;

        while (!fp.eof()) {
            fp >> x >> y;
            points.push_back(Point(x, y));
        }

        fp.close();
    } else {
        perror("readfile error");
        return -1;
    }

    points.pop_back();

    return 0;
}

int writefile(std::string filename, DataFrame &points, unsigned int *point_clusters) {
    std::filesystem::path dir("outputs");
    std::filesystem::path file(filename);
    std::filesystem::path pathname = dir / file;

    if (!std::filesystem::exists(dir)) {
        mkdir(dir.c_str(), MODE);
    }

    std::ofstream fp;
    fp.open(pathname, std::ofstream::out);

    if (fp.is_open()) {
        for (long unsigned int i = 0; i < points.size(); i++) {
            fp << std::setfill(' ') << std::setw(12) << std::setprecision(10) << points[i].x << std::setw(12) << std::setprecision(10) << points[i].y << std::setw(4) << point_clusters[i] << std::endl;
        }

        fp.close();
    } else {
        perror("writefile error");
        return -1;
    }

    return 0;
}

double calculate_time(const struct timespec &starttime, const struct timespec &endtime) {
        double elapsed;
        elapsed = endtime.tv_sec - starttime.tv_sec;
        elapsed += (endtime.tv_nsec - starttime.tv_nsec) / 1000000000.0;
        return elapsed;
}

void point_sum(Point *in_point, Point *in_out_point, int *length, MPI_Datatype *dtype) {
    for (int i = 0; i < *length; i++) { 
        in_out_point[i].x += in_point[i].x;
        in_out_point[i].y += in_point[i].y;
    }
}

long double square(double value) {
    return value * value;
}

long double squared_euclidean_distance(const Point &first, const Point &second) {
    return square(first.x - second.x) + square(first.y - second.y);
}

DataFrame kmeansSerial(const DataFrame &data, unsigned int k, unsigned int *point_clusters) {
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    // Pick centroids as random points from the dataset
    DataFrame means(k);
    for (unsigned int i = 0; i < means.size(); i++) {
        means[i] = data[indices(random_number_generator)];
    }

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        // Find the point belongs to which cluster
        for (long long point = 0; point < (long long) data.size(); point++) {
            long double best_distance = std::numeric_limits<double>::max();
            unsigned int best_cluster = 0;
            for (unsigned int cluster = 0; cluster < k; cluster++) {
                const long double distance = squared_euclidean_distance(data[point], means[cluster]);
                if (distance < best_distance) {
                    best_cluster = cluster;
                    best_distance = distance;
                }
            }

            point_clusters[point] = best_cluster;
        }

        // Sum up and count points for each cluster
        DataFrame new_means(k);
        std::vector<long long> counts(k, 0);
        for (long long point = 0; point < (long long) data.size(); point++) {
            const unsigned int cluster = point_clusters[point];
            new_means[cluster].x += data[point].x;
            new_means[cluster].y += data[point].y;
            counts[cluster] += 1;
        }

        // Divide sums by counts to get new centroids
        for (unsigned int cluster = 0; cluster < k; cluster++) {
            const long long count = std::max<long long>(1, counts[cluster]);
            means[cluster].x = new_means[cluster].x / count;
            means[cluster].y = new_means[cluster].y / count;
        }
    }

    return means;
}

DataFrame kmeansOMP(const DataFrame &data, unsigned int k, unsigned int *point_clusters) {
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    // Pick centroids as random points from the dataset
    DataFrame means(k);
    for (unsigned int i = 0; i < means.size(); i++) {
        means[i] = data[indices(random_number_generator)];
    }

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        DataFrame new_means(k);
        std::vector<long long> counts(k, 0);

        std::vector<DataFrame> local_new_means;
        std::vector<std::vector<long long>> local_counts;

        #pragma omp parallel
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
                unsigned int best_cluster = 0;
                for (unsigned int cluster = 0; cluster < k; cluster++) {
                    const long double distance = squared_euclidean_distance(data[point], means[cluster]);
                    if (distance < best_distance) {
                        best_cluster = cluster;
                        best_distance = distance;
                    }
                }

                point_clusters[point] = best_cluster;
            }

            // Sum up and count points for each cluster
            #pragma omp for
            for (long long point = 0; point < (long long) data.size(); point++) {
                const unsigned int cluster = point_clusters[point];
                local_new_means[thread_id][cluster].x += data[point].x;
                local_new_means[thread_id][cluster].y += data[point].y;
                local_counts[thread_id][cluster] += 1;
            }

            #pragma omp single
            {
                for (int i = 0; i < thread_nums; i++) {
                    DataFrame local_new_mean = local_new_means[i];
                    std::vector<long long> local_count = local_counts[i];

                    for (unsigned int cluster = 0; cluster < k; cluster++) {
                        new_means[cluster].x += local_new_mean[cluster].x;
                        new_means[cluster].y += local_new_mean[cluster].y;
                        counts[cluster] += local_count[cluster];
                    }
                }
            }

            // Divide sums by counts to get new centroids
            #pragma omp for
            for (unsigned int cluster = 0; cluster < k; cluster++) {
                const long long count = std::max<long long>(1, counts[cluster]);
                means[cluster].x = new_means[cluster].x / count;
                means[cluster].y = new_means[cluster].y / count;
            }
        }
    }

    return means;
}

DataFrame kmeansMPI(const DataFrame &data, unsigned int k, unsigned int *point_clusters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Define own MPI struct type
    MPI_Aint base_address;
    MPI_Aint displacements[2];
    MPI_Datatype point_type;
    MPI_Datatype types[2] = { MPI_DOUBLE, MPI_DOUBLE };

    struct Point point;
    int lengths[2] = { 1, 1 };
 
    MPI_Get_address(&point, &base_address);
    MPI_Get_address(&point.x, &displacements[0]);
    MPI_Get_address(&point.y, &displacements[1]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    
    MPI_Type_create_struct(2, lengths, displacements, types, &point_type);
    MPI_Type_commit(&point_type);

    // Define own MPI operation type
    MPI_Op POINT_SUM;
    MPI_Op_create((MPI_User_function *) point_sum, 1, &POINT_SUM);

    // Pick centroids as random points from the dataset
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    DataFrame return_means(k);
    Point *means = (Point*) calloc(k, sizeof(Point));

    if (world_rank == MASTER) {
        for (unsigned int i = 0; i < k; i++) {
            means[i] = data[indices(random_number_generator)];
        }
    }

    // Assign work
    int points;
    int offset;
    int extra_points = data.size() % world_size;
    int number_of_points = data.size() / world_size;

    int *points_array = NULL;
    int *offset_array = NULL;

    if (world_rank == MASTER) {
        points_array = (int*) calloc(world_size, sizeof(int));
        offset_array = (int*) calloc(world_size, sizeof(int));

        offset = 0;
        for (int worker = 0; worker < world_size; worker++) {
            points = (worker < extra_points)? number_of_points + 1: number_of_points;
            points_array[worker] = points;
            offset_array[worker] = offset;
            offset += points;
        }
    }

    // MPI scatter the value of points and offset to every process
    MPI_Scatter(&points_array[world_rank], 1, MPI_INT, &points, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(&offset_array[world_rank], 1, MPI_INT, &offset, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Local variable for each process
    unsigned int *local_point_clusters = (unsigned int*) calloc(points, sizeof(unsigned int)); 

    // MPI broadcast the value of means and change
    MPI_Bcast(means, k, point_type, MASTER, MPI_COMM_WORLD);

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        // Find the point belongs to which cluster
        for (int point = 0; point < points; point++) {
            long double best_distance = std::numeric_limits<double>::max();
            unsigned int best_cluster = 0;
            for (unsigned int cluster = 0; cluster < k; cluster++) {
                const long double distance = squared_euclidean_distance(data[point + offset], means[cluster]);
                if (distance < best_distance) {
                    best_cluster = cluster;
                    best_distance = distance;
                }
            }

            local_point_clusters[point] = best_cluster;
        }

        // Sum up and count points for each cluster
        Point *new_means = NULL;
        long long *counts = NULL;
        Point *local_new_means = (Point*) calloc(k, sizeof(Point));
        long long *local_counts = (long long*) calloc(k, sizeof(long long));

        for (int point = 0; point < points; point++) {
            const unsigned int cluster = local_point_clusters[point];
            local_new_means[cluster].x += data[point].x;
            local_new_means[cluster].y += data[point].y;
            local_counts[cluster] += 1;
        }

        if (world_rank == MASTER) {
            new_means = (Point*) calloc(k, sizeof(Point));
            counts = (long long*) calloc(k, sizeof(long long));
        }

        MPI_Reduce(local_counts, counts, k, MPI_LONG_LONG_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(local_new_means, new_means, k, point_type, POINT_SUM, MASTER, MPI_COMM_WORLD);

        // Divide sums by counts to get new centroids
        if (world_rank == MASTER) {
            for (unsigned int cluster = 0; cluster < k; cluster++) {
                const long long count = std::max<long long>(1, counts[cluster]);
                means[cluster].x = new_means[cluster].x / count;
                means[cluster].y = new_means[cluster].y / count;
            }
        }

        MPI_Bcast(means, k, point_type, MASTER, MPI_COMM_WORLD);

        // Free the space
        free(local_counts);
        free(local_new_means);

        if (world_rank == MASTER) {
            free(counts);
            free(new_means);
        }
    }

    MPI_Gather(local_point_clusters, points, MPI_UNSIGNED, point_clusters, points, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

    // Copy the value of means to return_means
    if (world_rank == MASTER) {
        for (long unsigned int i = 0; i < k; i++) {
            return_means[i] = means[i];
        }

        free(points_array);
        free(offset_array);
    }

    MPI_Op_free(&POINT_SUM);
    MPI_Type_free(&point_type);

    free(means);
    free(local_point_clusters);

    return return_means;
}

// hybrid = MPI + OMP
DataFrame kmeansHybridFull(const DataFrame &data, unsigned int k, unsigned int *point_clusters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Define own MPI struct type
    MPI_Aint base_address;
    MPI_Aint displacements[2];
    MPI_Datatype point_type;
    MPI_Datatype types[2] = { MPI_DOUBLE, MPI_DOUBLE };

    struct Point point;
    int lengths[2] = { 1, 1 };
 
    MPI_Get_address(&point, &base_address);
    MPI_Get_address(&point.x, &displacements[0]);
    MPI_Get_address(&point.y, &displacements[1]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    
    MPI_Type_create_struct(2, lengths, displacements, types, &point_type);
    MPI_Type_commit(&point_type);

    // Define own MPI operation type
    MPI_Op POINT_SUM;
    MPI_Op_create((MPI_User_function *) point_sum, 1, &POINT_SUM);

    // Pick centroids as random points from the dataset
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    DataFrame return_means(k);
    Point *means = (Point*) calloc(k, sizeof(Point));

    if (world_rank == MASTER) {
        for (unsigned int i = 0; i < k; i++) {
            means[i] = data[indices(random_number_generator)];
        }
    }

    // Assign work
    int points;
    int offset;
    int extra_points = data.size() % world_size;
    int number_of_points = data.size() / world_size;

    int *points_array = NULL;
    int *offset_array = NULL;

    if (world_rank == MASTER) {
        points_array = (int*) calloc(world_size, sizeof(int));
        offset_array = (int*) calloc(world_size, sizeof(int));

        offset = 0;
        for (int worker = 0; worker < world_size; worker++) {
            points = (worker < extra_points)? number_of_points + 1: number_of_points;
            points_array[worker] = points;
            offset_array[worker] = offset;
            offset += points;
        }
    }

    // MPI scatter the value of points and offset to every process
    MPI_Scatter(&points_array[world_rank], 1, MPI_INT, &points, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(&offset_array[world_rank], 1, MPI_INT, &offset, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Local variable for each process
    unsigned int *node_point_clusters = (unsigned int*) calloc(points, sizeof(unsigned int)); 

    // MPI broadcast the value of means and change
    MPI_Bcast(means, k, point_type, MASTER, MPI_COMM_WORLD);

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        std::vector<DataFrame> thrd_new_means;
        std::vector<std::vector<long long>> thrd_counts;
        Point *node_new_means = (Point*) calloc(k, sizeof(Point));
        long long *node_counts = (long long*) calloc(k, sizeof(long long));

        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            const int thread_nums = omp_get_num_threads();
            
            #pragma omp single
            {
                for (int i = 0; i < thread_nums; i++) {
                    DataFrame thrd_new_mean(k);
                    std::vector<long long> thrd_count(k, 0);
                    thrd_new_means.push_back(thrd_new_mean);
                    thrd_counts.push_back(thrd_count);
                }
            }

            // Find the point belongs to which cluster
            #pragma omp for
            for (int point = 0; point < points; point++) {
                long double best_distance = std::numeric_limits<double>::max();
                unsigned int best_cluster = 0;
                for (unsigned int cluster = 0; cluster < k; cluster++) {
                    const long double distance = squared_euclidean_distance(data[point + offset], means[cluster]);
                    if (distance < best_distance) {
                        best_cluster = cluster;
                        best_distance = distance;
                    }
                }

                node_point_clusters[point] = best_cluster;
            }

            // Sum up and count points for each cluster
            #pragma omp for
            for (int point = 0; point < points; point++) {
                const unsigned int cluster = node_point_clusters[point];
                thrd_new_means[thread_id][cluster].x += data[point].x;
                thrd_new_means[thread_id][cluster].y += data[point].y;
                thrd_counts[thread_id][cluster] += 1;
            }

            #pragma omp single
            {
                for (int i = 0; i < thread_nums; i++) {
                    DataFrame thrd_new_mean = thrd_new_means[i];
                    std::vector<long long> thrd_count = thrd_counts[i];

                    for (unsigned int cluster = 0; cluster < k; cluster++) {
                        node_new_means[cluster].x += thrd_new_mean[cluster].x;
                        node_new_means[cluster].y += thrd_new_mean[cluster].y;
                        node_counts[cluster] += thrd_count[cluster];
                    }
                }
            }
        }
        
        Point *new_means = NULL;
        long long *counts = NULL;

        if (world_rank == MASTER) {
            new_means = (Point*) calloc(k, sizeof(Point));
            counts = (long long*) calloc(k, sizeof(long long));
        }

        MPI_Reduce(node_counts, counts, k, MPI_LONG_LONG_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(node_new_means, new_means, k, point_type, POINT_SUM, MASTER, MPI_COMM_WORLD);

        // Divide sums by counts to get new centroids
        if (world_rank == MASTER) {
            for (unsigned int cluster = 0; cluster < k; cluster++) {
                const long long count = std::max<long long>(1, counts[cluster]);
                means[cluster].x = new_means[cluster].x / count;
                means[cluster].y = new_means[cluster].y / count;
            }
        }

        MPI_Bcast(means, k, point_type, MASTER, MPI_COMM_WORLD);

        // Free the space
        free(node_counts);
        free(node_new_means);

        if (world_rank == MASTER) {
            free(counts);
            free(new_means);
        }
    }

    MPI_Gather(node_point_clusters, points, MPI_UNSIGNED, point_clusters, points, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

    // Copy the value of means to return_means
    if (world_rank == MASTER) {
        for (long unsigned int i = 0; i < k; i++) {
            return_means[i] = means[i];
        }

        free(points_array);
        free(offset_array);
    }

    MPI_Op_free(&POINT_SUM);
    MPI_Type_free(&point_type);

    free(means);
    free(node_point_clusters);

    return return_means;
}

// hybrid = MPI + OMP
DataFrame kmeansHybrid(const DataFrame &data, unsigned int k, unsigned int *point_clusters) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Define own MPI struct type
    MPI_Aint base_address;
    MPI_Aint displacements[2];
    MPI_Datatype point_type;
    MPI_Datatype types[2] = { MPI_DOUBLE, MPI_DOUBLE };

    struct Point point;
    int lengths[2] = { 1, 1 };
 
    MPI_Get_address(&point, &base_address);
    MPI_Get_address(&point.x, &displacements[0]);
    MPI_Get_address(&point.y, &displacements[1]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    
    MPI_Type_create_struct(2, lengths, displacements, types, &point_type);
    MPI_Type_commit(&point_type);

    // Define own MPI operation type
    MPI_Op POINT_SUM;
    MPI_Op_create((MPI_User_function *) point_sum, 1, &POINT_SUM);

    // Pick centroids as random points from the dataset
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<long long> indices(0, data.size() - 1);

    DataFrame return_means(k);
    Point *means = (Point*) calloc(k, sizeof(Point));

    if (world_rank == MASTER) {
        for (unsigned int i = 0; i < k; i++) {
            means[i] = data[indices(random_number_generator)];
        }
    }

    // Assign work
    int points;
    int offset;
    int extra_points = data.size() % world_size;
    int number_of_points = data.size() / world_size;

    int *points_array = NULL;
    int *offset_array = NULL;

    if (world_rank == MASTER) {
        points_array = (int*) calloc(world_size, sizeof(int));
        offset_array = (int*) calloc(world_size, sizeof(int));

        offset = 0;
        for (int worker = 0; worker < world_size; worker++) {
            points = (worker < extra_points)? number_of_points + 1: number_of_points;
            points_array[worker] = points;
            offset_array[worker] = offset;
            offset += points;
        }
    }

    // MPI scatter the value of points and offset to every process
    MPI_Scatter(&points_array[world_rank], 1, MPI_INT, &points, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Scatter(&offset_array[world_rank], 1, MPI_INT, &offset, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Local variable for each process
    unsigned int *node_point_clusters = (unsigned int*) calloc(points, sizeof(unsigned int)); 

    // MPI broadcast the value of means and change
    MPI_Bcast(means, k, point_type, MASTER, MPI_COMM_WORLD);

    for (int iteration = 0; iteration < ITERATIONS; iteration++) {

        #pragma omp parallel
        {
            // Find the point belongs to which cluster
            #pragma omp for
            for (int point = 0; point < points; point++) {
                long double best_distance = std::numeric_limits<double>::max();
                unsigned int best_cluster = 0;
                for (unsigned int cluster = 0; cluster < k; cluster++) {
                    const long double distance = squared_euclidean_distance(data[point + offset], means[cluster]);
                    if (distance < best_distance) {
                        best_cluster = cluster;
                        best_distance = distance;
                    }
                }

                node_point_clusters[point] = best_cluster;
            }
        }
        
        // Sum up and count points for each cluster
        Point *new_means = NULL;
        long long *counts = NULL;
        Point *node_new_means = (Point*) calloc(k, sizeof(Point));
        long long *node_counts = (long long*) calloc(k, sizeof(long long));

        for (int point = 0; point < points; point++) {
            const unsigned int cluster = node_point_clusters[point];
            node_new_means[cluster].x += data[point].x;
            node_new_means[cluster].y += data[point].y;
            node_counts[cluster] += 1;
        }

        if (world_rank == MASTER) {
            new_means = (Point*) calloc(k, sizeof(Point));
            counts = (long long*) calloc(k, sizeof(long long));
        }

        MPI_Reduce(node_counts, counts, k, MPI_LONG_LONG_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(node_new_means, new_means, k, point_type, POINT_SUM, MASTER, MPI_COMM_WORLD);

        // Divide sums by counts to get new centroids
        if (world_rank == MASTER) {
            for (unsigned int cluster = 0; cluster < k; cluster++) {
                const long long count = std::max<long long>(1, counts[cluster]);
                means[cluster].x = new_means[cluster].x / count;
                means[cluster].y = new_means[cluster].y / count;
            }
        }

        MPI_Bcast(means, k, point_type, MASTER, MPI_COMM_WORLD);

        // Free the space
        free(node_counts);
        free(node_new_means);

        if (world_rank == MASTER) {
            free(counts);
            free(new_means);
        }
    }

    MPI_Gather(node_point_clusters, points, MPI_UNSIGNED, point_clusters, points, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

    // Copy the value of means to return_means
    if (world_rank == MASTER) {
        for (long unsigned int i = 0; i < k; i++) {
            return_means[i] = means[i];
        }

        free(points_array);
        free(offset_array);
    }

    MPI_Op_free(&POINT_SUM);
    MPI_Type_free(&point_type);

    free(means);
    free(node_point_clusters);

    return return_means;
}
