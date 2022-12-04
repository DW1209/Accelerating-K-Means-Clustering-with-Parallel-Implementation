CXX = mpicxx
CFLAGS = -std=c++17 -Wall -O3 -fopenmp

PROGS = kmeans
OBJS = kmeans.o main.o

all: $(PROGS)

kmeans: $(OBJS)
	$(CXX) $(CFLAGS) $^ -o $@

%.o:%.cpp
	$(CXX) $(CFLAGS) -c $^ -o $@

clean:
	rm $(PROGS) $(OBJS)