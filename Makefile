CXX = g++
CFLAGS = -std=c++17 -g -Wall -O3

PROGS = kmeans

all: $(PROGS)

kmeans: kmeans.o main.cpp
	$(CXX) $(CFLAGS) $^ -o $@

%.o:%.cpp %.h
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm $(PROGS) *.o