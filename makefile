all:
	g++ -std=c++11 -pthread -O3 -flto main.cpp ModelHistogram.cpp -o ModelHistogram
