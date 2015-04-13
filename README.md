# EM

In bioinformatics a common distribution to come across is a noisy histogram of data that resembles an exponential distribution and a normal distribution in all sorts of different orientations. Often you are interested in finding the local minimum between the two distributions for error correction or optimization etc. The problem is that the data is so noisy and unpredictable that you can't make many good assumptions about it beyond there will be some sort of exponential and normal distributions. This program takes in tab delimited data of the x and y coordinates of the histogram and uses an expectation maximization algorithm to model the data as well as possible and outputs the characterizing values for the distributions, how heavily weighted the distributions are use in modeling the data, and the difference between the log likelihood that the model is correct and the Kullback–Leibler divergence.



Execute:

./ModelHistogram ecoli_kmer_qual.txt

or

./ModelHistogram normal_qual.txt
