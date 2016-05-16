objects = main.o network.o calculations.o mnist_read.o utils.o
srcs = main.cpp network.cpp calculations.cpp mnist_read.cpp

CFLAGS = -O2 -Wall -g -std=c++11
#CFLAGS= -Wall -ggdb -std=c++11

CC = g++

all: nndeeplearn

network.o: network.cpp network.h calculations.h utils.h
	$(CC) -c $(CFLAGS) $<

main.o: main.cpp network.h mnist_read.h
	$(CC) -c $(CFLAGS) $<

%.o: %.cpp %.h
	$(CC) -c $(CFLAGS) $<

%.h: 
	;

nndeeplearn: $(objects)
	gcc -o nndeeplearn $(objects) -lstdc++ -lm

clean:
	rm nndeeplearn $(objects)
