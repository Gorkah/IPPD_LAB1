CC = gcc
CFLAGS = -fopenmp -O3
LIBS = -lm

all: argmax

argmax: main.c argmax.c
	$(CC) $(CFLAGS) -o argmax main.c argmax.c $(LIBS)

clean:
	rm -f argmax
