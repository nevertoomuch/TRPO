CC = gcc
CFLAGS = -fopenmp -O3 -Wall
LDFLAGS = -lopenblas -lm

all: benchmark_syrk test_syrk

benchmark_syrk: src/benchmark_syrk.c src/syrk.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_syrk: tests/test_syrk.c src/syrk.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test: test_syrk
	./test_syrk

perf: benchmark_syrk
	./benchmark_syrk

clean:
	rm -f benchmark_syrk test_syrk

.PHONY: all test perf clean
