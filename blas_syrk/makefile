CC = gcc
CFLAGS = -O3 -march=native -std=c11 -I./src -fopenmp
LDFLAGS = -lopenblas -lm -fopenmp

SYRK_SRC = src/syrk.c
BENCH_SRC = tests/benchmark_syrk.c
TARGET = benchmark_syrk

all: $(TARGET)

$(TARGET): $(SYRK_SRC) $(BENCH_SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SYRK_SRC) $(BENCH_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run