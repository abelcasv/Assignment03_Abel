CC      ?= gcc
CFLAGS  = -Wall -Wextra -O3 -std=c11 -fopenmp -march=native

ifeq ($(OS),Windows_NT)
    LDFLAGS = -lpsapi
    TARGET  = parallel_benchmark.exe
else
    LDFLAGS = -lm
    TARGET  = parallel_benchmark
endif

.PHONY: all clean validate pilot vec-report help

all: $(TARGET)

$(TARGET): parallel_benchmark.c
	$(CC) $(CFLAGS) -DBUILD_FLAGS='"$(CFLAGS)"' parallel_benchmark.c -o $(TARGET) $(LDFLAGS)

vec-report: parallel_benchmark.c
	$(CC) $(CFLAGS) -fopt-info-vec-optimized=vec_report.txt -c parallel_benchmark.c -o parallel_benchmark.o
	@echo [INFO] Vectorisation report written to vec_report.txt

validate: $(TARGET)
	./$(TARGET) --validate-only ..

pilot: $(TARGET)
	./$(TARGET) --pilot ..

clean:
	rm -f $(TARGET) parallel_benchmark.exe parallel_benchmark parallel_benchmark.o vec_report.txt

help:
	@echo Available make targets:
	@echo   make
	@echo   make vec-report
	@echo   make validate
	@echo   make pilot
	@echo   make clean
