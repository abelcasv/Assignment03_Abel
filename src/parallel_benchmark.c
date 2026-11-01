#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <direct.h>
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define MKDIR(dir) mkdir(dir, 0755)
#endif

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "not recorded by compiler invocation"
#endif

#if defined(__GNUC__) && !defined(__INTELLISENSE__) && !defined(__clang__)
#define NO_TREE_VECTORIZE_ATTR __attribute__((optimize("O3,no-tree-vectorize")))
#else
#define NO_TREE_VECTORIZE_ATTR
#endif

#define BLOCK_SIZE 32
#define MAX_DIR_LEN 1024
#define MAX_PATH_LEN 2048
#define LINE_BUFFER_SIZE 131072

static const int DEFAULT_SIZES[] = {250, 500, 750, 1000, 1500, 2000};
static const int DEFAULT_NUM_SIZES = sizeof(DEFAULT_SIZES) / sizeof(DEFAULT_SIZES[0]);

static const int DEFAULT_THREADS[] = {1, 2, 4, 8, 10, 12, 16, 24};
static const int DEFAULT_NUM_THREADS = sizeof(DEFAULT_THREADS) / sizeof(DEFAULT_THREADS[0]);

static const int PILOT_SIZES[] = {250, 500};
static const int PILOT_NUM_SIZES = sizeof(PILOT_SIZES) / sizeof(PILOT_SIZES[0]);

static const int PILOT_THREADS[] = {1, 2, 4, 12};
static const int PILOT_NUM_THREADS = sizeof(PILOT_THREADS) / sizeof(PILOT_THREADS[0]);

static const int VEC_THREAD_COUNTS[] = {1, 2, 4, 8, 10, 12};
static const int NUM_VEC_THREAD_COUNTS = sizeof(VEC_THREAD_COUNTS) / sizeof(VEC_THREAD_COUNTS[0]);

static const int VALIDATION_SIZE = 4;
static const int NUM_REPETITIONS = 5;

static const double ABSOLUTE_TOLERANCE = 1e-4;
static const double RELATIVE_TOLERANCE = 1e-4;

static const double PREVENTIVE_MAX_MEM_MIB = 2048.0;
static const double BUDGET_SECONDS = 120.0;

static volatile double result_sink;

static double get_time_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static double get_peak_memory_mib(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS mem_info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &mem_info, sizeof(mem_info))) {
        return (double)mem_info.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        #ifdef __APPLE__
        return (double)usage.ru_maxrss / (1024.0 * 1024.0);
        #else
        return (double)usage.ru_maxrss / 1024.0;
        #endif
    }
    return 0.0;
#endif
}

static double compute_structural_memory_mib(int n) {
    return (3.0 * (double)n * (double)n * sizeof(double)) / (1024.0 * 1024.0);
}

static int matrices_are_close(const double* a, const double* b, int n) {
    for (int i = 0; i < n * n; i++) {
        double diff = fabs(a[i] - b[i]);
        double bound = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * fabs(b[i]);
        if (diff > bound) {
            return 0;
        }
    }
    return 1;
}

static int parse_csv_row(char* line, double* row_dest, int n, int row_idx, const char* filepath) {
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == '\n')) {
        line[--len] = '\0';
    }
    if (len == 0) return -1;

    int col = 0;
    char* cur = line;

    while (*cur != '\0') {
        char* next_comma = strchr(cur, ',');
        if (next_comma) {
            *next_comma = '\0';
        }

        if (*cur == '\0') {
            fprintf(stderr, "[ERROR] Empty field at row %d, col %d in %s\n", row_idx, col, filepath);
            return 0;
        }

        if (col >= n) {
            fprintf(stderr, "[ERROR] Too many columns in %s: row %d has more than %d columns\n",
                    filepath, row_idx, n);
            return 0;
        }

        char* endptr = NULL;
        errno = 0;
        double val = strtod(cur, &endptr);

        if (errno == ERANGE) {
            fprintf(stderr, "[ERROR] Numerical range error (ERANGE) '%s' at (%d, %d) in %s\n",
                    cur, row_idx, col, filepath);
            return 0;
        }
        if (endptr == cur || *endptr != '\0') {
            fprintf(stderr, "[ERROR] Non-numeric entry '%s' at (%d, %d) in %s\n",
                    cur, row_idx, col, filepath);
            return 0;
        }
        if (isnan(val) || isinf(val)) {
            fprintf(stderr, "[ERROR] NaN/Inf entry at (%d, %d) in %s\n", row_idx, col, filepath);
            return 0;
        }

        row_dest[col] = val;
        col++;

        if (next_comma) {
            cur = next_comma + 1;
            if (*cur == '\0') {
                fprintf(stderr, "[ERROR] Trailing comma / empty field at end of row %d in %s\n", row_idx, filepath);
                return 0;
            }
        } else {
            break;
        }
    }

    if (col != n) {
        fprintf(stderr, "[ERROR] Column count mismatch in %s: row %d has %d columns, expected exactly %d\n",
                filepath, row_idx, col, n);
        return 0;
    }

    return 1;
}

static double* load_matrix(const char* filepath, int n) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "[ERROR] Cannot open matrix file: %s\n", filepath);
        return NULL;
    }

    double* matrix = (double*)malloc((size_t)n * n * sizeof(double));
    if (!matrix) {
        fprintf(stderr, "[ERROR] Memory allocation failed for %dx%d matrix from %s\n", n, n, filepath);
        fclose(file);
        return NULL;
    }

    char* line = (char*)malloc(LINE_BUFFER_SIZE);
    if (!line) {
        fprintf(stderr, "[ERROR] Line buffer allocation failed for %s\n", filepath);
        free(matrix);
        fclose(file);
        return NULL;
    }

    int row = 0;
    while (fgets(line, LINE_BUFFER_SIZE, file)) {
        size_t len = strlen(line);
        if (len == 0) continue;

        if (line[len - 1] != '\n' && !feof(file)) {
            fprintf(stderr, "[ERROR] Line buffer overflow: row %d exceeds %d bytes in %s\n",
                    row, LINE_BUFFER_SIZE, filepath);
            free(line);
            free(matrix);
            fclose(file);
            return NULL;
        }

        if (row >= n) {
            char* p = line;
            while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
            if (*p != '\0') {
                fprintf(stderr, "[ERROR] Extra row(s) beyond %d detected in %s\n", n, filepath);
                free(line);
                free(matrix);
                fclose(file);
                return NULL;
            }
            continue;
        }

        int res = parse_csv_row(line, &matrix[row * n], n, row, filepath);
        if (res <= 0) {
            if (res < 0) {
                fprintf(stderr, "[ERROR] Unexpected empty row %d in %s\n", row, filepath);
            }
            free(line);
            free(matrix);
            fclose(file);
            return NULL;
        }
        row++;
    }

    free(line);
    fclose(file);

    if (row != n) {
        fprintf(stderr, "[ERROR] Incomplete matrix in %s: read %d rows, expected exactly %d\n",
                filepath, row, n);
        free(matrix);
        return NULL;
    }
    return matrix;
}

void multiply_ijk(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

void seq_multiply_ikj(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = a[i * n + k];
            #pragma GCC ivdep
            for (int j = 0; j < n; j++) {
                c[i * n + j] += r * b[k * n + j];
            }
        }
    }
}

void seq_multiply_tiled_b32(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        int i_max = (ii + BLOCK_SIZE < n) ? ii + BLOCK_SIZE : n;
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            int j_max = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                int k_max = (kk + BLOCK_SIZE < n) ? kk + BLOCK_SIZE : n;
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        double r = a[i * n + k];
                        #pragma GCC ivdep
                        for (int j = jj; j < j_max; j++) {
                            c[i * n + j] += r * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int omp_multiply_ikj(const double* a, const double* b, double* c, int n, int threads) {
    int actual_threads = 1;
    memset(c, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double r = a[i * n + k];
                #pragma GCC ivdep
                for (int j = 0; j < n; j++) {
                    c[i * n + j] += r * b[k * n + j];
                }
            }
        }
    }
    return actual_threads;
}

int omp_multiply_tiled_b32(const double* a, const double* b, double* c, int n, int threads) {
    int actual_threads = 1;
    memset(c, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
        #pragma omp for schedule(static)
        for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
            int i_max = (ii + BLOCK_SIZE < n) ? ii + BLOCK_SIZE : n;
            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                    int j_max = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;
                for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                    int k_max = (kk + BLOCK_SIZE < n) ? kk + BLOCK_SIZE : n;
                    for (int i = ii; i < i_max; i++) {
                        for (int k = kk; k < k_max; k++) {
                            double r = a[i * n + k];
                            #pragma GCC ivdep
                            for (int j = jj; j < j_max; j++) {
                                c[i * n + j] += r * b[k * n + j];
                            }
                        }
                    }
                }
            }
        }
    }
    return actual_threads;
}

NO_TREE_VECTORIZE_ATTR
void seq_multiply_ikj_novec(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = a[i * n + k];
            for (int j = 0; j < n; j++) {
                c[i * n + j] += r * b[k * n + j];
            }
        }
    }
}

NO_TREE_VECTORIZE_ATTR
int omp_multiply_ikj_novec(const double* a, const double* b, double* c, int n, int threads) {
    int actual_threads = 1;
    memset(c, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double r = a[i * n + k];
                for (int j = 0; j < n; j++) {
                    c[i * n + j] += r * b[k * n + j];
                }
            }
        }
    }
    return actual_threads;
}

void seq_multiply_ikj_avx2(const double* a, const double* b, double* c, int n) {
    memset(c, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __m256d a_val = _mm256_set1_pd(a[i * n + k]);
            int j = 0;
            for (; j <= n - 4; j += 4) {
                __m256d c_val = _mm256_loadu_pd(&c[i * n + j]);
                __m256d b_val = _mm256_loadu_pd(&b[k * n + j]);
                c_val = _mm256_fmadd_pd(a_val, b_val, c_val);
                _mm256_storeu_pd(&c[i * n + j], c_val);
            }
            for (; j < n; j++) {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

int omp_multiply_ikj_avx2(const double* a, const double* b, double* c, int n, int threads) {
    int actual_threads = 1;
    memset(c, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                __m256d a_val = _mm256_set1_pd(a[i * n + k]);
                int j = 0;
                for (; j <= n - 4; j += 4) {
                    __m256d c_val = _mm256_loadu_pd(&c[i * n + j]);
                    __m256d b_val = _mm256_loadu_pd(&b[k * n + j]);
                    c_val = _mm256_fmadd_pd(a_val, b_val, c_val);
                    _mm256_storeu_pd(&c[i * n + j], c_val);
                }
                for (; j < n; j++) {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
    }
    return actual_threads;
}

static int validate_all_methods(const char* val_dir, const char* dense_dir) {
    printf("=================================================================\n");
    printf("Starting Rigorous Correctness Validation Suite...\n");
    printf("Tolerance: combined absolute 1e-4 + relative 1e-4\n");
    printf("=================================================================\n");

    int n = VALIDATION_SIZE;
    char path[MAX_PATH_LEN];

    snprintf(path, sizeof(path), "%s/known_A_4.csv", val_dir);
    double* known_a = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/known_B_4.csv", val_dir);
    double* known_b = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/expected_C_4.csv", val_dir);
    double* expected_c = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/identity_4.csv", val_dir);
    double* identity = load_matrix(path, n);
    snprintf(path, sizeof(path), "%s/zero_4.csv", val_dir);
    double* zero = load_matrix(path, n);

    if (!known_a || !known_b || !expected_c || !identity || !zero) {
        fprintf(stderr, "[FATAL] Failed to load 4x4 validation fixtures from %s\n", val_dir);
        if (known_a) free(known_a);
        if (known_b) free(known_b);
        if (expected_c) free(expected_c);
        if (identity) free(identity);
        if (zero) free(zero);
        return 0;
    }

    double* test_out = (double*)malloc((size_t)n * n * sizeof(double));
    if (!test_out) {
        fprintf(stderr, "[FATAL] Validation buffer allocation failed.\n");
        free(known_a); free(known_b); free(expected_c); free(identity); free(zero);
        return 0;
    }

    int all_passed = 1;

    #define TEST_FIXTURE_KERNEL(name, call_stmt) do { \
        call_stmt(known_a, known_b, test_out, n); \
        int pass_known = matrices_are_close(test_out, expected_c, n); \
        call_stmt(known_a, identity, test_out, n); \
        int pass_ident = matrices_are_close(test_out, known_a, n); \
        call_stmt(known_a, zero, test_out, n); \
        int pass_zero = matrices_are_close(test_out, zero, n); \
        int pass = pass_known && pass_ident && pass_zero; \
        printf("  [4x4 Fixtures] %-24s: %s (Known: %s, Ident: %s, Zero: %s)\n", \
               name, pass ? "PASS" : "FAIL", \
               pass_known ? "OK" : "FAIL", pass_ident ? "OK" : "FAIL", pass_zero ? "OK" : "FAIL"); \
        if (!pass) all_passed = 0; \
    } while(0)

    #define TEST_FIXTURE_OMP_KERNEL(name, call_stmt, th) do { \
        int act = call_stmt(known_a, known_b, test_out, n, th); \
        (void)act; \
        int pass_known = matrices_are_close(test_out, expected_c, n); \
        call_stmt(known_a, identity, test_out, n, th); \
        int pass_ident = matrices_are_close(test_out, known_a, n); \
        call_stmt(known_a, zero, test_out, n, th); \
        int pass_zero = matrices_are_close(test_out, zero, n); \
        int pass = pass_known && pass_ident && pass_zero; \
        printf("  [4x4 Fixtures] %-20s (p=%d): %s (Known: %s, Ident: %s, Zero: %s)\n", \
               name, th, pass ? "PASS" : "FAIL", \
               pass_known ? "OK" : "FAIL", pass_ident ? "OK" : "FAIL", pass_zero ? "OK" : "FAIL"); \
        if (!pass) all_passed = 0; \
    } while(0)

    TEST_FIXTURE_KERNEL("canonical_ijk", multiply_ijk);
    TEST_FIXTURE_KERNEL("seq_ikj (autovec)", seq_multiply_ikj);
    TEST_FIXTURE_KERNEL("seq_tiled_b32", seq_multiply_tiled_b32);
    TEST_FIXTURE_KERNEL("seq_ikj_novec", seq_multiply_ikj_novec);
    TEST_FIXTURE_KERNEL("seq_ikj_avx2", seq_multiply_ikj_avx2);

    for (int p_idx = 0; p_idx < 3; p_idx++) {
        int th = DEFAULT_THREADS[p_idx];
        TEST_FIXTURE_OMP_KERNEL("omp_ikj", omp_multiply_ikj, th);
        TEST_FIXTURE_OMP_KERNEL("omp_tiled_b32", omp_multiply_tiled_b32, th);
        TEST_FIXTURE_OMP_KERNEL("omp_ikj_novec", omp_multiply_ikj_novec, th);
        TEST_FIXTURE_OMP_KERNEL("omp_ikj_avx2", omp_multiply_ikj_avx2, th);
    }

    #undef TEST_FIXTURE_KERNEL
    #undef TEST_FIXTURE_OMP_KERNEL

    free(known_a); free(known_b); free(expected_c); free(identity); free(zero);
    free(test_out);

    if (!all_passed) {
        fprintf(stderr, "[FATAL] 4x4 fixture validation failed!\n");
        return 0;
    }

    printf("\nPhase 2: Benchmark Cross-Validation at n = 250...\n");
    if (dense_dir) {
        char file_a[MAX_PATH_LEN], file_b[MAX_PATH_LEN];
        snprintf(file_a, sizeof(file_a), "%s/A_250.csv", dense_dir);
        snprintf(file_b, sizeof(file_b), "%s/B_250.csv", dense_dir);

        double* bench_a = load_matrix(file_a, 250);
        double* bench_b = load_matrix(file_b, 250);
        if (!bench_a || !bench_b) {
            fprintf(stderr, "[FATAL] Failed loading n=250 dataset for cross-validation.\n");
            if (bench_a) free(bench_a);
            if (bench_b) free(bench_b);
            return 0;
        }

        double* ref_c = (double*)malloc((size_t)250 * 250 * sizeof(double));
        double* test_c = (double*)malloc((size_t)250 * 250 * sizeof(double));

        if (!ref_c || !test_c) {
            fprintf(stderr, "[FATAL] Failed allocating buffers for n=250 cross-validation.\n");
            if (ref_c) free(ref_c);
            if (test_c) free(test_c);
            free(bench_a); free(bench_b);
            return 0;
        }

        seq_multiply_ikj(bench_a, bench_b, ref_c, 250);

        seq_multiply_tiled_b32(bench_a, bench_b, test_c, 250);
        int pass_tiled = matrices_are_close(ref_c, test_c, 250);
        printf("  [Cross-Check n=250] seq_tiled_b32     : %s\n", pass_tiled ? "PASS" : "FAIL");
        if (!pass_tiled) all_passed = 0;

        seq_multiply_ikj_novec(bench_a, bench_b, test_c, 250);
        int pass_novec = matrices_are_close(ref_c, test_c, 250);
        printf("  [Cross-Check n=250] seq_ikj_novec     : %s\n", pass_novec ? "PASS" : "FAIL");
        if (!pass_novec) all_passed = 0;

        seq_multiply_ikj_avx2(bench_a, bench_b, test_c, 250);
        int pass_avx2 = matrices_are_close(ref_c, test_c, 250);
        printf("  [Cross-Check n=250] seq_ikj_avx2      : %s\n", pass_avx2 ? "PASS" : "FAIL");
        if (!pass_avx2) all_passed = 0;

        for (int p_idx = 0; p_idx < DEFAULT_NUM_THREADS; p_idx++) {
            int th = DEFAULT_THREADS[p_idx];
            
            omp_multiply_ikj(bench_a, bench_b, test_c, 250, th);
            int pass_omp_ikj = matrices_are_close(ref_c, test_c, 250);
            if (!pass_omp_ikj) {
                printf("  [Cross-Check n=250] omp_ikj (p=%2d)    : FAIL\n", th);
                all_passed = 0;
            }

            omp_multiply_tiled_b32(bench_a, bench_b, test_c, 250, th);
            int pass_omp_tiled = matrices_are_close(ref_c, test_c, 250);
            if (!pass_omp_tiled) {
                printf("  [Cross-Check n=250] omp_tiled (p=%2d)  : FAIL\n", th);
                all_passed = 0;
            }

            omp_multiply_ikj_novec(bench_a, bench_b, test_c, 250, th);
            int pass_omp_novec = matrices_are_close(ref_c, test_c, 250);
            if (!pass_omp_novec) {
                printf("  [Cross-Check n=250] omp_novec (p=%2d)  : FAIL\n", th);
                all_passed = 0;
            }

            omp_multiply_ikj_avx2(bench_a, bench_b, test_c, 250, th);
            int pass_omp_avx2 = matrices_are_close(ref_c, test_c, 250);
            if (!pass_omp_avx2) {
                printf("  [Cross-Check n=250] omp_avx2 (p=%2d)   : FAIL\n", th);
                all_passed = 0;
            }
        }

        if (all_passed) {
            printf("  [Cross-Check n=250] All OpenMP kernels passed across all p.\n");
        }

        free(ref_c);
        free(test_c);
        free(bench_a);
        free(bench_b);
    }

    if (all_passed) {
        printf("\n>>> [SUCCESS] All validation tests passed cleanly! <<<\n\n");
        return 1;
    } else {
        fprintf(stderr, "\n>>> [FAILURE] One or more validation tests failed! <<<\n\n");
        return 0;
    }
}

typedef void (*SeqMatMulFn)(const double*, const double*, double*, int);
typedef int (*OmpMatMulFn)(const double*, const double*, double*, int, int);

static void run_sequential_benchmark(
    FILE* writer,
    const char* method_name,
    const char* algorithm,
    const char* vectorisation,
    SeqMatMulFn fn,
    const double* a,
    const double* b,
    double* c,
    int n,
    double structural_mem_mib
) {
    printf("  %-18s (seq, 1 th)  | Warm-up... ", method_name);
    fn(a, b, c, n);

    double warm_chk = 0.0;
    for (size_t idx = 0; idx < (size_t)n * n; idx++) {
        warm_chk += c[idx];
    }
    result_sink = warm_chk;
    printf("Done. Repetitions: ");

    int skip = 0;
    for (int rep = 1; rep <= NUM_REPETITIONS; rep++) {
        if (skip) break;

        double t0 = get_time_seconds();
        fn(a, b, c, n);
        double t1 = get_time_seconds();

        double chk = 0.0;
        for (size_t idx = 0; idx < (size_t)n * n; idx++) {
            chk += c[idx];
        }
        result_sink = chk;

        double elapsed = t1 - t0;
        double peak_mem = get_peak_memory_mib();

        printf("%.4fs ", elapsed);
        fflush(stdout);

        fprintf(writer, "%s,%s,%s,%s,%d,%d,%d,%d,%s,%d,%.6f,%.2f,%.2f\n",
                method_name, algorithm, "sequential", vectorisation,
                n, 1, 1, 0, "double", rep, elapsed, peak_mem, structural_mem_mib);
        fflush(writer);

        if (elapsed > BUDGET_SECONDS) {
            printf("\n    [REACTIVE BUDGET EXCEEDED] Repetition %d elapsed %.2fs > %.0fs threshold. Skipping.\n",
                   rep, elapsed, BUDGET_SECONDS);
            skip = 1;
        }
    }
    printf("\n");
}

static void run_openmp_benchmark(
    FILE* writer,
    const char* method_name,
    const char* algorithm,
    const char* vectorisation,
    OmpMatMulFn fn,
    const double* a,
    const double* b,
    double* c,
    int n,
    int threads,
    int hardware_contexts,
    double structural_mem_mib
) {
    int oversubscribed = (threads > hardware_contexts) ? 1 : 0;
    printf("  %-18s (omp, %2d th%s) | Warm-up... ",
           method_name, threads, oversubscribed ? " [OVERSUB]" : "");
    int actual_threads = fn(a, b, c, n, threads);

    double warm_chk = 0.0;
    for (size_t idx = 0; idx < (size_t)n * n; idx++) {
        warm_chk += c[idx];
    }
    result_sink = warm_chk;
    printf("Done (actual: %d th). Repetitions: ", actual_threads);

    int skip = 0;
    for (int rep = 1; rep <= NUM_REPETITIONS; rep++) {
        if (skip) break;

        double t0 = get_time_seconds();
        actual_threads = fn(a, b, c, n, threads);
        double t1 = get_time_seconds();

        double chk = 0.0;
        for (size_t idx = 0; idx < (size_t)n * n; idx++) {
            chk += c[idx];
        }
        result_sink = chk;

        double elapsed = t1 - t0;
        double peak_mem = get_peak_memory_mib();

        printf("%.4fs ", elapsed);
        fflush(stdout);

        fprintf(writer, "%s,%s,%s,%s,%d,%d,%d,%d,%s,%d,%.6f,%.2f,%.2f\n",
                method_name, algorithm, "openmp", vectorisation,
                n, threads, actual_threads, oversubscribed, "double", rep, elapsed, peak_mem, structural_mem_mib);
        fflush(writer);

        if (elapsed > BUDGET_SECONDS) {
            printf("\n    [REACTIVE BUDGET EXCEEDED] Repetition %d elapsed %.2fs > %.0fs threshold. Skipping.\n",
                   rep, elapsed, BUDGET_SECONDS);
            skip = 1;
        }
    }
    printf("\n");
}

static void record_environment_metadata(const char* filepath, int hardware_contexts) {
    FILE* f = fopen(filepath, "w");
    if (!f) return;

    #ifdef _OPENMP
    fprintf(f, "openmp_spec_version: %d\n", _OPENMP);
    #else
    fprintf(f, "openmp_spec_version: not_defined\n");
    #endif
    fprintf(f, "omp_num_procs: %d\n", hardware_contexts);
    fprintf(f, "omp_max_threads: %d\n", omp_get_max_threads());
    fprintf(f, "omp_dynamic_status: %d\n", omp_get_dynamic());
    fprintf(f, "compiler: GCC %s\n", __VERSION__);
    fprintf(f, "build_flags: %s\n", BUILD_FLAGS);

    const char* proc_bind = getenv("OMP_PROC_BIND");
    fprintf(f, "omp_proc_bind: %s\n", proc_bind ? proc_bind : "unset");

    const char* places = getenv("OMP_PLACES");
    fprintf(f, "omp_places: %s\n", places ? places : "unset");

    const char* schedule = getenv("OMP_SCHEDULE");
    fprintf(f, "omp_schedule: %s\n", schedule ? schedule : "unset");

    const char* thread_limit = getenv("OMP_THREAD_LIMIT");
    fprintf(f, "omp_thread_limit: %s\n", thread_limit ? thread_limit : "unset");

    const char* max_active_levels = getenv("OMP_MAX_ACTIVE_LEVELS");
    fprintf(f, "omp_max_active_levels: %s\n", max_active_levels ? max_active_levels : "unset");

    time_t now = time(NULL);
    fprintf(f, "timestamp: %ld\n", (long)now);

    fclose(f);
}

int main(int argc, char* argv[]) {
    omp_set_dynamic(0);

    const char* base_dir = ".";
    int validate_only = 0;
    int is_pilot = 0;
    int run_simd = 0;
    const char* custom_output = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate-only") == 0) {
            validate_only = 1;
        } else if (strcmp(argv[i], "--pilot") == 0) {
            is_pilot = 1;
        } else if (strcmp(argv[i], "--simd") == 0) {
            run_simd = 1;
        } else if (strcmp(argv[i], "--no-simd") == 0) {
            run_simd = 0;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            custom_output = argv[++i];
        } else if (argv[i][0] != '-') {
            base_dir = argv[i];
        }
    }

    int hardware_contexts = omp_get_num_procs();

    char val_dir[MAX_DIR_LEN], dense_dir[MAX_DIR_LEN], results_dir[MAX_DIR_LEN], results_path[MAX_PATH_LEN], env_path[MAX_PATH_LEN];

    if (snprintf(val_dir, sizeof(val_dir), "%s/data/validation", base_dir) >= (int)sizeof(val_dir) ||
        snprintf(dense_dir, sizeof(dense_dir), "%s/data/dense", base_dir) >= (int)sizeof(dense_dir) ||
        snprintf(results_dir, sizeof(results_dir), "%s/results", base_dir) >= (int)sizeof(results_dir)) {
        fprintf(stderr, "[FATAL] Base directory path exceeds buffer limits.\n");
        return EXIT_FAILURE;
    }

    if (custom_output) {
        snprintf(results_path, sizeof(results_path), "%s", custom_output);
    } else {
        const char* default_csv = is_pilot ? "pilot_results.csv" : "parallel_results.csv";
        if (snprintf(results_path, sizeof(results_path), "%s/%s", results_dir, default_csv) >= (int)sizeof(results_path)) {
            fprintf(stderr, "[FATAL] Results path exceeds buffer limits.\n");
            return EXIT_FAILURE;
        }
    }

    snprintf(env_path, sizeof(env_path), "%s/environment_info.txt", results_dir);

    printf("=================================================================\n");
    printf("Assignment 3: Parallel Matrix Multiplication\n");
    printf("Author: Abel Castro Villar (ULPGC)\n");
    printf("Hardware Contexts (omp_get_num_procs): %d | Max Threads: %d\n",
           hardware_contexts, omp_get_max_threads());
    printf("Dynamic Threads: %s | Mode: %s\n",
           omp_get_dynamic() ? "Enabled" : "Disabled (Enforced)",
           is_pilot ? "PILOT RUN" : "FULL BENCHMARK");
    #ifdef _OPENMP
    printf("OpenMP Specification Version: %d\n", _OPENMP);
    #endif
    printf("Target Results Output: %s\n", results_path);
    printf("=================================================================\n\n");

    if (!validate_all_methods(val_dir, dense_dir)) {
        fprintf(stderr, "[FATAL] Correctness validation failed. Aborting execution.\n");
        return EXIT_FAILURE;
    }

    if (validate_only) {
        printf("[INFO] Flag --validate-only provided. Exiting successfully.\n");
        return EXIT_SUCCESS;
    }

    MKDIR(results_dir);
    record_environment_metadata(env_path, hardware_contexts);

    FILE* test_exists = fopen(results_path, "r");
    if (test_exists) {
        fclose(test_exists);
        char backup_path[MAX_PATH_LEN + 16];
        snprintf(backup_path, sizeof(backup_path), "%s.bak", results_path);
        printf("[NOTICE] Output file '%s' exists. Creating backup at '%s'\n", results_path, backup_path);
        remove(backup_path);
        rename(results_path, backup_path);
    }

    FILE* writer = fopen(results_path, "w");
    if (!writer) {
        fprintf(stderr, "[FATAL] Cannot open results file for writing: %s\n", results_path);
        return EXIT_FAILURE;
    }

    fprintf(writer, "method,algorithm,parallel_model,vectorisation,n,threads,actual_threads,oversubscribed,numeric_type,repetition,time_s,peak_memory_mib,structural_memory_mib\n");
    fflush(writer);

    const int* sizes_to_run = is_pilot ? PILOT_SIZES : DEFAULT_SIZES;
    int num_sizes = is_pilot ? PILOT_NUM_SIZES : DEFAULT_NUM_SIZES;

    const int* threads_to_run = is_pilot ? PILOT_THREADS : DEFAULT_THREADS;
    int num_threads = is_pilot ? PILOT_NUM_THREADS : DEFAULT_NUM_THREADS;

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes_to_run[s];
        double structural_mem_mib = compute_structural_memory_mib(n);

        printf("\n=================================================================\n");
        printf("Benchmarking Matrix Dimension: n = %d\n", n);
        printf("Structural Memory Required: %.2f MiB (Budget Limit: %.2f MiB)\n",
               structural_mem_mib, PREVENTIVE_MAX_MEM_MIB);
        printf("=================================================================\n");

        if (structural_mem_mib > PREVENTIVE_MAX_MEM_MIB) {
            printf("[PREVENTIVE BUDGET] Dimension n=%d requires %.2f MiB, exceeding limit of %.2f MiB. Skipping.\n",
                   n, structural_mem_mib, PREVENTIVE_MAX_MEM_MIB);
            continue;
        }

        char file_a[MAX_PATH_LEN], file_b[MAX_PATH_LEN];
        snprintf(file_a, sizeof(file_a), "%s/A_%d.csv", dense_dir, n);
        snprintf(file_b, sizeof(file_b), "%s/B_%d.csv", dense_dir, n);

        double* a = load_matrix(file_a, n);
        double* b = load_matrix(file_b, n);
        double* c = (double*)malloc((size_t)n * n * sizeof(double));

        if (!a || !b || !c) {
            fprintf(stderr, "[FATAL] Memory allocation or loading failed for n=%d\n", n);
            if (a) free(a);
            if (b) free(b);
            if (c) free(c);
            fclose(writer);
            return EXIT_FAILURE;
        }

        printf("\n-- Core Thread Scaling Kernels (autovec_O3) --\n");

        run_sequential_benchmark(writer, "seq_ikj", "ikj", "autovec_O3",
                                seq_multiply_ikj, a, b, c, n, structural_mem_mib);

        run_sequential_benchmark(writer, "seq_tiled_b32", "tiled_b32", "autovec_O3",
                                seq_multiply_tiled_b32, a, b, c, n, structural_mem_mib);

        for (int p_idx = 0; p_idx < num_threads; p_idx++) {
            int p = threads_to_run[p_idx];
            run_openmp_benchmark(writer, "omp_ikj", "ikj", "autovec_O3",
                                omp_multiply_ikj, a, b, c, n, p, hardware_contexts, structural_mem_mib);
        }

        for (int p_idx = 0; p_idx < num_threads; p_idx++) {
            int p = threads_to_run[p_idx];
            run_openmp_benchmark(writer, "omp_tiled_b32", "tiled_b32", "autovec_O3",
                                omp_multiply_tiled_b32, a, b, c, n, p, hardware_contexts, structural_mem_mib);
        }

        if (run_simd && !is_pilot && (n == 1000 || n == 2000)) {
        printf("\n-- Optional controlled vectorisation study (n = %d) --\n", n);

            run_sequential_benchmark(writer, "seq_ikj_novec", "ikj", "no_vec",
                                    seq_multiply_ikj_novec, a, b, c, n, structural_mem_mib);

            for (int v_idx = 0; v_idx < NUM_VEC_THREAD_COUNTS; v_idx++) {
                int p = VEC_THREAD_COUNTS[v_idx];
                run_openmp_benchmark(writer, "omp_ikj_novec", "ikj", "no_vec",
                                    omp_multiply_ikj_novec, a, b, c, n, p, hardware_contexts, structural_mem_mib);
            }

            run_sequential_benchmark(writer, "seq_ikj_avx2", "ikj", "avx2_intrinsics",
                                    seq_multiply_ikj_avx2, a, b, c, n, structural_mem_mib);

            for (int v_idx = 0; v_idx < NUM_VEC_THREAD_COUNTS; v_idx++) {
                int p = VEC_THREAD_COUNTS[v_idx];
                run_openmp_benchmark(writer, "omp_ikj_avx2", "ikj", "avx2_intrinsics",
                                    omp_multiply_ikj_avx2, a, b, c, n, p, hardware_contexts, structural_mem_mib);
            }
        }

        free(a);
        free(b);
        free(c);
    }

    fclose(writer);

    printf("\n=================================================================\n");
    printf("[BENCHMARK COMPLETE] All requested experiments finished successfully.\n");
    printf("Raw CSV Results Saved to: %s\n", results_path);
    printf("Environment Metadata:     %s\n", env_path);
    printf("=================================================================\n");

    return EXIT_SUCCESS;
}
