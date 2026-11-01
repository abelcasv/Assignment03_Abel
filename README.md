# Assignment 3 — Parallel Matrix Multiplication

**Course:** Big Data, 2026–2027  
**Author:** Abel Castro Villar  

This study measures the effect of OpenMP thread count on dense square matrix multiplication in C11. Its required primary comparison is between the loop-interchanged sequential kernel (`seq_ikj`) and the equivalent OpenMP parallel kernel (`omp_ikj`); the cache-blocked (`B_s=32`) implementation (`seq_tiled_b32` vs. `omp_tiled_b32`) is evaluated against its own matched sequential baseline as a secondary algorithm comparison. SIMD vectorisation extensions are optional and excluded from the core thread-scaling experiment.

## Study and measurement protocol

- **Dimensions and inputs:** Matrix dimensions are `n in {250, 500, 750, 1000, 1500, 2000}`, where `n` is the side length of an `n x n` square matrix. Each input pair is shared by all methods at a given `n`; values are IEEE 754 binary64 (`double`).
- **Deterministic generator:** `src/generate_data.py` uses Python's `random.Random(42)` MT19937 generator. It draws each value with `random()` in `[0,1)` and writes six digits after the decimal point (stored range `[0.000000, 1.000000]`). The script writes `data/SHA256SUMS.txt` for all validation fixtures and dense matrices. All 12 dense input files were verified against their SHA-256 hashes.
- **Thread configurations:** Tested OpenMP team sizes are `p in {1, 2, 4, 8, 10, 12, 16, 24}`. The host processor (Intel Core i5-1335U) exposes 10 physical cores (2 P-cores + 8 E-cores) and 12 logical processors; team sizes 16 and 24 are explicitly flagged as oversubscribed (`oversubscribed=1`). Dynamic thread adjustment is disabled (`omp_set_dynamic(0)`), `OMP_PROC_BIND=false`, and `OMP_PLACES` is unset. Actual team sizes are verified inside every parallel region via `omp_get_num_threads()`.
- **Repetitions and summary statistics:** Each configuration executes one untimed warm-up run followed by five measured repetitions. All raw observations are retained in `results/parallel_results.csv`. Central tendency and variability are reported via the median and interquartile range (`IQR = Q3 - Q1`; 25th to 75th percentile).
- **Timing boundary and output consumption:** The monotonic timer (`QueryPerformanceCounter` on Windows; `clock_gettime(CLOCK_MONOTONIC)` on POSIX) brackets a single kernel call, including zero-initialisation (`memset`) of output matrix `C` and OpenMP team/loop scheduling overhead. Input CSV loading and heap allocation (`malloc`) occur strictly outside the timed region. Immediately after timing, a full-matrix checksum is accumulated into a `volatile` sink outside the timed window to prevent dead-code elimination.
- **Correctness validation and tolerances:** Correctness is validated before timing on `4 x 4` analytical known-answer, identity (`A x I = A`), and zero (`A x 0 = 0`) fixtures, followed by cross-validation of all kernels and thread counts against the trusted reference on the `n=250` benchmark input. Comparisons enforce the combined tolerance `|a - b| <= 1e-4 + 1e-4|b|`, which accounts for six-decimal CSV input rounding (`+-5e-7`) and non-associative IEEE 754 floating-point summation order across compiler auto-vectorisation (`-O3 -march=native`) and `32 x 32` cache blocking.
- **Resource budgets:** The benchmark enforces a preventive three-array structural memory budget of `2,048 MiB` (`24n^2` bytes) prior to buffer allocation and a reactive per-repetition stop threshold of `120 s` (if a completed call exceeds `120 s`, that sample is retained and remaining repetitions for that configuration are skipped). The largest tested size meeting both budgets is `n=2000` (`91.55 MiB` structural memory); larger dimensions were not attempted.
- **Memory metrics:** `peak_memory_mib` records the OS process resident-set high-water mark (`GetProcessMemoryInfo` / `PeakWorkingSetSize`). Because all configurations execute within a single process, this metric reflects a cumulative process-lifetime high-water mark (`97.36 MiB` overall) rather than an independent per-configuration peak. `structural_memory_mib` reports the exact per-size three-array footprint (`3 x 8n^2` bytes). Both metrics are reported separately and are not interchangeable.

## Environment

The benchmark host, OS build, physical memory, GCC version, OpenMP runtime parameters, and build flags are recorded in `results/environment_host.txt` and `results/environment_info.txt`. Python analysis package versions are recorded in `results/analysis_environment.txt`. The C11 benchmark is compiled with MSYS2 UCRT64 GCC 16.1.0 using `-Wall -Wextra -O3 -std=c11 -fopenmp -march=native -lpsapi`.

## Repository layout

    data/validation/                      Small 4x4 correctness fixtures
    data/dense/                           Generated shared dense inputs (regenerated locally)
    data/SHA256SUMS.txt                   SHA-256 checksums for validation and dense CSV inputs
    src/parallel_benchmark.c              C11/OpenMP benchmark harness and correctness suite
    src/generate_data.py                  Deterministic input and SHA-256 manifest generator
    src/Makefile                          Cross-platform build, validation, and pilot targets
    src/capture_environment.ps1           Host hardware and OS metadata capture script
    src/analysis/analyze_assignment03.py  Strict CSV validator, summary generator, and plotter
    results/parallel_results.csv          Raw full-run benchmark observations (540 rows)
    results/parallel_results_summary.csv  Derived medians, quartiles, IQR, speedup, and efficiency
    results/parallel_results_plots/       Reproducible 300 DPI analysis figures
    results/environment_*.txt             Recorded host, compiler, and OpenMP runtime metadata
    results/analysis_environment.txt      Recorded Python and library versions
    report/Assignment03_Report.tex        LaTeX research report source
    report/Assignment03_Report.pdf        Compiled final research report PDF
    report/plots/                         Figures embedded in the LaTeX report
    requirements.txt                      Python dependencies for analysis and plotting

The large dense CSV files in `data/dense/` are excluded from version control via `.gitignore` and can be deterministically regenerated with `python src/generate_data.py`.

## Build, validate and run (Windows + MSYS2 UCRT64)

Requires the MSYS2 UCRT64 GCC toolchain (`gcc`, `make`, OpenMP) and Python 3.12+.

1. **Generate datasets, capture host metadata, compile, and run pre-benchmark validation** (from the repository root in PowerShell):

       python src/generate_data.py
       & .\src\capture_environment.ps1
       $env:PATH = 'C:\msys64\ucrt64\bin;C:\msys64\usr\bin;' + $env:PATH
       make -B -C src
       make -C src validate

2. **Execute the full core benchmark** (automatically backs up any existing `results/parallel_results.csv` to `.bak`):

       $env:OMP_PROC_BIND = 'false'
       Remove-Item Env:OMP_PLACES -ErrorAction SilentlyContinue
       Push-Location src
       try { .\parallel_benchmark.exe .. } finally { Pop-Location }

   *(To run the smaller pilot verification grid `n in {250, 500}`, run `make -C src pilot`, which writes `results/pilot_results.csv`).*

3. **Validate raw CSV data and regenerate summary statistics and 300 DPI figures:**

       python -m pip install -r requirements.txt
       python src/analysis/analyze_assignment03.py
       Copy-Item results/parallel_results_plots/*.png report/plots/ -Force

4. **Compile the final PDF report from the LaTeX source:**

       pdflatex -output-directory=report report/Assignment03_Report.tex
       pdflatex -output-directory=report report/Assignment03_Report.tex

## Raw data and derived measures

`results/parallel_results.csv` is the authoritative raw measurement record (540 observations across 108 configurations). For each matrix dimension `n` and algorithm, speedup is defined as `S(p) = T_seq(n) / T_p(n)` using the median of 5 repetitions against the matched sequential counterpart, and parallel efficiency is `E(p) = S(p) / p`. Single-thread OpenMP (`p=1`) is explicitly compared against `T_seq(n)` to quantify runtime overhead.

## AI assistance disclosure

In compliance with the course Generative AI policy, the submitted report includes an explicit declaration detailing the supporting AI tools used (`Gemini`, `Claude`, and `OpenAI Codex`) and their scope. All experimental runs were executed locally on the author's hardware, and the author takes full responsibility for defending the code, data, and conclusions.