# Assignment 3 — Parallel Matrix Multiplication

**Course:** Big Data, 2026–2027  
**Author:** Abel Castro Villar  
**Repository:** <https://github.com/abelcasv/Assignment03_Abel>

This study measures the effect of OpenMP thread count on dense square matrix multiplication in C. Its required comparison is between the loop-interchanged sequential kernel and the same kernel with OpenMP; the blocked (`B=32`) implementation is a secondary algorithm comparison. SIMD is optional and is not part of the reported core experiment.

## Study and measurement protocol

- Matrix dimensions are `n ∈ {250, 500, 750, 1000, 1500, 2000}`. Each input pair is shared by all methods at a given `n`; values are IEEE 754 binary64 (`double`).
- `src/generate_data.py` uses Python's `random.Random(42)` MT19937 generator. It draws each value with `random()` (`[0,1)`) and writes six digits after the decimal point. Rounding means stored CSV values can equal `1.000000`; the observed stored range is `[0,1]`. The script writes `data/SHA256SUMS.txt` for all fixtures and dense matrices. The Python version used for analysis and generation is in `results/analysis_environment.txt`; the 12 current dense input files were regenerated and their SHA-256 hashes matched. Random streams are not assumed to match other languages.
- Tested OpenMP team sizes are `p ∈ {1, 2, 4, 8, 10, 12, 16, 24}`. The host exposes 12 logical processors; 16 and 24 are oversubscribed. `OMP_PROC_BIND=false`, `OMP_PLACES` unset, and `OMP_DYNAMIC` disabled in the program. No core affinity or P-core/E-core placement is claimed.
- Each configuration has one warm-up and five measured repetitions. All raw observations are retained. The report uses the median and interquartile range (IQR; 25th to 75th percentile).
- The monotonic timer covers the kernel call, including zero-initialisation of output matrix `C` and OpenMP team/loop overhead. Input loading and allocation are outside the timed region. A checksum over the full output is calculated after timing and stored in a volatile sink.
- Correctness is checked on known, identity and zero fixtures, then against a trusted reference for the dense `n=250` input. The comparison tolerance is `|a-b| ≤ 1e-4 + 1e-4|b|`.
- The benchmark uses a 120-second per-repetition stop threshold: if a completed call exceeds it, that sample is retained and remaining repetitions for that configuration are skipped. It checks a 2,048 MiB structural matrix-storage limit before running each size. The largest size run is 2,000; larger sizes were not attempted. Three `double` matrices require `3 × 8n²` bytes; this excludes runtime and allocator overhead.
- `peak_memory_mib` is the process resident-set high-water mark on Windows (`GetProcessMemoryInfo` / `PeakWorkingSetSize`). As a single process runs all configurations, this is a process-level high-water observation, not an independent peak-memory value for each configuration. Use `structural_memory_mib` for the per-size three-array storage estimate. The two metrics are different and are not interchangeable.
- The reference implementation is `canonical_ijk` in the source. The core timing study includes `seq_ikj`, `omp_ikj` and the corresponding sequential/OpenMP blocked kernels. Optional scalar/AVX2 modes are opt-in with `--simd` and excluded from the core CSV and conclusions.

## Environment

The benchmark host, OS build, available memory, GCC version, OpenMP runtime settings and build flags are captured in `results/environment_host.txt` and `results/environment_info.txt`. Analysis-language/library versions are in `results/analysis_environment.txt`. Processor core counts and architecture are taken from the Intel product specification; the experiment does not pin workers to particular cores. The compiler uses `-Wall -Wextra -O3 -std=c11 -fopenmp -march=native`; the resulting binary is machine-specific.

## Repository layout

```text
data/validation/                 Small known-answer fixtures
data/dense/                      Generated shared inputs (regenerate below)
data/SHA256SUMS.txt              Checksums for validation and dense CSV inputs
src/parallel_benchmark.c         C11/OpenMP benchmark and correctness checks
src/generate_data.py             Deterministic input generator
src/Makefile                     Build, validation and pilot targets
src/capture_environment.ps1      Host metadata capture
src/analysis/analyze_assignment03.py  CSV checks, summaries and plots
src/analysis/build_report_pdf.py      Validated, data-driven PDF report builder
results/parallel_results.csv     Raw full-run observations
results/parallel_results_summary.csv  Derived medians, quartiles, speedup and efficiency
results/parallel_results_plots/  Reproducible analysis figures
results/environment_*.txt       Host, compiler and runtime capture
results/analysis_environment.txt Analysis-language and library versions
report/Assignment03_Report.tex   Research report source
report/Assignment03_Report.pdf   Final PDF generated from the validated CSV
report/plots/                    Figures embedded in the report
results/archive/                 Preserved pre-repair data; not the final dataset
```

The dense input files are generated and intentionally excluded from version control. Regenerate them before a run; this replaces the files in `data/dense/`.

## Build, validate and run (Windows + MSYS2 UCRT64)

Requires the MSYS2 UCRT64 GCC toolchain with OpenMP and `make`, plus Python 3.12 or later for data generation and analysis.

From the repository root in PowerShell:

```powershell
python src/generate_data.py
& .\src\capture_environment.ps1
$env:PATH = 'C:\msys64\ucrt64\bin;C:\msys64\usr\bin;' + $env:PATH
make -B -C src
make -C src validate
```

Run the full core experiment. This overwrites `results/parallel_results.csv` after preserving any existing file as `.bak`:

```powershell
$env:OMP_PROC_BIND = 'false'
Remove-Item Env:OMP_PLACES -ErrorAction SilentlyContinue
Push-Location src
try { .\parallel_benchmark.exe .. } finally { Pop-Location }
```

To run the smaller pilot, use `make -C src pilot`; it writes a separate `results/pilot_results.csv`. To run the optional vectorisation extension, pass `--simd` to the executable and save its output separately with `--output ..\results\simd_results.csv`. The AVX2 path requires a supporting x86-64 processor and must not be mixed with the core thread-scaling conclusions.

Install analysis dependencies and regenerate summary/figures from the raw CSV:

```powershell
python -m pip install -r requirements.txt
python src/analysis/analyze_assignment03.py
Copy-Item results/parallel_results_plots/*.png report/plots/ -Force
python src/analysis/build_report_pdf.py
```

The analyzer writes `results/parallel_results_summary.csv` and figures to `results/parallel_results_plots/`; copy the figures to `report/plots/` before building the PDF. `build_report_pdf.py` validates the raw data again and generates `report/Assignment03_Report.pdf`. The editable LaTeX source is also provided and compiles in the Codex LaTeX editor. To analyse a pilot or optional run, pass its CSV path as the positional argument and add `--allow-partial` for a deliberately incomplete grid.

## Raw data and derived measures

`results/parallel_results.csv` is the source of record. Each row labels method, algorithm, vectorisation mode, `n`, requested and actual team size, oversubscription, numeric type, repetition, elapsed seconds and both memory quantities. Do not replace or delete raw samples; regenerate the summary and plots from the CSV.

Earlier, pre-repair raw and pilot data are retained under `results/archive/` for transparency. Use only `results/parallel_results.csv` for the final report; archived files contain the superseded experimental setup.

For each `n`, speedup is `S(p) = median(T_seq(n)) / median(T_p(n))` using the matching sequential algorithm, and efficiency is `E(p) = S(p) / p`. The `p=1` OpenMP result is reported to expose runtime overhead. A speedup above one is not assumed: the measurements determine whether parallel execution helps at each size.

The analysis script checks schema, finite and positive values, duplicate samples, five repetitions per configuration, matching sequential baselines, actual versus requested team size, oversubscription labels, and the full required size/thread grid before writing the final summary and figures.

## AI assistance disclosure

AI assistance must be disclosed in the submitted report. The report records the known tools and their roles; the student should review that statement for completeness before submission and remains responsible for understanding, validating and defending the work.
