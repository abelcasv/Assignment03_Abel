"""Build the final, data-driven Assignment 3 PDF from raw CSV and report figures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_assignment03 import (  # noqa: E402
    compute_summary_statistics,
    validate_parallel_dataframe,
)

RAW = ROOT / "results" / "parallel_results.csv"
OUT = ROOT / "report" / "Assignment03_Report.pdf"
FIGURES = ROOT / "report" / "plots"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F9E")
PALE = colors.HexColor("#EDF3F8")
INK = colors.HexColor("#1E2933")
MID = colors.HexColor("#546371")


def row_for(summary: pd.DataFrame, method: str, n: int, threads: int = 1) -> pd.Series:
    rows = summary[
        (summary["method"] == method)
        & (summary["n"] == n)
        & (summary["threads"] == threads)
        & (summary["vectorisation"] == "autovec_O3")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {method}, n={n}, p={threads}; got {len(rows)}")
    return rows.iloc[0]


def time_iqr(row: pd.Series) -> str:
    return f"{row.median_time:.4f} [{row.q1_time:.4f}, {row.q3_time:.4f}]"


def parse_metadata(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
    return values


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=23, leading=28, textColor=NAVY, alignment=TA_CENTER, spaceAfter=6 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontSize=11, leading=16,
            textColor=MID, alignment=TA_CENTER, spaceAfter=5 * mm,
        ),
        "abstract": ParagraphStyle(
            "Abstract", parent=base["BodyText"], fontSize=9, leading=13,
            alignment=TA_JUSTIFY, borderColor=colors.HexColor("#D7E2EC"),
            borderWidth=0.6, borderPadding=9, backColor=PALE, spaceAfter=5 * mm,
        ),
        "h1": ParagraphStyle(
            "Section", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=15, leading=19, textColor=NAVY, spaceBefore=5 * mm, spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Subsection", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=BLUE, spaceBefore=3 * mm,
            spaceAfter=1.5 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=8.8,
            leading=12.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=2.4 * mm,
            allowWidows=0, allowOrphans=0,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontSize=8, leading=10,
            textColor=MID, alignment=TA_LEFT, spaceBefore=1.2 * mm, spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontSize=7.3, leading=9.2,
            textColor=INK, alignment=TA_LEFT,
        ),
        "ref": ParagraphStyle(
            "Reference", parent=base["BodyText"], fontSize=7.7, leading=10.3,
            textColor=INK, leftIndent=5 * mm, firstLineIndent=-5 * mm, spaceAfter=1.8 * mm,
        ),
    }


def styled_table(data, widths, styles, header=True, font_size=7.5):
    cell_style = ParagraphStyle(
        "TableCell", fontName="Helvetica", fontSize=font_size,
        leading=font_size + 2, textColor=INK, alignment=TA_LEFT,
    )
    header_style = ParagraphStyle(
        "TableHeader", parent=cell_style, fontName="Helvetica-Bold", textColor=colors.white,
    )
    wrapped = [
        [Paragraph(str(value), header_style if header and row_idx == 0 else cell_style) for value in row]
        for row_idx, row in enumerate(data)
    ]
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6DEE6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        start = 1
    else:
        start = 0
    for r in range(start, len(data)):
        if (r - start) % 2 == 1:
            commands.append(("BACKGROUND", (0, r), (-1, r), PALE))
    table.setStyle(TableStyle(commands))
    return table


def add_figure(story, filename: str, caption: str, max_width: float, styles):
    path = FIGURES / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required report figure is missing: {path}")
    img = Image(str(path))
    ratio = min(max_width / img.imageWidth, 82 * mm / img.imageHeight)
    img.drawWidth *= ratio
    img.drawHeight *= ratio
    story.append(KeepTogether([img, Paragraph(caption, styles["caption"])]))


def page_chrome(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D6DEE6"))
    canvas.setLineWidth(0.5)
    canvas.line(17 * mm, height - 13 * mm, width - 17 * mm, height - 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID)
    canvas.drawString(17 * mm, height - 10 * mm, "ASSIGNMENT 3  |  PARALLEL MATRIX MULTIPLICATION")
    canvas.drawRightString(width - 17 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    if not RAW.is_file():
        raise FileNotFoundError(f"Raw benchmark CSV not found: {RAW}")

    raw = pd.read_csv(RAW)
    validate_parallel_dataframe(raw, str(RAW), allow_partial=False)
    summary = compute_summary_statistics(raw)
    styles = make_styles()
    story = []
    page_width = A4[0] - 34 * mm

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Assignment 3", styles["title"]))
    story.append(Paragraph("Thread Scaling in Shared-Memory Matrix Multiplication", styles["subtitle"]))
    story.append(Paragraph(
        "<b>Abel Castro Villar</b><br/>Data Science and Engineering<br/>"
        "Universidad de Las Palmas de Gran Canaria (ULPGC)<br/>"
        "Big Data | Academic year 2026-2027<br/>"
        "<link href='https://github.com/abelcasv/Assignment03_Abel' color='#2E6F9E'>"
        "github.com/abelcasv/Assignment03_Abel</link>", styles["subtitle"]
    ))

    primary2000 = row_for(summary, "omp_ikj", 2000, 12)
    seq2000 = row_for(summary, "seq_ikj", 2000)
    story.append(Paragraph(
        "<b>Abstract.</b> This study asks how OpenMP thread count affects the runtime and memory "
        "use of dense square matrix multiplication, and when parallel execution helps on one hybrid "
        "CPU. A C11 loop-interchanged kernel is compared with its sequential counterpart on identical "
        "binary64 inputs for six dimensions and eight requested team sizes. Each configuration has "
        "one warm-up and five measured repetitions; results are reported as medians and interquartile "
        f"ranges. For n=2000, median time falls from {seq2000.median_time:.3f} s sequentially to "
        f"{primary2000.median_time:.3f} s with 12 OpenMP threads ({primary2000.speedup_vs_seq:.2f}x "
        f"speedup; {primary2000.efficiency * 100:.1f}% efficiency). The OpenMP one-thread run is "
        "slower than the sequential baseline at this size. Scaling is not monotonic at every size, "
        "and a secondary blocked algorithm has a different thread-count profile. The evidence supports "
        "parallel execution for the larger tested matrices on this host, but does not identify individual "
        "core placement or establish a universal optimum.", styles["abstract"]
    ))

    story.append(Paragraph("1. Introduction", styles["h1"]))
    story.append(Paragraph(
        "Dense matrix multiplication is useful for studying shared-memory parallelism because its "
        "arithmetic work grows as O(n^3), while three dense n x n binary64 arrays require 24n^2 bytes. "
        "This report follows Assignment 3's requirement to compare equivalent sequential and parallel "
        "implementations, vary thread count, validate correctness, and report time and memory with "
        "enough detail to reproduce the measurements [1].", styles["body"]
    ))
    story.append(Paragraph("Question and hypotheses", styles["h2"]))
    story.append(Paragraph(
        "The research question is: how does OpenMP team size affect runtime and memory use for dense "
        "matrix multiplication, and at which tested matrix sizes does the parallel version outperform "
        "its sequential counterpart on this host? Before measurement, we expected small workloads to be "
        "sensitive to parallel-region overhead, larger workloads to benefit from multiple threads, and "
        "performance to stop improving once thread count exceeded the available execution contexts. "
        "These are predictions; the results test them and do not assume them to be true.", styles["body"]
    ))

    story.append(Paragraph("2. Methods", styles["h1"]))
    story.append(Paragraph("Implementations and inputs", styles["h2"]))
    story.append(Paragraph(
        "The primary sequential kernel and its OpenMP counterpart use row-major double arrays and the "
        "same i-k-j loop order. The OpenMP version statically distributes the outer i loop; each iteration "
        "writes a separate row of C, avoiding shared writes to the same output elements. A second blocked "
        "implementation uses 32 x 32 tiles, a common locality technique for matrix multiplication [2], "
        "and is compared only with its own sequential version. It is a secondary algorithm comparison, "
        "not a substitute baseline. Matrix input is shared by every method at a given size.", styles["body"]
    ))
    story.append(Paragraph(
        "The generator uses Python random.Random(42), draws each element using random() in [0,1), and "
        "writes six decimal places to CSV. Rounding can produce stored values of 1.000000; the observed "
        "CSV range is [0,1]. Values are read as IEEE 754 binary64. Dimensions are "
        "n in {250, 500, 750, 1000, 1500, 2000}. Python documents this generator as using Mersenne "
        "Twister; reproducibility is tied to the included generator and saved stream, not to matching "
        "random sequences in other languages [3]. The script writes data/SHA256SUMS.txt for the 12 dense "
        "inputs and five correctness fixtures. The dense files were regenerated and verified to have "
        "matching SHA-256 hashes.", styles["body"]
    ))
    story.append(Paragraph("Host and execution configuration", styles["h2"]))
    host = parse_metadata(ROOT / "results" / "environment_host.txt")
    openmp = parse_metadata(ROOT / "results" / "environment_info.txt")
    analysis = parse_metadata(ROOT / "results" / "analysis_environment.txt")
    host_data = [
        ["Item", "Recorded configuration"],
        ["CPU", "Intel Core i5-1335U; 10 physical cores (2 P-cores + 8 E-cores); 12 logical processors [4]"],
        ["Memory / OS", f"{host.get('physical_memory_total_gib', '15.63')} GiB physical RAM; Windows build {host.get('windows_build', '26300')}"],
        ["Compiler / flags", "MSYS2 UCRT64 GCC 16.1.0; -Wall -Wextra -O3 -std=c11 -fopenmp -march=native"],
        ["OpenMP / analysis", f"API 5.2 (_OPENMP={openmp.get('openmp_spec_version', '202111')}) [6]; Python {analysis.get('analysis_python', '3.12.14')}; NumPy 2.5.4; pandas 3.0.1; Matplotlib 3.11.2"],
    ]
    story.append(styled_table(host_data, [34 * mm, page_width - 34 * mm], styles, font_size=7.3))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "Requested team sizes were p in {1, 2, 4, 8, 10, 12, 16, 24}; every recorded actual team size "
        "matched its request. Dynamic adjustment was disabled, OMP_PROC_BIND=false and OMP_PLACES was "
        "unset. No explicit affinity was used, so worker placement on P- or E-cores is unknown. Requests "
        "above 12 are labelled oversubscribed relative to the host's logical processor count.", styles["body"]
    ))
    story.append(Paragraph("Timing, memory and correctness", styles["h2"]))
    story.append(Paragraph(
        "Inputs are loaded and arrays allocated outside timing. A monotonic Windows QueryPerformanceCounter "
        "timer surrounds one kernel call. It includes zero-initialisation of C and, for OpenMP, team and "
        "loop overhead; it excludes file input, allocation, and a checksum over every output element. The "
        "checksum is consumed through a volatile sink. One warm-up is excluded; five measured repetitions "
        "are retained per configuration. The reported centre is the median; quartiles are NumPy's 25th and "
        "75th percentiles, and IQR is Q3-Q1.", styles["body"]
    ))
    story.append(Paragraph(
        "Known-answer, identity and zero 4 x 4 fixtures are checked, followed by comparison of every kernel "
        "and tested OpenMP size against a trusted n=250 reference. A result passes when the absolute "
        "difference is at most 1e-4 + 1e-4 times the reference magnitude. The complete benchmark repeated these checks successfully. A per-repetition "
        "120 s stop threshold and a three-array structural-storage budget of 2048 MiB were set before "
        "the full run. The time threshold is checked after a kernel call returns; if exceeded, that sample "
        "would be retained and the remaining repetitions skipped. The largest size tested was n=2000; "
        "larger sizes were not attempted. No call exceeded the time threshold or the memory budget. "
        "The memory budget estimates matrix-array storage, not total "
        "process memory.", styles["body"]
    ))
    story.append(Paragraph(
        "Windows process peak working set is recorded through GetProcessMemoryInfo [5]. Because all "
        "configurations run in one process and this is a high-water mark, CSV values are not independent "
        "per-configuration peaks. The report gives the overall observed process peak and per-size "
        "structural estimates separately.", styles["body"]
    ))

    story.append(PageBreak())
    story.append(Paragraph("3. Results", styles["h1"]))
    story.append(Paragraph(
        f"The validated dataset contains {len(raw)} observations: {raw.groupby(['method','algorithm','parallel_model','vectorisation','n','threads']).ngroups} "
        "configurations with five repetitions each. It covers six sizes, two algorithms (loop-interchanged "
        "and blocked), and eight OpenMP team sizes, with a matched sequential baseline per algorithm and "
        "size. The analysis checks schema, full grid, repetitions, duplicates, finite positive times, team "
        "sizes and oversubscription labels before generating summaries.", styles["body"]
    ))
    story.append(Paragraph("Primary thread-scaling comparison", styles["h2"]))
    size_data = [["n", "Sequential ikj median [Q1, Q3] (s)", "OpenMP p=12 median [Q1, Q3] (s)", "S(12)", "E(12)"]]
    for n in [500, 1000, 1500, 2000]:
        seq = row_for(summary, "seq_ikj", n)
        par = row_for(summary, "omp_ikj", n, 12)
        size_data.append([str(n), time_iqr(seq), time_iqr(par), f"{par.speedup_vs_seq:.2f}x", f"{par.efficiency*100:.1f}%"])
    story.append(styled_table(size_data, [13 * mm, 48 * mm, 48 * mm, 17 * mm, 17 * mm], styles, font_size=7))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "At n=2000, the median falls from 15.259 s sequentially to 1.377 s with 12 OpenMP threads, "
        "a matched speedup of 11.08x. At n=500 the speedup is 3.77x; at n=1500 it is 3.89x. Results "
        "do not increase smoothly with size: the sequential n=1000 sample has a broad IQR, and the "
        "12-thread speedup at n=1500 is lower than at n=1000 or n=2000. Results are therefore compared "
        "per size rather than summarized as one scaling trend.", styles["body"]
    ))
    add_figure(story, "parallel_runtime_scaling.png",
               "Figure 1. Runtime medians and IQR by matrix size for selected sequential and OpenMP configurations. The vertical axis is logarithmic.",
               page_width, styles)
    story.append(Paragraph("OpenMP one-thread overhead", styles["h2"]))
    overhead_rows = [["n", "OpenMP p=1 difference from sequential median"]]
    for n in [250, 500, 750, 1000, 1500, 2000]:
        row = row_for(summary, "omp_ikj", n, 1)
        overhead_rows.append([str(n), f"{row.omp_p1_overhead_pct:+.1f}%"])
    story.append(styled_table(overhead_rows, [30 * mm, 62 * mm], styles, font_size=7.5))
    story.append(Paragraph(
        "The direction of the one-thread difference varies. It is 0.9% slower at n=250, 7.5% slower at "
        "n=500, 12.8% slower at n=750, and 100.5% slower at n=1500; it is 19.8% and 7.3% faster in "
        "the measured n=1000 and n=2000 samples. This is not a fixed runtime penalty; variability and "
        "system state affect the observed comparison.", styles["body"]
    ))

    story.append(PageBreak())
    story.append(Paragraph("Thread scaling at n=2000", styles["h2"]))
    seq = row_for(summary, "seq_ikj", 2000)
    thread_data = [["p", "OpenMP median [Q1, Q3] (s)", "S(p)", "E(p)"]]
    for p in [1, 2, 4, 8, 10, 12, 16, 24]:
        row = row_for(summary, "omp_ikj", 2000, p)
        thread_data.append([str(p), time_iqr(row), f"{row.speedup_vs_seq:.2f}x", f"{row.efficiency*100:.1f}%"])
    story.append(styled_table(thread_data, [16 * mm, 71 * mm, 21 * mm, 24 * mm], styles, font_size=7.5))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "At n=2000, speedup rises from 0.93x at one OpenMP thread to 11.08x at p=12, then falls to "
        "6.62x at p=16 and 7.07x at p=24. The 12-thread configuration has the best observed primary-kernel "
        "median. The oversubscribed results are slower than p=12, but p=24 is slightly faster than p=16; "
        "the data do not show a strictly monotonic oversubscription penalty.", styles["body"]
    ))
    story.append(Paragraph(
        "The p=2 and p=4 medians exceed ideal linear efficiency. Team sizes were verified; inputs and "
        "timing boundaries were matched; all samples were retained; and correctness passed. The p=2 IQR "
        "is wide, and no hardware counters, frequency logs or affinity were recorded. Cache, frequency "
        "or scheduling effects are plausible explanations but are not demonstrated. Superlinear values "
        "are observations that warrant further controlled investigation, not proof of a specific cause.", styles["body"]
    ))
    add_figure(story, "parallel_speedup_curves.png",
               "Figure 2. Matched-baseline speedup for the primary kernel. Vertical markers indicate hardware counts, not worker placement.",
               page_width, styles)
    add_figure(story, "parallel_efficiency_curves.png",
               "Figure 3. Parallel efficiency E(p)=S(p)/p. Values over 100% remain visible; shading marks requests above 12 logical processors.",
               page_width, styles)

    story.append(Paragraph("Secondary blocked-kernel comparison and memory", styles["h2"]))
    tiled_seq = row_for(summary, "seq_tiled_b32", 2000)
    tiled_data = [["Configuration", "Median [Q1, Q3] (s)", "Matched speedup", "Efficiency"]]
    tiled_data.append(["Sequential tiled B=32", time_iqr(tiled_seq), "1.00x", "100.0%"])
    for p in [8, 10, 12, 16, 24]:
        row = row_for(summary, "omp_tiled_b32", 2000, p)
        tiled_data.append([f"OpenMP tiled p={p}", time_iqr(row), f"{row.speedup_vs_seq:.2f}x", f"{row.efficiency*100:.1f}%"])
    story.append(styled_table(tiled_data, [42 * mm, 63 * mm, 24 * mm, 26 * mm], styles, font_size=7.2))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "At n=2000, sequential tiled multiplication has a median of 4.7202 s, compared with 15.2589 s "
        "for sequential ikj. This is an algorithm difference, not a pure thread-count effect. The blocked "
        "OpenMP kernel reaches 0.6026 s at p=8 and 0.5826 s at p=24. The latter is the fastest observed "
        "blocked median despite oversubscription; p=16 is slower. The small p=8/p=24 difference should "
        "be interpreted cautiously and does not support a general claim that oversubscription helps.", styles["body"]
    ))
    mem_data = [["n", "Three-array structural estimate (MiB)"]]
    for n in [250, 500, 750, 1000, 1500, 2000]:
        mem_data.append([str(n), f"{row_for(summary, 'seq_ikj', n).structural_memory_mib:.2f}"])
    mem_data.append(["Overall process peak working set", f"{raw.peak_memory_mib.max():.2f} MiB"])
    story.append(styled_table(mem_data, [63 * mm, 48 * mm], styles, font_size=7.5))
    story.append(Paragraph(
        "At n=2000, three arrays require an estimated 91.55 MiB; the observed process high-water working "
        "set across the full run is 97.36 MiB. The process metric also includes runtime and other resident "
        "pages, and is cumulative; it cannot be treated as an independent peak for every row. No size "
        "exceeded the stated structural budget.", styles["body"]
    ))

    story.append(Paragraph("4. Discussion and limitations", styles["h1"]))
    story.append(Paragraph(
        "The required matched comparison answers the research question within the tested scope: at "
        "n=2000, 12-thread OpenMP is substantially faster than sequential ikj, while OpenMP with one "
        "thread is not faster. Smaller-size results vary. The smallest kernels have short durations, "
        "and the n=1000 sequential baseline has a broad IQR, so a thread count should not be selected "
        "from one measurement alone.", styles["body"]
    ))
    story.append(Paragraph(
        "Thread scaling is not monotonic across every configuration. At n=2000, the primary kernel is "
        "fastest at p=12, with slower medians at 16 and 24; in the blocked extension the fastest observed "
        "median is at 24. These results do not isolate scheduling, thermal state, CPU frequency, or P/E-core "
        "placement. A controlled follow-up with recorded affinity and hardware performance counters is "
        "needed for causal explanation.", styles["body"]
    ))
    story.append(Paragraph(
        "The study uses five repetitions and one full run on a single Windows host. Process peak working "
        "set is not an isolated per-configuration metric. Inputs are rounded to six decimal places; "
        "outputs use explicit absolute and relative tolerances. The tested range ends at n=2000; no claim "
        "is made for larger matrices or other processors. SIMD is optional under the assignment and is "
        "excluded from the final core results; no SIMD conclusion is drawn.", styles["body"]
    ))
    story.append(Paragraph("5. Conclusion", styles["h1"]))
    story.append(Paragraph(
        f"The main comparison uses identical inputs, data representation and loop order for sequential "
        f"and OpenMP kernels, validates correctness, and records five repetitions for each of {len(summary)} "
        f"configurations. At n=2000, 12 OpenMP threads reduce the median primary-kernel time from "
        f"{seq2000.median_time:.3f} s to {primary2000.median_time:.3f} s "
        f"({primary2000.speedup_vs_seq:.2f}x speedup; {primary2000.efficiency*100:.1f}% efficiency). "
        "The one-thread and oversubscribed results show that overhead and scaling depend on the "
        "configuration. These conclusions apply to the measured implementation and host only. Raw data, "
        "the deterministic generator, analysis script, environment records, exact commands, report source "
        "and this reproducible PDF are included in the repository.", styles["body"]
    ))
    story.append(Paragraph("AI assistance declaration", styles["h2"]))
    story.append(Paragraph(
        "Gemini was used to draft the initial CLAUDE.md project guidance and prompt plan. Claude was used "
        "to carry out implementation and documentation work through those prompts. OpenAI Codex was used "
        "for an independent audit and subsequent corrections to the benchmark, analysis, reproducibility "
        "materials and report. AI-generated material is not evidence; the student remains responsible "
        "for understanding and defending the code, measurements, interpretation and references.", styles["body"]
    ))

    story.append(Paragraph("References", styles["h2"]))
    references = [
        "[1] Big Data Course Team, <i>BIG DATA | Individual Assignments | Revised proposal 2026-2027</i>, Assignment 3 and common protocol, course document, 2026.",
        "[2] K. Goto and R. A. van de Geijn, 'Anatomy of high-performance matrix multiplication,' <i>ACM Transactions on Mathematical Software</i>, vol. 34, no. 3, Article 12, 2008. <link href='https://doi.org/10.1145/1356052.1356053' color='#2E6F9E'>doi:10.1145/1356052.1356053</link>.",
        "[3] Python Software Foundation, 'random - Generate pseudo-random numbers,' Python 3 documentation. <link href='https://docs.python.org/3/library/random.html' color='#2E6F9E'>docs.python.org/3/library/random.html</link>. Accessed Nov. 1, 2026.",
        "[4] Intel Corporation, 'Intel Core i5-1335U Processor Specifications.' <link href='https://www.intel.com/content/www/us/en/products/sku/232153/intel-core-i51335u-processor-12m-cache-up-to-4-60-ghz/specifications.html' color='#2E6F9E'>Intel product specification</link>. Accessed Nov. 1, 2026.",
        "[5] Microsoft, 'Process Memory Usage Information,' Windows API documentation. <link href='https://learn.microsoft.com/en-us/windows/win32/psapi/process-memory-usage-information' color='#2E6F9E'>Microsoft Learn</link>. Accessed Nov. 1, 2026.",
        "[6] OpenMP Architecture Review Board, <i>OpenMP Application Programming Interface Version 5.2</i>, Nov. 2021. <link href='https://www.openmp.org/spec-html/5.2/openmp.html' color='#2E6F9E'>OpenMP specification</link>.",
    ]
    for ref in references:
        story.append(Paragraph(ref, styles["ref"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
        topMargin=19 * mm, bottomMargin=17 * mm,
        title="Assignment 3: Thread Scaling in Shared-Memory Matrix Multiplication",
        author="Abel Castro Villar", subject="Big Data individual assignment",
    )
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    print(f"Built {OUT}")
    print(f"Validated and embedded {len(raw)} raw observations across {len(summary)} configurations.")


if __name__ == "__main__":
    build()
