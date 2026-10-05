# TSA exam materials

Exam materials of *Serii de timp / Time Series Analysis* (bachelor, year 3, CSIE, ASE).
All documents compile with XeLaTeX (system fonts Helvetica Neue and Menlo) and use the shared preamble
`tsa_exam_preamble.tex` (ASE and IDA logos, automatic numbering of problems and sub-tasks, the `\ifrez` switch for
solutions, `\softout{...}` for software outputs).

The exam keeps the format of the earlier TSA exams: interpretation of software output (now Python: statsmodels, arch,
scikit-learn), short derivations and model choice. Every chart and every software output is regenerated from the
course data.

| Path | Content | In git |
|---|---|---|
| `tsa_exam_preamble.tex` | Shared article preamble (RO/EN through `\def\examlang{ro\|en}`) | yes |
| `exam_common.py` | Helpers: course style, RO/EN tick formats, `write_out` (software output as text), `correlogram`, `sm_tables` | yes |
| `make_figs.py` | Runs every chart and output generator (practice set and bank) from `Quantlets/common/tsa_data.py` and `tsa_style.py` | yes |
| `build_variant.py` | Assembles an exam variant from the bank and compiles it | yes |
| `practice/` | Public practice set without solutions (numerical answers only): `probleme_examen_ro.pdf`, `exam_problems_en.pdf`, `figs_practice.py`, `figs/`, `out/` | yes |
| `bank/` | Instructor-only problem bank: `ro/chNN.tex`, `en/chNN.tex`, `figs_chNN.py`, `figs/`, `out/`, `bank_numbers.json`, `bank_*.pdf` | **no** |
| `variants/` | Assembled variants, problems and solutions | **no** |

Solutions are never committed. `exam/.gitignore` ignores `bank/`, `variants/`, `archive/`, copies of the earlier
exam papers and every `*_rezolvare.*`, `*_Rezolvare.*`, `*_solutions.*` and `*_barem.*` file under `exam/`. Keep the
bank and the solution PDFs locally (or in the instructor folder outside the repository).

## Data

Market series are read from `data/market` (last day 18 September 2026); macro series (Eurostat, FRED) and the
statsmodels data sets are read online by `tsa_data`. Official statistics are revised, so a later run can change the
last decimals of a macro number: `make_figs.py` saves every quoted number in `bank/bank_numbers.json` and
`practice/practice_numbers.json`; compare them with the statements after regenerating.

## Building an exam variant

```bash
python3 exam/make_figs.py                                        # charts and outputs (practice set and bank)
python3 exam/make_figs.py --only bank --ch 2,5                   # bank chapters 2 and 5 only
python3 exam/build_variant.py --list                             # problems in the bank
python3 exam/build_variant.py --name V1 --seed 2027 --n 9        # 9 problems from 9 different chapters, RO
python3 exam/build_variant.py --name V1 --seed 2027 --n 9 --lang both --date "iunie 2027"
python3 exam/build_variant.py --name R1 --ids ch01-p2,ch05-p1,ch10-p4 --lang both   # chosen problems
python3 exam/build_variant.py --name V2 --seed 11 --chapters 1-7                    # chapters 1..7 only
python3 exam/build_variant.py --bank --lang both                 # the whole bank with solutions
```

- The same seed always gives the same variant; RO and EN use the same problem ids.
- Each bank problem is worth 1 point (sub-tasks with their points, step-by-step solution and grading scheme).
  With `--n 9` a variant has 9 points + 1 point by default = 10 points; the default draw covers chapters 0--10.
- Output: `exam/variants/<name>_ro.pdf` and `<name>_ro_rezolvare.pdf`, `<name>_en.pdf` and
  `<name>_en_solutions.pdf`. Each document is compiled twice; the script reports errors and overfull boxes.

## Problem format in the bank

```latex
%<problem id=ch05-p2 pts=1>
\problem[ch05-p2]{Title}{1p}
Context with the data.
\softout{bank/out/ch05_garch.txt}          % software output written by bank/figs_ch05.py
\Q First sub-task (one question). \pts{0,5p}
\Q Second sub-task (one question). \pts{0,5p}
\ifrez\Rez
\begin{enumerate} \item step 1 \item step 2 \end{enumerate}
\barem{0,25p ...; 0,25p ...; 0,5p ...}
\fi
%</problem>
```

A chapter module `exam/bank/figs_chNN.py` defines `make(fig_dir, out_dir)`: it writes the charts (`figs/*_ro.pdf`,
`figs/*_en.pdf`) and the software outputs (`out/*.txt`, at most 84 characters per line) and returns the numbers that
`make_figs.py` saves in `bank/bank_numbers.json`.

## Practice set

`practice/probleme_examen_ro.tex` and `practice/exam_problems_en.tex`: problems from chapters 1--10, different from
the bank problems (other data series), with numerical answers at the end. Recompile after
`python3 exam/make_figs.py --only practice`.
Site links (chapter 15): `https://danpele.github.io/Time-Series-Analysis/exam/practice/probleme_examen_ro.pdf` and
`https://danpele.github.io/Time-Series-Analysis/exam/practice/exam_problems_en.pdf`.
