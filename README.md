# Statistical Signal Processing

Study notes, step-by-step solutions and Python checks for the course
**Statistical Signal Processing** (L.048.24014, Universität Paderborn, Signal and System Theory Group, Dr.-Ing. Tanuj Hasija), WiSe 2026/27.

The notes are written for someone learning the material from scratch: every concept needed for the tutorial problems is built up and derived, and no solution steps are skipped.

## Repository layout

```
Statistical_Signal_Processing/
├── README.md                                    ← you are here
├── course-material/                             ← put your local copies of the course PDFs here (not published)
├── Week-01_Introduction-and-Probability-Basics/
│   ├── README.md                                ← overview and progress of this week
│   ├── 01_Concepts.md                           ← Lecture 1 + probability basics for Tutorial 1
│   ├── 02_Solutions.md                          ← Tutorial 1, step by step
│   ├── pdf/                                     ← PDF versions of the notes
│   └── python/                                  ← script that verifies every result, and figures
├── Week-02_Random-Variables-and-Joint-Distributions/
│   ├── README.md
│   ├── 01_Concepts.md                           ← Lecture 2: joint/continuous/Gaussian RVs, random vectors
│   ├── 02_Solutions.md                          ← Tutorial 2, step by step
│   ├── pdf/
│   └── python/
└── tools/
    └── build_pdfs.sh                            ← regenerate the PDFs from the Markdown
```

## Weeks

| Week | Lecture | Tutorial | Concepts | Solutions | Code | Status |
|---|---|---|---|---|---|---|
| 1 | L1 Introduction (+ L2 §2.1–2.2 basics) | T1: Monty Hall, Poisson, Binomial, channel decoder | [md](Week-01_Introduction-and-Probability-Basics/01_Concepts.md) · [pdf](Week-01_Introduction-and-Probability-Basics/pdf/01_Concepts.pdf) | [md](Week-01_Introduction-and-Probability-Basics/02_Solutions.md) · [pdf](Week-01_Introduction-and-Probability-Basics/pdf/02_Solutions.pdf) | [python](Week-01_Introduction-and-Probability-Basics/python) | Problems 1–5 ✅ |
| 2 | L2 Review of probability theory | T2: two discrete RVs, bivariate pdfs | [md](Week-02_Random-Variables-and-Joint-Distributions/01_Concepts.md) · [pdf](Week-02_Random-Variables-and-Joint-Distributions/pdf/01_Concepts.pdf) | [md](Week-02_Random-Variables-and-Joint-Distributions/02_Solutions.md) · [pdf](Week-02_Random-Variables-and-Joint-Distributions/pdf/02_Solutions.pdf) | [python](Week-02_Random-Variables-and-Joint-Distributions/python) | Problems 1–3 ✅ |

## How to use

1. Read the **Concepts** file for a week first. It contains all the background needed for that week's tutorial, plus a cheat sheet at the end (useful for the one handwritten A4 formula sheet allowed in the exam).
2. Try the problems yourself, then compare with the **Solutions** file.
3. Run the **Python** scripts to check the numbers:
   ```bash
   pip install numpy scipy matplotlib sympy
   python Week-01_Introduction-and-Probability-Basics/python/tutorial1_verify.py
   python Week-02_Random-Variables-and-Joint-Distributions/python/tutorial2_verify.py
   ```

![Uniform pdf on a triangle with conditional slices](Week-02_Random-Variables-and-Joint-Distributions/python/figures/p3_triangle.png)

The Markdown files use LaTeX math, which GitHub renders directly. The PDFs can be rebuilt with `bash tools/build_pdfs.sh` (needs `pandoc` and `xelatex`).
