# Averaging in gene perturbation benchmarks hides perturbation-strength variation: Consequences for model evaluation

Code and analysis for our study of **realized perturbation strength as a hidden source of heterogeneity in single-cell perturbation benchmarks**.

Most perturbation benchmarks treat all cells assigned the same nominal intervention as one condition and score models against an averaged response. This repository contains the analyses used to test what that averaging hides.

The central distinction is between two levels of perturbation information:

1. **Perturbation identity** — which intervention occurred?
2. **Within-perturbation realized strength** — how strongly was that intervention realized?

A representation can preserve or collapse either level independently.

## What this repository contains

The analysis is organized around four questions:

- **Does guide abundance track realized knockdown?**  
  We validate guide UMI abundance as a proxy for realized perturbation strength.

- **Is there reproducible biological structure within one nominal perturbation?**  
  We split cells within each perturbation by realized dose and measure perturbation-specific split-half reproducibility using matched controls and dose-shuffled nulls.

- **Do model outputs retain perturbation identity and realized-strength structure?**  
  We evaluate PCA, Cell2Sentence (C2S), and STATE.

- **Can perturbation identity be decoded from the response representation?**  
  Retrieval diagnostics compare observed responses with model predictions.

HCT116 provides the primary dose-resolved analysis. HEK293T is used as an independent stress test of the dose proxy and evaluation design.

## Repository structure

```text
.
├── config/                 # frozen evaluation sets and selection/evaluation splits
├── data_prep/              # dataset preparation, screening, and dose validation
├── dose_analysis/          # dose binning, reproducibility, and robustness analyses
├── models/
│   ├── c2s/                # Cell2Sentence evaluation
│   ├── diagnostics/        # perturbation-identity retrieval
│   ├── pca/                # PCA reference
│   └── state/              # project-specific STATE drivers and adapters
├── figures/                # scripts used to generate the main figures
├── results/                # compact outputs underlying reported analyses
├── third_party/
│   └── state/              # upstream STATE implementation used in our experiments
├── utils/
├── pyproject.toml
└── uv.lock
```

Large raw datasets, checkpoints, inference caches, Slurm logs, and other machine-specific intermediate files are intentionally not stored in Git.

## Analysis map

| Scientific component | Main code |
|---|---|
| Dose-proxy validation | `data_prep/validate_dose_proxy.py` |
| Perturbation screening | `data_prep/nsr_rho_screen.py` |
| Frozen evaluation set | `data_prep/build_eval_set_v2.py` |
| Selection/evaluation split | `data_prep/make_split.py` |
| Dose binning | `dose_analysis/dose_binning.py` |
| Dose-resolved reproducibility | `dose_analysis/dose_reproducibility.py` |
| Robustness analyses | `dose_analysis/robustness.py` |
| PCA reference | `models/pca/pca.py` |
| C2S evaluation | `models/c2s/c2s_zero.py` |
| STATE evaluation | `models/state/` |
| Perturbation-identity retrieval | `models/diagnostics/` |
| Main figures | `figures/` |

## Core analysis design

For a perturbation \(p\), the pooled benchmark target can be viewed as a mixture over realized dose \(d\):

\[
P(Y \mid p) = \int P(Y \mid p,d)P(d \mid p)\,dd.
\]

The dose-resolved analysis asks whether \(P(Y \mid p,d)\) contains reproducible structure that is lost when cells are pooled only by nominal perturbation.

`dose_analysis/dose_reproducibility.py`:

- bins cells within each perturbation by realized dose;
- matches negative controls to the cell-state composition of each bin;
- estimates log-fold-change profiles on independent cell halves;
- projects out a generic perturbation direction;
- measures perturbation-specific split-half reproducibility; and
- compares the observed dose ordering against a within-perturbation dose-shuffled null.

The frozen evaluation-set and split definitions used by the downstream analyses are stored in `config/`.

## Model diagnostics

The model analyses distinguish two different kinds of representational collapse.

### Cell2Sentence and STATE

Both models can produce cell-dependent predictions. The relevant question here is whether the resulting response representation retains enough **perturbation-specific information** for the requested intervention to remain recoverable.

The retrieval diagnostic therefore asks an operational information-retention question:

> Given a response representation, how much more identifiable is the perturbation that produced it than it would be by chance?

This is evaluated directly as a retrieval problem rather than interpreted as an estimate of mutual information.

### PCA reference

The PCA reference learns a perturbation-specific shift from selection-half observations pooled across realized dose and then applies the same perturbation template to both dose bins.

It therefore provides a useful complementary case: perturbation identity is retained by construction, while within-perturbation realized-strength structure is collapsed into a single template.

## Results

Compact result files used for the reported analyses are stored under `results/`, grouped by analysis:

```text
results/
├── dose_validation/
├── dose_reproducibility/
├── identity_retrieval/
├── models/
└── screening/
```

These are intended to make the reported statistics auditable without requiring the full raw expression matrices or model checkpoints.

## Data

Raw Perturb-seq data are not included in this repository.

The main analyses expect:

- an expression matrix in `.h5ad` format;
- per-cell guide calls containing perturbation identity and guide UMI counts;
- a frozen evaluation-set definition; and
- a frozen selection/evaluation split.

Dataset paths are supplied at runtime. For script-specific arguments, use `--help`, for example:

```bash
python data_prep/validate_dose_proxy.py --help
python dose_analysis/dose_reproducibility.py --help
python models/pca/pca.py --help
```

## Environment

The Python environment is specified by `pyproject.toml` and `uv.lock`.

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

Some model-specific analyses additionally require the corresponding public model checkpoints.

## STATE

`models/state/` contains the project-specific STATE evaluation and adaptation code.

`third_party/state/` contains the upstream STATE implementation used by these experiments. It is kept separate so that upstream model code is clearly distinguished from code written for this study.

For scripts importing the vendored STATE package:

```bash
export PYTHONPATH="$PWD/third_party:$PYTHONPATH"
```

See `MODEL_LICENSE.md` and `MODEL_ACCEPTABLE_USE_POLICY.md` for the applicable upstream terms.

## Typical workflow

```text
raw expression + guide calls
            │
            ▼
    validate dose proxy
            │
            ▼
 screen / freeze evaluation set
            │
            ▼
 freeze selection/evaluation split
            │
       ┌────┴─────────────┐
       ▼                  ▼
dose-resolved         model evaluation
biology               PCA / C2S / STATE
       │                  │
       └────────┬─────────┘
                ▼
      identity / dose diagnostics
                │
                ▼
             figures
```

The exact commands depend on local dataset and checkpoint paths, but all executable analysis scripts expose their arguments through `--help`.

## Interpretation

Perturbation responses contain nested structure: variation **between perturbations** and variation in **realized strength within a perturbation**.

A flat dose-resolved model score can therefore arise for different reasons. A representation may fail to preserve which perturbation occurred, or it may preserve perturbation identity while averaging away within-perturbation realized strength. The analyses in this repository are designed to distinguish those cases rather than treating them as a single failure mode.

## Citation

If you use this code, please cite the accompanying manuscript:

> *To average or not to average: realized perturbation dose is a hidden axis in gene perturbation benchmarking.*

Preprint citation information will be added here when available.

## License

See `LICENSE` for code released with this repository.

Third-party model code and checkpoints remain subject to their respective upstream licenses and acceptable-use terms.
