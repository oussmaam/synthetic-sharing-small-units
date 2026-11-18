# Sharing a risk model with a small clinical unit

Code and results for:

> Maamoun MO, Gregori D, Boucherit A, Berdjouh C. *Sharing a risk model with a small clinical unit:
> reweighting, transfer and donor-generated synthetic data compared on simulated and real sites.*
> Submitted to the Journal of Biomedical Informatics.

The study compares ways in which a large hospital (the *donor*) can help a small unit with 200 labelled
records (the *receiver*) without sending patient records: updated sharing of a score, reweighting,
untargeted and targeted synthetic data labelled by a teacher model, a COMMUTE-style transfer, and the
augmentation schemes of Brown et al.

## Contents

| Folder | What it holds |
|---|---|
| `data/` | Scripts that build the two analysis files from the public sources |
| `experiments/` | One script per experiment; each writes one results file |
| `results/` | The results files used in the paper (one row per strategy, scenario and repetition) |
| `analysis/` | Scripts that turn the results files into every table and figure |
| `notebooks/` | The notebooks of the development versions v11 to v16, kept as a record (Supplement S1) |

## Data

All data are public. Neither source file is redistributed here.

1. **US Natality 2023** (National Center for Health Statistics). The experiment scripts download
   `Nat2023us.zip` (about 230 MB) themselves when run with the `natality` option and keep a 2% random sample
   in `natality2023_sample.csv`. The `unzip` program must be installed (the archive uses Deflate64).
2. **CDC Diabetes Health Indicators** (UCI repository, id 891), derived from BRFSS 2015:
   `python data/prepare_cdc_diabetes.py` writes `cdc_diabetes_binary.csv` (253,088 records after removing
   592 records with a recorded BMI of 70 or more).
3. **BRFSS 2015** (CDC), for the real-site experiments: download `LLCP2015XPT.zip`, unzip it into `data/`
   and run `python data/prepare_brfss_subset.py`, which writes `brfss2015_subset.csv.gz` (441,456 respondents).

Copy the analysis files next to the experiment scripts before running them.

## Running the experiments

```bash
pip install -r requirements.txt
cd experiments

# Experiment 1 (simulation on the two datasets); add --quick for a 2-repetition test run
python exp1_semisynthetic.py natality            # Table 3, Figure 4 (with Brown et al.)
python exp1_semisynthetic.py cdc_diabetes
python exp1_registry.py natality                 # registry runs (Supplement S5)
python exp1_registry.py cdc_diabetes
python exp1_commute.py natality                  # COMMUTE-style transfer (Table 3, footnote)
python exp1_commute.py cdc_diabetes
python exp1_support_softlabels_semisynthetic.py natality      # lack of support, soft labels (Figure 6, S6)
python exp1_support_softlabels_registry.py natality

# Experiment 2 (real sites, BRFSS 2015)
python exp2_sites_rule_R1.py full                # Kansas and Nebraska as donors, rule R1 (Figure 7)
python exp2_sites_rule_R2.py full                # Minnesota and Washington as donors, rule R2 (Supplement S4)

# Analyses added after review
python review_variants.py age_nl cdc_diabetes    # non-linearity on age (Supplement S8)
python review_variants.py concept cdc_diabetes   # change in the effect of BMI
python review_weight_sites.py full               # weight of the synthetic records on real sites (S7)
python dca_simulation.py cdc_diabetes            # decision curves (Supplement S9)
python dca_sites.py full
```

Without `full`, the real-site scripts run a short test on six receivers. Every script resumes from its
results file if it is interrupted.

Run times on one CPU core are long: about 20 seconds per scenario and repetition for the simulation and
one to four minutes per donor-receiver pair on real sites. The full set takes several days.

## Reproducing the tables and figures

```bash
cd analysis
python tables.py            # all numbers in the tables of the paper and the Supplement
python fig_simulation.py    # Figure 4
python fig_support.py       # Figure 6
python fig_states.py        # Figure 7
python decision_curves.py   # Supplementary Figure S1
```

These scripts read only `results/` and run in seconds.

## Reproducibility

Random numbers are drawn from seeds fixed per scenario and repetition (base seed 2026), so a rerun gives
the same numbers. The decision-curve runs, which repeat the main pipeline with additional outputs, matched
the published log-loss and AUC exactly in every completed cell (1,898 simulation rows and 729 real-site
rows). The natality results in `results/` were produced from the notebooks in `notebooks/` with the
same code as the scripts.

## Development history

The design was reached over versions v2 to v18. Supplement S1 lists what each version added and found.
The scripts here correspond to the final versions (v11a, v15a/b, v16a/b, v17, v18) and to the analyses
added after review.

## Licence

Code: MIT (see `LICENSE`). The result files in `results/` are derived from public data and may be reused with citation of the paper.

## Contact

Mohamed Oussama Maamoun, LIAP, University of El Oued: maamoun.oussama3@gmail.com
