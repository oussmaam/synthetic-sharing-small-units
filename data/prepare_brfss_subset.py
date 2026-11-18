"""Build brfss2015_subset.csv.gz, the analysis file of the real-site experiments (Experiment 2).

Source: Behavioral Risk Factor Surveillance System 2015, file LLCP2015.XPT (SAS transport format), from
https://www.cdc.gov/brfss/annual_data/annual_2015.html (download LLCP2015XPT.zip and unzip it here).
Only the columns used by the experiments are kept; all derivations (outcome, predictors, completeness,
BMI < 70) are done inside the experiment scripts. Expected result: 441,456 respondents.
Copy the output file next to the experiment scripts before running them.
"""
import pandas as pd

COLS = ["_STATE", "DIABETE3", "_RFHYPE5", "TOLDHI2", "_CHOLCHK", "_BMI5", "SMOKE100", "CVDSTRK3", "_MICHD",
        "_TOTINDA", "_FRTLT1", "_VEGLT1", "_RFDRHV5", "HLTHPLN1", "MEDCOST", "GENHLTH", "MENTHLTH", "PHYSHLTH",
        "DIFFWALK", "SEX", "_AGEG5YR", "EDUCA", "INCOME2", "_LLCPWT"]

d = pd.read_sas("LLCP2015.XPT", format="xport", encoding="latin-1")
d.columns = [c.strip() for c in d.columns]
d[COLS].to_csv("brfss2015_subset.csv.gz", index=False, compression="gzip")
print(f"{len(d)} respondents written to brfss2015_subset.csv.gz")
