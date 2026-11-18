"""Build cdc_diabetes_binary.csv, the analysis file of the diabetes data (Experiment 1).

Source: CDC Diabetes Health Indicators dataset (UCI Machine Learning Repository, id 891), derived from BRFSS 2015.
Records with a recorded BMI of 70 kg/m2 or more are removed (recording errors; Supplement S2).
Expected result: 253,680 records read, 592 removed, 253,088 kept, prevalence 0.139.
Copy the output file next to the experiment scripts before running them.
"""
from ucimlrepo import fetch_ucirepo

d = fetch_ucirepo(id=891)
df = d.data.features.copy()
df["y"] = d.data.targets["Diabetes_binary"].values
n0 = len(df)
df = df[df.BMI < 70].reset_index(drop=True)
df.to_csv("cdc_diabetes_binary.csv", index=False)
print(f"{n0} records read, {n0 - len(df)} removed (BMI >= 70), {len(df)} kept, prevalence {df.y.mean():.3f}")
