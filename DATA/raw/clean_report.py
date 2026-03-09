import pandas as pd
import numpy as np

report = pd.read_csv("Report_meditazione.csv")
report = report[2::]
report = report["Report"].dropna().reset_index(drop=True)
save_path = "Report_meditazione_cleaned.csv"
report.to_csv(save_path, index=False)
