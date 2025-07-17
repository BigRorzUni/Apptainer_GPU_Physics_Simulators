import os
import pandas as pd
from pathlib import Path
from functools import reduce

BASE_DIR = Path("/home/david/test/pendulum_speed_test/data")
OUTPUT_DIR = BASE_DIR / "combined"
OUTPUT_DIR.mkdir(exist_ok=True)

file_types = ["speed", "env_fps", "total_fps"]

def merge_dfs_on_steps(dfs):
    if not dfs:
        return None
    return reduce(lambda left, right: pd.merge(left, right, on='steps', how='outer'), dfs)

for ftype in file_types:
    dfs = []

    for sim_dir in BASE_DIR.iterdir():
        if not sim_dir.is_dir():
            continue
        sim_name = sim_dir.name

        for file in sim_dir.glob(f"*{ftype}.csv"):
            fname = file.name

            try:
                batch_size = int(fname.split("_")[0])
            except ValueError:
                batch_size = None  # Mujoco or unbatched

            df = pd.read_csv(file)

            # Identify the metric column (assuming there's only one metric column apart from steps)
            metric_cols = [c for c in df.columns if c != "steps"]
            if len(metric_cols) != 1:
                raise ValueError(f"Expected exactly one metric column, got {metric_cols} in {file}")

            metric_col = metric_cols[0]

            # Rename metric column to "Simulator - Batch size XXX" or "Simulator" if no batch
            if batch_size is None:
                new_col_name = f"{sim_name}"
            else:
                new_col_name = f"{sim_name} - Batch Size {batch_size}"

            df_renamed = df[["steps", metric_col]].rename(columns={metric_col: new_col_name})

            dfs.append(df_renamed)

    combined_df = merge_dfs_on_steps(dfs)
    if combined_df is not None:
        combined_df = combined_df.sort_values("steps")
        output_path = OUTPUT_DIR / f"sim_name_{ftype}.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Wrote combined CSV: {output_path}")
    else:
        print(f"No data found for metric {ftype}")
