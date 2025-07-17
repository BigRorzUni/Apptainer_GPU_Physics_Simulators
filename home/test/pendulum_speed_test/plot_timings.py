import argparse
from pathlib import Path
import pandas as pd
from functools import reduce
import timing_helper

def load_and_merge(base_dir, simulators, file_type, batch_size=None):
    dfs = []
    base_dir = Path(base_dir)

    for sim in simulators:
        sim_dir = base_dir / sim
        if not sim_dir.exists():
            print(f"Simulator directory not found: {sim_dir}")
            continue
        
        if sim.lower() == "mujoco":
            # Mujoco doesn't have batch sizes in filenames
            file_path = sim_dir / f"{file_type}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Rename metric column to unique name (exclude 'steps')
                metric_cols = [c for c in df.columns if c != "steps"]
                if len(metric_cols) != 1:
                    raise ValueError(f"Expected exactly one metric column in {file_path}")
                metric_col = metric_cols[0]
                df = df.rename(columns={metric_col: sim})
                dfs.append(df)
            else:
                print(f"File not found: {file_path}")
        else:
            # For other sims, either load specific batch size or all batches
            if batch_size is not None:
                filenames = [f"{batch_size}_{file_type}.csv"]
            else:
                # load all matching files like "*_speed.csv"
                filenames = [f.name for f in sim_dir.glob(f"*_{file_type}.csv")]

            for fname in filenames:
                file_path = sim_dir / fname
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    metric_cols = [c for c in df.columns if c != "steps"]
                    if len(metric_cols) != 1:
                        raise ValueError(f"Expected exactly one metric column in {file_path}")
                    metric_col = metric_cols[0]
                    # Extract batch size from filename (assumed to be first part before _)
                    try:
                        bs = int(fname.split("_")[0])
                    except:
                        bs = "unknown"
                    col_name = f"{sim} - Batch Size {bs}"
                    df = df.rename(columns={metric_col: col_name})
                    dfs.append(df)
                else:
                    print(f"File not found: {file_path}")

    if not dfs:
        print("No files loaded, returning None")
        return None

    # Merge all DataFrames on 'steps'
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="steps", how="outer"), dfs)
    return merged_df

def get_plot_labels_and_title(file_type, simulators, batch_size=None):
    # Define y-axis label based on file type
    if file_type == "speed":
        y_col = "Time (s)"
        title_prefix = "Simulation Time"
    elif file_type == "env_fps":
        y_col = "FPS per Environment"
        title_prefix = "FPS per Environment"
    elif file_type == "total_fps":
        y_col = "Total FPS"
        title_prefix = "Total FPS"
    else:
        y_col = "Value"
        title_prefix = "Timing Data"

    # Build a human-readable simulator list string
    sims_str = ", ".join(simulators)

    # Batch size description
    batch_str = f"Batch Size {batch_size}" if batch_size else "All Batch Sizes"

    # Compose plot title
    title = f"{title_prefix} for {sims_str} ({batch_str})"

    return y_col, title


def main():
    parser = argparse.ArgumentParser(description="Load and merge timing CSVs from simulators")
    parser.add_argument("--simulators", nargs="+", default=None, help="Simulators to include (e.g. Genesis MJX Mujoco Newton)")
    parser.add_argument("--file_type", type=str, required=True, choices=["speed", "env_fps", "total_fps"], help="Type of timing data to load")
    parser.add_argument("--batch_size", type=int, default=None, help="Specify batch size (default: load all)")
    parser.add_argument("--base_dir", type=str, default="data", help="Base data directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    all_simulators = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])

    if not args.simulators:
        args.simulators = all_simulators
        print(f"No simulators specified, using all found: {args.simulators}")

    df = load_and_merge(args.base_dir, args.simulators, args.file_type, args.batch_size)

    if df is not None:
        print(df.head())
        
        # Check if all simulators selected (ignoring order)
        selected_sorted = sorted(args.simulators)
        if selected_sorted == all_simulators:
            # Use simple filename when all simulators included
            out_filename = f"plots/{args.file_type}.png"
        else:
            out_filename = f"plots/{'_'.join(args.simulators)}_{args.file_type}"
            if args.batch_size:
                out_filename += f"_{args.batch_size}"
            out_filename += ".png"

        y_col, title = get_plot_labels_and_title(args.file_type, args.simulators, args.batch_size)
        timing_helper.plot_timings(df, x_col='steps', y_col=y_col, title=title, log_x=True, output=out_filename)


if __name__ == "__main__":
    main()
