import timing_helper
import pandas as pd
import argparse



if __name__ == "__main__":
    dfMujoco = timing_helper.read_timing_csv('data/mujoco_clutter.csv')
    dfMJX = timing_helper.read_timing_csv('data/mjx_clutter.csv')
    dfGenesis = timing_helper.read_timing_csv('data/genesis_clutter.csv')
    dfNewton = timing_helper.read_timing_csv('data/newton_clutter.csv')

    df_combined = pd.merge(dfMujoco, dfMJX, on='N', how='outer')
    df_combined = pd.merge(df_combined, dfGenesis, on='N', how='outer')
    df_combined = pd.merge(df_combined, dfNewton, on="N", how="outer")

    #print(df_combined)

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True, help="Output image file path")
    args = parser.parse_args()

    #print(f"Saving plot to: {args.out}")

    timing_helper.plot_timings(df=df_combined, x_col='N', output=args.out)