import timing_helper
import pandas as pd
import argparse



if __name__ == "__main__":
    dfMujoco = timing_helper.read_timing_csv('data/mujoco_clutter.csv')
#    dfMJX = timing_helper.read_timing_csv('data/mjx_pendulum_speed.csv')
 #   dfGenesis = timing_helper.read_timing_csv('data/genesis_pendulum_speed.csv')
  #  dfNewton = timing_helper.read_timing_csv('data/newton_pendulum_speed.csv')

    df_combined = dfMujoco
   # df_combined = pd.merge(dfMujoco, dfMJX, on='steps', how='outer')
    #df_combined = pd.merge(df_combined, dfGenesis, on='steps', how='outer')
    #df_combined = pd.merge(df_combined, dfNewton, on="steps", how="outer")

    #print(df_combined)

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True, help="Output image file path")
    args = parser.parse_args()

    #print(f"Saving plot to: {args.out}")

    timing_helper.plot_timings(df=df_combined, x_col='N', output=args.out)