import timing_helper
import pandas as pd

if __name__ == "__main__":
    dfMujoco = timing_helper.read_timing_csv('mujoco_pendulum_speed.csv')
    dfMJX = timing_helper.read_timing_csv('mjx_pendulum_speed.csv')
    dfGenesis = timing_helper.read_timing_csv('genesis_pendulum_speed.csv')
    # genesis
    # newton

    df_combined = pd.merge(dfMujoco, dfMJX, on='steps', how='outer')
    df_combined = pd.merge(df_combined, dfGenesis, on='steps', how='outer')
    # merge genesis
    # merge newton

    print(df_combined)

    timing_helper.plot_timings(df=df_combined, x_col='steps', output="pendulum_speed_plot.png")