#!/bin/bash

# Define variables with absolute paths
DATA_DIR="data"
SCRIPT_DIR="test_scripts"

XML="../xml/pendulum.xml"
INPUT_LB=3
INPUT_UB=9
INPUT_POINTS=8

print_help() {
    echo "USAGE: ./pendulum_speed_test [ENGINE_TO_TEST]"
    echo ""
    echo "PARAMETERS:"
    echo "  ENGINE_TO_TEST:  ALL (DEFAULT) - runs all physics engines tests (Newton, Genesis, Mujoco & MJX)"
    echo "                   Newton - runs only Newton test"
    echo "                   Genesis - runs only Genesis test"
    echo "                   Mujoco or MJX - runs only Mujoco & MJX tests"
    echo ""
    echo "USAGE: ./pendulum_speed_test SPECIAL_MODE [OUTPUT_IMAGE]"
    echo "SPECIAL MODES:"
    echo "  HELP               Print this help message"
    echo "  PLOT [output.png]  Plot results and save to output image file."
    echo "                     If output image filename is not provided, defaults to pendulum_speed_plot.png"
    exit 0
}

if [[ "$#" -eq 0 ]]; then
    echo "Running test on Newton"
    python test_scripts/newton_pendulum.py $INPUT_LB $INPUT_UB $INPUT_POINTS

    echo "Running test on Genesis"
    python test_scripts/genesis_pendulum.py $INPUT_LB $INPUT_UB $INPUT_POINTS

    echo "Running test on Mujoco & MJX"
    python test_scripts/mujoco_mjx_pendulum.py $INPUT_LB $INPUT_UB $INPUT_POINTS

    exit 0
fi


case "$1" in
    Newton|newton)
        echo "Running test on ONLY Newton"
        python test_scripts/newton_pendulum.py $INPUT_LB $INPUT_UB $INPUT_POINTS
        ;;
    Genesis|genesis)
        echo "Running test on Genesis"
        python test_scripts/genesis_pendulum.py $INPUT_LB $INPUT_UB $INPUT_POINTS
        ;;
    Mujoco|mujoco|MJX|mjx)
        echo "Running test on Mujoco & MJX"
        python test_scripts/mujoco_mjx_pendulum.py $INPUT_LB $INPUT_UB $INPUT_POINTS
        ;;
    HELP|help)
        print_help
        ;;
    PLOT|plot)
        if [[ "$#" -eq 2 ]]; then
        OUTPUT_IMAGE="$2"
        else
            OUTPUT_IMAGE="pendulum_speed_plot.png"
        fi

        echo "Plotting results to $OUTPUT_IMAGE"
        python "plot_pendulum_speed.py" --out "$OUTPUT_IMAGE"
        ;;
    *)
        echo "Unknown simulator: $arg"
        print_help
        ;;
esac
