#!/bin/bash

# Define variables with absolute paths
DATA_DIR="data"
SCRIPT_DIR="test_scripts"

INPUT_LB=5
INPUT_UB=9
INPUT_POINTS=5

BATCH_SIZES=(2048 4096 8192)

print_help() {
    echo "USAGE: ./loading_test [ENGINE_TO_TEST]"
    echo ""
    echo "PARAMETERS:"
    echo "  ENGINE_TO_TEST:  ALL (DEFAULT) - runs all physics engines tests (Newton, Genesis, Mujoco & MJX)"
    echo "                   Newton - runs only Newton test"
    echo "                   Genesis - runs only Genesis test"
    echo "                   Mujoco or MJX - runs only Mujoco & MJX tests"
    echo ""
    echo "USAGE: ./loading_test SPECIAL_MODE [OUTPUT_IMAGE]"
    echo "SPECIAL MODES:"
    echo "  HELP               Print this help message"
    echo "  PLOT [output.png]  Plot results and save to output image file."
    echo "                     If output image filename is not provided, defaults to pendulum_speed_plot.png"
    exit 0
}

run_batch_sim() {
    local sim="$1"
    for B in "${BATCH_SIZES[@]}"
    do
        echo "Running simulation with batch size $B"
        python "$sim" -B "$B"
    done
}

if [[ "$#" -eq 0 ]]; then
    echo "Running test on Newton"
    run_batch_sim test_scripts/newton_loading.py "N"

    echo "Running test on Genesis"
    run_batch_sim test_scripts/genesis_loading.py

    echo "Running test on Mujoco"
    python test_scripts/mujoco_loading.py 

    echo "Running test on MJX"
    run_batch_sim "test_scripts/mjx_loading.py"

    exit 0
fi


case "$1" in
    Newton|newton)
        echo "Running test on Newton"
        run_batch_sim test_scripts/newton_loading.py 
        ;;
    Genesis|genesis)
        echo "Running test on Genesis"
        run_batch_sim test_scripts/genesis_loading.py
        ;;
    Mujoco|mujoco)
        echo "Running test on Mujoco"
        run_batch_sim test_scripts/mujoco_loading.py
        ;;
    MJX|mjx)
        echo "Running test on MJX"
        run_batch_sim "test_scripts/mjx_loading.py"
        ;;
    HELP|help)
        print_help
        ;;
    PLOT|plot)
        FILE_TYPE=""
        BATCH_SIZE=""
        OUTPUT_IMAGE=""
        SIMULATORS_ARG=""

        # Check number of args after 'plot'
        if [[ "$#" -lt 2 ]]; then
            echo "Usage: $0 plot [simulator(s)] <file_type> [batch_size] [output_image]"
            exit 1
        fi

        # Detect if the first arg after 'plot' looks like a file_type (speed, env_fps, total_fps)
        # If so, user skipped simulators and we set no simulators arg to python (which means 'all')
        case "$2" in
            speed|env_fps|total_fps)
                FILE_TYPE="$2"
                shift 2
                ;;
            *)
                # Assume first arg is simulators
                SIMULATORS="$2"
                # Convert comma-separated to space-separated
                SIMULATORS_ARG="--simulators $(echo $SIMULATORS | tr ',' ' ')"
                FILE_TYPE="$3"
                shift 3
                ;;
        esac

        # If FILE_TYPE is still empty, error out
        if [[ -z "$FILE_TYPE" ]]; then
            echo "Error: file_type must be specified"
            exit 1
        fi

        # Optional batch size
        if [[ "$#" -ge 1 ]]; then
            BATCH_SIZE="--batch_size $1"
            shift 1
        fi


        if [[ -n "$BATCH_SIZE" ]]; then
            BATCH_SIZE_VAL="${BATCH_SIZE#--batch_size }"
        else
            BATCH_SIZE_VAL="all"
        fi

        echo "Plotting results for simulators: ${SIMULATORS:-all}, file_type: $FILE_TYPE, batch_size: ${BATCH_SIZE_VAL:-all}"

        python plot_timings.py $SIMULATORS_ARG --base_dir "../loading_test/data" --file_type "$FILE_TYPE" $BATCH_SIZE
        ;;

esac

# 2048
# 9.883847391605377 s (average time)
# 0.023828735351562498 MB (average mem per env)

# 4096
# 10.033642868995667 s (average time)
# 0.02927520751953125 MB (average mem per env)

# 8192
#10.795728571414948 s (average time)
#0.032043991088867185 MB (average mem per env)