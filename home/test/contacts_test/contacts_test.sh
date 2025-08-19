#!/bin/bash

BATCH_SIZES=(2048 4096 8192)

# Handle CLEAR command to remove all generated files
case "$1" in
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

        python plot_contacts.py $SIMULATORS_ARG --base_dir "../contacts_test/data" --file_type "$FILE_TYPE" $BATCH_SIZE 
        exit 0
        ;;


esac


STEPS=$(awk 'BEGIN {printf "%d\n", 1e9}') 
SPHERE_COUNTS=(1 5 7 10)


XML_PATHS=()

# Loop and generate only if file doesn't already exist
for n in "${SPHERE_COUNTS[@]}"; do
    XML_FILE="../xml/contact_${n}.xml"
    XML_PATHS+=("$XML_FILE")
done

echo "Generated or found XML files:"

run_batch_sim() {
    local sim="$1"
    local N="$2"
    for B in "${BATCH_SIZES[@]}"
    do
        echo "Running simulation with batch size $B"
        python "$sim" "$STEPS" "${XML_PATHS[@]}" -B "$B"
    done

    if [[ "$N" == "N" ]]; then
        echo "Rerunning with --Featherstone for $sim"
        for B in "${BATCH_SIZES[@]}"
        do
            echo "Running simulation with batch size $B and --Featherstone"
            python "$sim" "$STEPS" "${XML_PATHS[@]}" -B "$B" --Featherstone
        done

        echo "Rerunning with --MJWarp for $sim"
        for B in "${BATCH_SIZES[@]}"
        do
            echo "Running simulation with batch size $B and --MJWarp"
            python "$sim" "$STEPS" "${XML_PATHS[@]}" -B "$B" --MJWarp
        done
    fi
}

if [[ "$#" -eq 0 ]]; then
    echo "Running test on Newton"
    run_batch_sim test_scripts/newton_contacts.py "N"

    echo "Running test on Genesis"
    run_batch_sim test_scripts/genesis_contacts.py 

    echo "Running test on Mujoco"
    python "test_scripts/mujoco_contacts.py" "$STEPS" "${XML_PATHS[@]}"

    echo "Running test on MJX"
    run_batch_sim test_scripts/mjx_contacts.py

    exit 0
fi


case "$1" in
    Newton|newton)
        echo "Running test on Newton"
        run_batch_sim test_scripts/newton_contacts.py "N"
        ;;
    Genesis|genesis)
        echo "Running test on Genesis"
        run_batch_sim test_scripts/genesis_contacts.py 
        ;;
    Mujoco|mujoco)
        echo "Running test on Mujoco"
        python "test_scripts/mujoco_contacts.py" "$STEPS" "${XML_PATHS[@]}"
        ;;
    MJX|mjx)
        echo "Running test on MJX"
        run_batch_sim test_scripts/mjx_contacts.py
        ;;
    HELP|help)
        print_help
        ;;
esac
