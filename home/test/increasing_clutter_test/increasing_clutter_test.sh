#!/bin/bash

GENERATOR_SCRIPT="generate_random_spheres.py"

# Handle CLEAR command to remove all generated files
case "$1" in
    CLEAR|clear)
    echo "Removing all random_spheres_*.xml files..."
    cd ../xml
    rm -f random_spheres_*.xml
    echo "Done."
    cd ../increasing_clutter_test
    exit 0
    
esac


STEPS=$(awk 'BEGIN {printf "%d\n", 1e2}')  # -> 100 as integer
SPHERE_COUNTS=(1 10)


XML_PATHS=()

# Loop and generate only if file doesn't already exist
for n in "${SPHERE_COUNTS[@]}"; do
    XML_FILE="../xml/random_spheres_${n}.xml"
    if [[ -f "$XML_FILE" ]]; then
        echo "File $XML_FILE already exists. Skipping."
    else
        echo "Generating $XML_FILE"
        python "$GENERATOR_SCRIPT" "$n"
    fi
    XML_PATHS+=("$XML_FILE")
done

echo "Generated or found XML files:"

if [[ "$#" -eq 0 ]]; then
    echo "Running test on Newton"
    python test_scripts/newton_clutter.py "$STEPS" "${XML_PATHS[@]}"

    echo "Running test on Genesis"
    python test_scripts/genesis_clutter.py "$STEPS" "${XML_PATHS[@]}"

    echo "Running test on Mujoco and MJX"
    python test_scripts/mujoco_mjx_clutter.py "$STEPS" "${XML_PATHS[@]}"

    exit 0
fi


case "$1" in
    Newton|newton)
        echo "Running test on ONLY Newton"
        python test_scripts/newton_clutter.py "$STEPS" "${XML_PATHS[@]}"
        ;;
    Genesis|genesis)
        echo "Running test on Genesis"
        python test_scripts/genesis_clutter.py "$STEPS" "${XML_PATHS[@]}"
        ;;
    Mujoco|mujoco|MJX|mjx)
        echo "Running test on Mujoco and MJX"
        python test_scripts/mujoco_mjx_clutter.py "$STEPS" "${XML_PATHS[@]}"
        ;;
    HELP|help)
        print_help
        ;;
    PLOT|plot)
        if [[ "$#" -eq 2 ]]; then
        OUTPUT_IMAGE="$2"
        else
            OUTPUT_IMAGE="clutter_plot.png"
        fi

        echo "Plotting results to $OUTPUT_IMAGE"
        python plot_clutter.py --out "$OUTPUT_IMAGE"
        ;;
    *)
        echo "Unknown simulator: $arg"
        print_help
        ;;
esac