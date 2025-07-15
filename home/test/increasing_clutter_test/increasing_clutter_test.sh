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


SPHERE_COUNTS=(1 10 100)


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

XML_1=${XML_PATHS[0]}
XML_10=${XML_PATHS[1]}
XML_100=${XML_PATHS[2]}
#XML_1000=${XML_PATHS[3]}

echo "Generated or found XML files:"
echo "  XML_1:     $XML_1"
echo "  XML_10:    $XML_10"
echo "  XML_100:   $XML_100"
#echo "  XML_1000:  $XML_1000"

INPUT_LB=2
INPUT_UB=2
INPUT_POINTS=1

execute_tests() {
    test=$1
    echo $test
    for i in "${XML_PATHS[@]}"!
    do 
        echo "testing scene $i"
        python $test $INPUT_LB $INPUT_UB $INPUT_POINTS $i
    done 

}

if [[ "$#" -eq 0 ]]; then
    echo "Running test on Newton"
    #python test_scripts/newton_clutter.py "${XML_PATHS[@]}"

    echo "Running test on Genesis"
    #python test_scripts/genesis_clutter.py "${XML_PATHS[@]}"

    echo "Running test on Mujoco"
    python test_scripts/mujoco_clutter.py "${XML_PATHS[@]}"

    echo "Running test on MJX"
    python test_scripts/mjx_clutter.py "${XML_PATHS[@]}"

    exit 0
fi


case "$1" in
    Newton|newton)
        echo "Running test on ONLY Newton"
        #python test_scripts/newton_clutter.py "${XML_PATHS[@]}"
        ;;
    Genesis|genesis)
        echo "Running test on Genesis"
        #python test_scripts/genesis_clutter.py "${XML_PATHS[@]}"
        ;;
    Mujoco|mujoco|MJX|mjx)
        echo "Running test on Mujoco and MJX"
        execute_tests test_scripts/mujoco_mjx_clutter.py
        ;;
    MJX|mjx)
        echo "Running test on MJX"
        python test_scripts/mjx_clutter.py "${XML_PATHS[@]}"
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