#!/bin/bash

cd "doing pendulum speed test"

cd pendulum_speed_test
./pendulum_speed_test.sh
cd ..

cd "now doing clutter test"

cd increasing_clutter_test
./increasing_clutter_test.sh
cd ..

echo "now doing articulation test"

cd articulation_test
./articulation_test.sh
cd ..

echo "tests complete"