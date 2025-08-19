#!/bin/bash

#echo "doing pendulum speed test"
#
#cd pendulum_speed_test
#./pendulum_speed_test.sh
#cd ..
#
echo "now doing clutter test"

cd increasing_clutter_test
./increasing_clutter_test.sh
cd ..

echo "now doing contact test"

cd contacts_test
./contacts_test.sh
cd ..

cd loading_test
./loading_test.sh
cd ..


#cd articulation_test
#./articulation_test.sh
#cd ..
#
#cd walker_test
#./walker_test.sh
#cd ..

echo "tests complete"