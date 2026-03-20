#!/bin/bash
echo ""
echo "Let's build the NMPC!"
echo "enter your platform_type"
echo "default: race"
echo 'options: race race2 race_S voxl2 raxl2 iris eagle'
echo ""
read platform_type
platform_type=${platform_type:-race}
echo 'thank you!'
echo ""

python3 scripts/build_payload_planner.py $COLCON_PAYLOAD_WS_DIR/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml

# Creating the folder where we are going to paste the files
mkdir $COLCON_PAYLOAD_WS_DIR/install/planner_payload_cpp/
mkdir $COLCON_PAYLOAD_WS_DIR/install/planner_payload_cpp/lib/

cp c_generated_code/libacados_ocp_solver_planner_payload.so $COLCON_PAYLOAD_WS_DIR/install/planner_payload_cpp/lib/
##echo "Deleting old Files"
cd $COLCON_PAYLOAD_WS_DIR
source $COLCON_PAYLOAD_WS_DIR/install/setup.bash
colcon build --symlink-install --packages-select planner_payload_cpp
source $COLCON_PAYLOAD_WS_DIR/install/setup.bash
