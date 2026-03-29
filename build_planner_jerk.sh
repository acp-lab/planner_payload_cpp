#!/bin/bash
set -e

echo ""
echo "Generating payload NMPC acados code"
echo "  - acceleration-input model"
echo "  - jerk-input model"
echo ""

python3 scripts/build_payload_planner.py \
  "$COLCON_PAYLOAD_WS_DIR/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml"

python3 scripts/build_payload_planner_jerk.py \
  "$COLCON_PAYLOAD_WS_DIR/src/acp-autonomy-stack/config/eagle/default/dq_control.yaml"

mkdir -p "$COLCON_PAYLOAD_WS_DIR/install/planner_payload_cpp/lib"

cp c_generated_code/libacados_ocp_solver_planner_payload.so \
  "$COLCON_PAYLOAD_WS_DIR/install/planner_payload_cpp/lib/"

if [ -f c_generated_code_jerk/libacados_ocp_solver_planner_payload_jerk.so ]; then
  cp c_generated_code_jerk/libacados_ocp_solver_planner_payload_jerk.so \
    "$COLCON_PAYLOAD_WS_DIR/install/planner_payload_cpp/lib/"
fi

echo "Generated acceleration solver artifacts under:"
echo "  $(pwd)/c_generated_code"
echo "  $(pwd)/acados_ocp_planner_payload.json"
echo "Generated jerk solver artifacts under:"
echo "  $(pwd)/c_generated_code_jerk"
echo "  $(pwd)/acados_ocp_planner_payload_jerk.json"

##echo "Deleting old Files"
cd $COLCON_PAYLOAD_WS_DIR
source $COLCON_PAYLOAD_WS_DIR/install/setup.bash
colcon build --symlink-install --packages-select planner_payload_cpp
source $COLCON_PAYLOAD_WS_DIR/install/setup.bash
