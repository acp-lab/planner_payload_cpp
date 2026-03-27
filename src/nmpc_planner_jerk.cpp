#include "planner_payload_cpp/nmpc_planner_jerk.h"

#include <iostream>

namespace planner_payload_jerk_nodelet {

NMPCControlJerk::NMPCControlJerk()
    : solve_from_scratch_(true),
      current_state_(Eigen::Matrix<double, kStateSizeJerk, 1>::Zero()),
      reference_states_(
          Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>::Zero()),
      reference_inputs_(
          Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>::Zero()),
      predicted_states_(
          Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>::Zero()),
      predicted_inputs_(
          Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>::Zero()) {
  current_state_(8) = -1.0;
  reference_states_.row(8).setConstant(-1.0);
  predicted_states_.row(8).setConstant(-1.0);
}

void NMPCControlJerk::setState(
    const Eigen::Matrix<double, kStateSizeJerk, 1> &state, double stamp) {
  current_state_ = state;
  stamp_current_state_ = stamp;
}

void NMPCControlJerk::setReferenceStates(
    const Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>
        &reference_states) {
  reference_states_ = reference_states;
}

void NMPCControlJerk::setReferenceInputs(
    const Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>
        &reference_inputs) {
  reference_inputs_ = reference_inputs;
}

void NMPCControlJerk::setMass(double mass) { wrapper_.setMass(mass); }
void NMPCControlJerk::setGravity(double gravity) {
  wrapper_.setGravity(gravity);
}
void NMPCControlJerk::setWeightMatrices(std::vector<double> Q,
                                        std::vector<double> Q_e,
                                        std::vector<double> R) {
  wrapper_.setWeightMatrices(Q, Q_e, R);
}

double NMPCControlJerk::getStampState() { return stamp_current_state_; }

Eigen::Matrix<double, kStateSizeJerk, 1> NMPCControlJerk::getPredictedState() {
  return predicted_states_.col(std::min(1, kSamplesJerk - 1));
}

Eigen::Matrix<double, kInputSizeJerk, 1> NMPCControlJerk::getPredictedInput() {
  return predicted_inputs_.col(0);
}

Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>
NMPCControlJerk::getPredictedStates() {
  return predicted_states_;
}

Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>
NMPCControlJerk::getPredictedInputs() {
  return predicted_inputs_;
}

Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>
NMPCControlJerk::getReferenceStates() {
  return reference_states_;
}

Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>
NMPCControlJerk::getReferenceInputs() {
  return reference_inputs_;
}

int NMPCControlJerk::run() {
  wrapper_.setTrajectory(reference_states_, reference_inputs_);
  if (solve_from_scratch_) {
    std::cout << "Solving jerk NMPC with hover as initial guess.\n";
    wrapper_.prepare(current_state_);
    solve_from_scratch_ = false;
  }

  const int acados_status = wrapper_.update(current_state_);
  wrapper_.getStates(predicted_states_);
  wrapper_.getInputs(predicted_inputs_);
  return acados_status;
}

} // namespace planner_payload_jerk_nodelet
