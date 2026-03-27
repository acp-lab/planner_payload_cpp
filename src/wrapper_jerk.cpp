#include "planner_payload_cpp/wrapper_jerk.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>

namespace planner_payload_jerk_nodelet {

solver_input_jerk acados_in_jerk;
solver_output_jerk acados_out_jerk;

NMPCWrapperJerk::NMPCWrapperJerk() {
  acados_ocp_capsule_ = planner_payload_jerk_acados_create_capsule();
  const int status = planner_payload_jerk_acados_create_with_discretization(
      acados_ocp_capsule_, N_JERK, new_time_steps_);
  if (status != 0) {
    std::cerr << "planner_payload_jerk_acados_create() returned status "
              << status << std::endl;
    std::exit(1);
  }

  nlp_config_ = planner_payload_jerk_acados_get_nlp_config(acados_ocp_capsule_);
  nlp_dims_ = planner_payload_jerk_acados_get_nlp_dims(acados_ocp_capsule_);
  nlp_in_ = planner_payload_jerk_acados_get_nlp_in(acados_ocp_capsule_);
  nlp_out_ = planner_payload_jerk_acados_get_nlp_out(acados_ocp_capsule_);
  nlp_solver_ = planner_payload_jerk_acados_get_nlp_solver(acados_ocp_capsule_);
  nlp_opts_ = planner_payload_jerk_acados_get_nlp_opts(acados_ocp_capsule_);

  Eigen::Matrix<double, kStateSizeJerk, 1> hover_state =
      Eigen::Matrix<double, kStateSizeJerk, 1>::Zero();
  hover_state(8) = -1.0;
  resetWarmStart(hover_state);
}

void NMPCWrapperJerk::resetWarmStart(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state) {
  acados_initial_state_ = state;
  acados_states_ = state.replicate(1, kSamplesJerk);
  acados_inputs_.setZero();

  ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                "lbx", acados_in_jerk.x0);
  ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                "ubx", acados_in_jerk.x0);

  acados_reference_states_.block(0, 0, kStateSizeJerk, kSamplesJerk) =
      state.replicate(1, kSamplesJerk);
  acados_reference_states_.block(kStateSizeJerk, 0, kInputSizeJerk,
                                 kSamplesJerk) =
      kHoverInput_.replicate(1, kSamplesJerk);
  acados_reference_end_state_.segment(0, kStateSizeJerk) = state;
  acados_reference_end_state_.segment(kStateSizeJerk, kInputSizeJerk).setZero();
}

void NMPCWrapperJerk::shiftWarmStart(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state) {
  if (!acados_is_prepared_) {
    resetWarmStart(state);
    return;
  }

  Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> shifted_states =
      acados_states_;
  Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> shifted_inputs =
      acados_inputs_;
  if (kSamplesJerk > 1) {
    shifted_states.leftCols(kSamplesJerk - 1) =
        acados_states_.rightCols(kSamplesJerk - 1);
    shifted_inputs.leftCols(kSamplesJerk - 1) =
        acados_inputs_.rightCols(kSamplesJerk - 1);
  }
  shifted_states.col(0) = state;
  shifted_states.col(kSamplesJerk - 1) = acados_states_.col(kSamplesJerk - 1);
  shifted_inputs.col(kSamplesJerk - 1) = acados_inputs_.col(kSamplesJerk - 1);

  acados_states_ = shifted_states;
  acados_inputs_ = shifted_inputs;
  for (int i = 0; i <= N_JERK; ++i) {
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x",
                    acados_out_jerk.x_out + i * NX_JERK);
  }
  for (int i = 0; i < N_JERK; ++i) {
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u",
                    acados_out_jerk.u_out + i * NU_JERK);
  }
}

bool NMPCWrapperJerk::prepare(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state) {
  resetWarmStart(state);
  for (int i = 0; i <= N_JERK; ++i) {
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x",
                    acados_out_jerk.x_out + i * NX_JERK);
  }
  for (int i = 0; i < N_JERK; ++i) {
    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u",
                    acados_out_jerk.u_out + i * NU_JERK);
  }
  acados_is_prepared_ = true;
  return true;
}

int NMPCWrapperJerk::update(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state) {
  shiftWarmStart(state);

  acados_initial_state_ = state;
  ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, 0, "x",
                  acados_in_jerk.x0);
  ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                "lbx", acados_in_jerk.x0);
  ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                "ubx", acados_in_jerk.x0);

  int y_indices[kYRefSizeJerk];
  std::iota(y_indices, y_indices + kYRefSizeJerk, 0);
  for (int i = 0; i < N_JERK; ++i) {
    planner_payload_jerk_acados_update_params_sparse(
        acados_ocp_capsule_, i, y_indices,
        acados_in_jerk.yref + i * kYRefSizeJerk, kYRefSizeJerk);
  }
  planner_payload_jerk_acados_update_params_sparse(
      acados_ocp_capsule_, N_JERK, y_indices, acados_in_jerk.yref_e,
      kYRefSizeJerk);

  const int acados_status =
      planner_payload_jerk_acados_solve(acados_ocp_capsule_);

  for (int i = 0; i <= nlp_dims_->N; ++i) {
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "x",
                    &acados_out_jerk.x_out[i * NX_JERK]);
  }
  for (int i = 0; i < nlp_dims_->N; ++i) {
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "u",
                    &acados_out_jerk.u_out[i * NU_JERK]);
  }
  return acados_status;
}

void NMPCWrapperJerk::getStates(
    Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> &return_state) {
  return_state = acados_states_;
}

void NMPCWrapperJerk::getInputs(
    Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> &return_input) {
  return_input = acados_inputs_;
}

void NMPCWrapperJerk::setTrajectory(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>>
        states,
    const Eigen::Ref<const Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>>
        inputs) {
  acados_reference_states_.block(0, 0, kStateSizeJerk, kSamplesJerk) = states;
  acados_reference_states_.block(kStateSizeJerk, 0, kInputSizeJerk,
                                 kSamplesJerk) = inputs;
  acados_reference_end_state_.segment(0, kStateSizeJerk) =
      states.col(kSamplesJerk - 1);
  acados_reference_end_state_.segment(kStateSizeJerk, kInputSizeJerk).setZero();
}

void NMPCWrapperJerk::setMass(double mass) { mass_ = mass; }
void NMPCWrapperJerk::setGravity(double gravity) { gravity_ = gravity; }

void NMPCWrapperJerk::setWeightMatrices(std::vector<double> Q,
                                        std::vector<double> Q_e,
                                        std::vector<double> R) {
  std::vector<double> params;
  params.reserve(Q.size() + Q_e.size() + R.size());
  for (const auto &vec : {Q, Q_e, R}) {
    params.insert(params.end(), vec.begin(), vec.end());
  }

  const int params_size = static_cast<int>(params.size());
  const int expected_cost_params = NP_JERK - kYRefSizeJerk;
  if (params_size != expected_cost_params) {
    std::cerr << "[NMPCWrapperJerk] setWeightMatrices size mismatch: got "
              << params_size << " expected " << expected_cost_params
              << std::endl;
    return;
  }

  std::vector<int> params_indices(params_size);
  std::iota(params_indices.begin(), params_indices.end(), kYRefSizeJerk);
  for (int i = 0; i < N_JERK; ++i) {
    const int status = planner_payload_jerk_acados_update_params_sparse(
        acados_ocp_capsule_, i, params_indices.data(), params.data(),
        params_size);
    if (status != 0) {
      std::cerr << "[NMPCWrapperJerk] update_params_sparse failed at stage "
                << i << " with status " << status << std::endl;
      return;
    }
  }
}

} // namespace planner_payload_jerk_nodelet
