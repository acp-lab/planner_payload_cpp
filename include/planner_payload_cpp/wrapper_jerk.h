#ifndef NMPC_WRAPPER_JERK_H
#define NMPC_WRAPPER_JERK_H

#include <vector>

#include <Eigen/Eigen>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_planner_payload_jerk.h"

#define NX_JERK PLANNER_PAYLOAD_JERK_NX
#define NZ_JERK PLANNER_PAYLOAD_JERK_NZ
#define NU_JERK PLANNER_PAYLOAD_JERK_NU
#define NP_JERK PLANNER_PAYLOAD_JERK_NP
#define NBU_JERK PLANNER_PAYLOAD_JERK_NBU
#define NSH_JERK PLANNER_PAYLOAD_JERK_NSH
const int N_JERK = PLANNER_PAYLOAD_JERK_N;

namespace planner_payload_jerk_nodelet {
static constexpr int kStateSizeJerk = PLANNER_PAYLOAD_JERK_NX;
static constexpr int kSamplesJerk = PLANNER_PAYLOAD_JERK_N;
static constexpr int kInputSizeJerk = PLANNER_PAYLOAD_JERK_NU;
static constexpr int kYRefSizeJerk =
    PLANNER_PAYLOAD_JERK_NX + PLANNER_PAYLOAD_JERK_NU;

struct solver_output_jerk {
  double status, KKT_res, cpu_time;
  double u0[NU_JERK];
  double u1[NU_JERK];
  double x1[NX_JERK];
  double x2[NX_JERK];
  double x4[NX_JERK];
  double xi[NU_JERK];
  double ui[NU_JERK];
  double u_out[NU_JERK * N_JERK];
  double x_out[NX_JERK * (N_JERK + 1)];
};

struct solver_input_jerk {
  double x0[NX_JERK];
  double x[NX_JERK * N_JERK];
  double u[NU_JERK * N_JERK];
  double yref[(NX_JERK + NU_JERK) * N_JERK];
  double yref_e[(NX_JERK + NU_JERK)];
  double W[1];
  double WN[1];
};

extern solver_input_jerk acados_in_jerk;
extern solver_output_jerk acados_out_jerk;

class NMPCWrapperJerk {
public:
  NMPCWrapperJerk();

  bool prepare(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state);
  int update(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state);

  void
  getStates(Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> &return_state);
  void
  getInputs(Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> &return_input);

  void
  setTrajectory(const Eigen::Ref<
                    const Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>>
                    states,
                const Eigen::Ref<
                    const Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>>
                    inputs);
  void setMass(double mass);
  void setGravity(double gravity);
  void setWeightMatrices(std::vector<double> Q, std::vector<double> Q_e,
                         std::vector<double> R);

private:
  void resetWarmStart(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state);
  void shiftWarmStart(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> state);

  planner_payload_jerk_solver_capsule *acados_ocp_capsule_{nullptr};
  ocp_nlp_in *nlp_in_{nullptr};
  ocp_nlp_out *nlp_out_{nullptr};
  ocp_nlp_solver *nlp_solver_{nullptr};
  void *nlp_opts_{nullptr};
  ocp_nlp_config *nlp_config_{nullptr};
  ocp_nlp_dims *nlp_dims_{nullptr};
  double *new_time_steps_{nullptr};
  double mass_{1.0};
  double gravity_{9.81};
  bool acados_is_prepared_{false};

  Eigen::Map<
      Eigen::Matrix<double, kYRefSizeJerk, kSamplesJerk, Eigen::ColMajor>>
      acados_reference_states_{acados_in_jerk.yref};
  Eigen::Map<Eigen::Matrix<double, kStateSizeJerk, 1, Eigen::ColMajor>>
      acados_initial_state_{acados_in_jerk.x0};
  Eigen::Map<Eigen::Matrix<double, kYRefSizeJerk, 1, Eigen::ColMajor>>
      acados_reference_end_state_{acados_in_jerk.yref_e};
  Eigen::Map<
      Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk, Eigen::ColMajor>>
      acados_states_in_{acados_in_jerk.x};
  Eigen::Map<
      Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk, Eigen::ColMajor>>
      acados_inputs_in_{acados_in_jerk.u};
  Eigen::Map<
      Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk, Eigen::ColMajor>>
      acados_states_{acados_out_jerk.x_out};
  Eigen::Map<
      Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk, Eigen::ColMajor>>
      acados_inputs_{acados_out_jerk.u_out};
  Eigen::Matrix<real_t, kInputSizeJerk, 1> kHoverInput_ =
      Eigen::Matrix<real_t, kInputSizeJerk, 1>::Zero();
};

} // namespace planner_payload_jerk_nodelet

#endif
