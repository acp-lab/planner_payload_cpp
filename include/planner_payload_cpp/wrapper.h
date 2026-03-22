#include <stdlib.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Eigen>
#include <boost/array.hpp>
#include <ctime>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <numeric>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_planner_payload.h"

#define NX PLANNER_PAYLOAD_NX
#define NZ PLANNER_PAYLOAD_NZ
#define NU PLANNER_PAYLOAD_NU
#define NP PLANNER_PAYLOAD_NP
#define NBX PLANNER_PAYLOAD_NBX
#define NBX0 PLANNER_PAYLOAD_NBX0
#define NBU PLANNER_PAYLOAD_NBU
#define NSBX PLANNER_PAYLOAD_NSBX
#define NSBU PLANNER_PAYLOAD_NSBU
#define NSH PLANNER_PAYLOAD_NSH
#define NSG PLANNER_PAYLOAD_NSG
#define NSPHI PLANNER_PAYLOAD_NSPHI
#define NSHN PLANNER_PAYLOAD_NSHN
#define NSGN PLANNER_PAYLOAD_NSGN
#define NSPHIN PLANNER_PAYLOAD_NSPHIN
#define NSBXN PLANNER_PAYLOAD_NSBXN
#define NS PLANNER_PAYLOAD_NS
#define NSN PLANNER_PAYLOAD_NSN
#define NG PLANNER_PAYLOAD_NG
#define NBXN PLANNER_PAYLOAD_NBXN
#define NGN PLANNER_PAYLOAD_NGN
#define NY0 PLANNER_PAYLOAD_NY0
#define NY PLANNER_PAYLOAD_NY
#define NYN PLANNER_PAYLOAD_NYN
#define NH PLANNER_PAYLOAD_NH
#define NPHI PLANNER_PAYLOAD_NPHI
#define NHN PLANNER_PAYLOAD_NHN
#define NPHIN PLANNER_PAYLOAD_NPHIN
#define NR PLANNER_PAYLOAD_NR
const int N = PLANNER_PAYLOAD_N;

namespace planner_payload_nodelet {
static constexpr int kStateSize = PLANNER_PAYLOAD_NX;
static constexpr int kSamples = PLANNER_PAYLOAD_N;
static constexpr int kInputSize = PLANNER_PAYLOAD_NU;
static constexpr int yRefSize = PLANNER_PAYLOAD_NX + PLANNER_PAYLOAD_NU;

struct solver_output {
  // The Eigen Maps initialized in the class can directly change these values
  // below without worrying about transforming between matrices and arrays the
  // relevant sections of the arrays can then be passed to the solver
  double status, KKT_res, cpu_time;
  double u0[NU];
  double u1[NU];
  double x1[NX];
  double x2[NX];
  double x4[NX];
  double xi[NU];
  double ui[NU];
  double u_out[NU * (N)];
  double x_out[NX * (N + 1)];
};

struct solver_input {
  double x0[NX];
  double x[NX * (N)];
  double u[NU * N];
  double yref[(NX + NU) * N];
  double yref_e[(NX + NU)];
  double W[NY * NY];
  double WN[NX * NX];
};

// PLEASE DO NOT MOVE THESE ANYWHERE
// THEY BELONG HERE
// ELSE, EXPECT RANDOM PROBLEMS
extern solver_input acados_in;
extern solver_output acados_out;

class NMPCWrapper {
public:
  NMPCWrapper();
  NMPCWrapper(const Eigen::VectorXd Q_, const Eigen::VectorXd R_,
              const Eigen::VectorXd lbu_, const Eigen::VectorXd ubu_);

  bool
  prepare(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state);
  bool
  update(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state);

  void getStates(Eigen::Matrix<double, kStateSize, kSamples> &return_state);
  void getInputs(Eigen::Matrix<double, kInputSize, kSamples> &return_input);

  void setTrajectory(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSize, kSamples>>
          states,
      const Eigen::Ref<const Eigen::Matrix<double, kInputSize, kSamples>>
          inputs);
  void setMass(double mass);
  void setGravity(double gravity);
  void setWeightMatrices(std::vector<double> Q, std::vector<double> Q_e,
                         std::vector<double> R);

  void initStates();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  planner_payload_solver_capsule *acados_ocp_capsule;
  ocp_nlp_in *nlp_in;
  ocp_nlp_out *nlp_out;
  ocp_nlp_solver *nlp_solver;
  void *nlp_opts;
  ocp_nlp_plan_t *nlp_solver_plan;
  ocp_nlp_config *nlp_config;
  ocp_nlp_dims *nlp_dims;
  double *new_time_steps;
  int status;
  double mass_{1.0};
  double gravity_{9.81};
  void updateHoverInput();
  bool acados_is_prepared_{false};
  int acados_status;
  double *initial_state;
  Eigen::Map<Eigen::Matrix<double, yRefSize, kSamples, Eigen::ColMajor>>
      acados_reference_states_{acados_in.yref};
  Eigen::Map<Eigen::Matrix<double, kStateSize, 1, Eigen::ColMajor>>
      acados_initial_state_{acados_in.x0};
  Eigen::Map<Eigen::Matrix<double, yRefSize, 1, Eigen::ColMajor>>
      acados_reference_end_state_{acados_in.yref_e};
  Eigen::Map<Eigen::Matrix<double, kStateSize, kSamples, Eigen::ColMajor>>
      acados_states_in_{acados_in.x};
  Eigen::Map<Eigen::Matrix<double, kInputSize, kSamples, Eigen::ColMajor>>
      acados_inputs_in_{acados_in.u};
  Eigen::Map<Eigen::Matrix<double, kStateSize, kSamples, Eigen::ColMajor>>
      acados_states_{acados_out.x_out};
  Eigen::Map<Eigen::Matrix<double, kInputSize, kSamples, Eigen::ColMajor>>
      acados_inputs_{acados_out.u_out};
  Eigen::Matrix<real_t, kInputSize, 1> kHoverInput_ =
      Eigen::Matrix<real_t, kInputSize, 1>::Zero();
};

} // namespace planner_payload_nodelet
