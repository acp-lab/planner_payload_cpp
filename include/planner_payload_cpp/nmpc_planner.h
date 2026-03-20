#ifndef NMPC_CONTROL_H
#define NMPC_CONTROL_H

#include <planner_payload_cpp/wrapper.h>

namespace planner_payload_nodelet {
class NMPCControl {
public:
  NMPCControl();

  void setState(const Eigen::Matrix<double, kStateSize, 1> &state,
                double stamp);
  void setReferenceStates(
      const Eigen::Matrix<double, kStateSize, kSamples> &reference_states);
  void setReferenceInputs(
      const Eigen::Matrix<double, kInputSize, kSamples> &reference_inputs);
  void setMass(double mass);
  void setGravity(double gravity);
  void setWeightMatrices(std::vector<double> Q, std::vector<double> Q_e,
                         std::vector<double> R);

  Eigen::Matrix<double, kStateSize, 1> getState() { return current_state_; }
  Eigen::Matrix<double, kStateSize, 1> getPredictedState();
  Eigen::Matrix<double, kInputSize, 1> getPredictedInput();
  Eigen::Matrix<double, kStateSize, kSamples> getPredictedStates();
  Eigen::Matrix<double, kInputSize, kSamples> getPredictedInputs();
  Eigen::Matrix<double, kStateSize, kSamples> getReferenceStates();
  Eigen::Matrix<double, kInputSize, kSamples> getReferenceInputs();

  int run();
  double getStampState();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  bool solve_from_scratch_;
  double stamp_current_state_;
  Eigen::Matrix<double, kStateSize, 1> current_state_;
  Eigen::Matrix<double, kStateSize, kSamples> reference_states_;
  Eigen::Matrix<double, kInputSize, kSamples> reference_inputs_;
  Eigen::Matrix<double, kStateSize, kSamples> predicted_states_;
  Eigen::Matrix<double, kInputSize, kSamples> predicted_inputs_;

  NMPCWrapper wrapper_;
};

} // namespace planner_payload_nodelet

#endif
