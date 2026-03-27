#ifndef NMPC_CONTROL_JERK_H
#define NMPC_CONTROL_JERK_H

#include <planner_payload_cpp/wrapper_jerk.h>

namespace planner_payload_jerk_nodelet {
class NMPCControlJerk {
public:
  NMPCControlJerk();

  void setState(const Eigen::Matrix<double, kStateSizeJerk, 1> &state,
                double stamp);
  void setReferenceStates(const Eigen::Matrix<double, kStateSizeJerk,
                                              kSamplesJerk> &reference_states);
  void setReferenceInputs(const Eigen::Matrix<double, kInputSizeJerk,
                                              kSamplesJerk> &reference_inputs);
  void setMass(double mass);
  void setGravity(double gravity);
  void setWeightMatrices(std::vector<double> Q, std::vector<double> Q_e,
                         std::vector<double> R);

  Eigen::Matrix<double, kStateSizeJerk, 1> getPredictedState();
  Eigen::Matrix<double, kInputSizeJerk, 1> getPredictedInput();
  Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> getPredictedStates();
  Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> getPredictedInputs();
  Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> getReferenceStates();
  Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> getReferenceInputs();
  int run();
  double getStampState();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  bool solve_from_scratch_;
  double stamp_current_state_{0.0};
  Eigen::Matrix<double, kStateSizeJerk, 1> current_state_;
  Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> reference_states_;
  Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> reference_inputs_;
  Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> predicted_states_;
  Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> predicted_inputs_;

  NMPCWrapperJerk wrapper_;
};

} // namespace planner_payload_jerk_nodelet

#endif
