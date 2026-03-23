#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <planner_payload_cpp/nmpc_planner.h>
#include <quadrotor_msgs/msg/position_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace planner_payload_nodelet {
class NMPCControlNodelet : public rclcpp::Node {
public:
  NMPCControlNodelet(const rclcpp::NodeOptions &options)
      : Node("nmpc_control_nodelet", options), frame_id_("world"),
        enable_motors_(false), _optimization_error(false), _aux_initial(false),
        set_pre_odom_quat_(false) {

    this->declare_parameter("mass_payload", 0.1128);
    this->declare_parameter("gravity", 9.81);
    this->declare_parameter("cable_length", 0.88);

    logParameter("mass_payload", mass_payload_, "%.4f");
    logParameter("gravity", gravity_, "%.4f");
    logParameter("cable_length", cable_length_, "%.4f");

    inertia_matrix_ = Eigen::Matrix3d::Zero();

    this->declare_parameter("ixx", 0.0);
    this->declare_parameter("iyy", 0.0);
    this->declare_parameter("izz", 0.0);

    logParameter("ixx", inertia_matrix_(0, 0), "%.4f");
    logParameter("iyy", inertia_matrix_(1, 1), "%.4f");
    logParameter("izz", inertia_matrix_(2, 2), "%.4f");

    this->declare_parameter("platform_type", "");
    logParameter("platform_type", platform_type_, "%s");

    this->declare_parameter<std::vector<double>>(
        "nmpc.Q", std::vector<double>{210., 210., 210., 1., 1., 1., 5., 5., 5.,
                                      1., 1., 1., 50., 10., 10., 10.});
    this->declare_parameter<std::vector<double>>(
        "nmpc.Q_e", std::vector<double>{210., 210., 210., 1., 1., 1., 5., 5.,
                                        5., 1., 1., 1., 50., 10., 10., 10.});
    this->declare_parameter<std::vector<double>>(
        "nmpc.R", std::vector<double>{0.5, 0.1, 0.1, 0.1});

    rclcpp::Parameter Q_param = this->get_parameter("nmpc.Q");
    rclcpp::Parameter Q_e_param = this->get_parameter("nmpc.Q_e");
    rclcpp::Parameter R_param = this->get_parameter("nmpc.R");

    RCLCPP_INFO(this->get_logger(), "[NMPC Payload Planner] Q: %s",
                Q_param.value_to_string().c_str());
    RCLCPP_INFO(this->get_logger(), "[NMPC Payload Planner] Q_e: %s",
                Q_e_param.value_to_string().c_str());
    RCLCPP_INFO(this->get_logger(), "[NMPC Payload Planner] R: %s",
                R_param.value_to_string().c_str());

    Q_param_ = Q_param.as_double_array();
    Q_e_param_ = Q_e_param.as_double_array();
    R_param_ = R_param.as_double_array();

    clock_ = rclcpp::Clock();

    controller_.setMass(mass_payload_);
    controller_.setGravity(gravity_);
    controller_.setWeightMatrices(Q_param_, Q_e_param_, R_param_);

    // custom QoS
    auto qos_profile = rclcpp::SensorDataQoS();

    // Publish payload desired and predictions
    pub_ref_traj_ = this->create_publisher<nav_msgs::msg::Path>(
        "/quadrotor/payload_reference_path", 1);

    pub_pred_traj_ = this->create_publisher<nav_msgs::msg::Path>(
        "/quadrotor/payload_predicted_path", 1);

    // Publish quadrotor desired
    pub_desired_quadrotor_ =
        this->create_publisher<quadrotor_msgs::msg::PositionCommand>(
            "/quadrotor/payload_planner_quadrotor_cmd", 1);

    // Subscribers
    sub_payload_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/quadrotor/payload/odom", qos_profile,
        std::bind(&NMPCControlNodelet::payloadOdomCallback, this,
                  std::placeholders::_1));
    sub_quad_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/quadrotor/odom", qos_profile,
        std::bind(&NMPCControlNodelet::quadOdomCallback, this,
                  std::placeholders::_1));
    sub_position_cmd_ =
        this->create_subscription<quadrotor_msgs::msg::PositionCommand>(
            "/quadrotor/position_cmd", 1,
            std::bind(&NMPCControlNodelet::referenceCallback, this,
                      std::placeholders::_1));

    srv_activate_payload_ = this->create_service<std_srvs::srv::SetBool>(
        "/quadrotor/activate_payload_nmpc_controller",
        std::bind(&NMPCControlNodelet::activate_payload_callback, this,
                  std::placeholders::_1, std::placeholders::_2));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  template <typename T>
  void logParameter(const std::string &param_name, T &param_value,
                    const std::string &format) {
    if (!this->get_parameter(param_name, param_value)) {
      RCLCPP_ERROR(this->get_logger(), "[NMPC Payload Planner] No %s!",
                   param_name.c_str());
    } else {
      if constexpr (std::is_same_v<T, std::string>) {
        RCLCPP_INFO(this->get_logger(), "[NMPC Payload Planner] %s: %s",
                    param_name.c_str(), param_value.c_str());
      } else {
        RCLCPP_INFO(
            this->get_logger(),
            ("[NMPC Payload Planner] " + param_name + ": " + format).c_str(),
            param_value);
      }
    }
  }

  NMPCControl controller_;
  rclcpp::Clock clock_;

  // from odom callback
  std::string frame_id_;
  Eigen::Vector4d pre_odom_quat_;
  bool enable_motors_;
  bool _optimization_error;
  bool _aux_initial;
  bool set_pre_odom_quat_;

  // from param server
  double mass_payload_;
  double gravity_;
  double cable_length_;

  Eigen::Matrix4d mixer_matrix_inv_;
  Eigen::Matrix3d inertia_matrix_;

  std::string platform_type_;
  std::vector<double> Q_param_;
  std::vector<double> Q_e_param_;
  std::vector<double> R_param_;

  Eigen::Vector3d quad_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d quad_velocity_{Eigen::Vector3d::Zero()};
  bool use_nmpc_payload_{false};

  void payloadOdomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  void quadOdomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  void referenceCallback(
      const quadrotor_msgs::msg::PositionCommand::SharedPtr pos_cmd);
  void run();
  void publishPrediction();
  void publishReference();
  void publishDesiredQuadrotorCommand();
  Eigen::Vector3d quadrotorPositionFromPayloadState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> &state)
      const;
  Eigen::Vector3d quadrotorVelocityFromPayloadState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> &state)
      const;
  Eigen::Vector3d quadrotorAccelerationFromPayloadState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> &state)
      const;

  void activate_payload_callback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response);

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_ref_traj_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_pred_traj_;

  rclcpp::Publisher<quadrotor_msgs::msg::PositionCommand>::SharedPtr
      pub_desired_quadrotor_;

  // rclcpp::Publisher<mujoco_msgs::msg::Dual>::SharedPtr pub_dual_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      sub_payload_odometry_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_quad_odometry_;

  rclcpp::Subscription<quadrotor_msgs::msg::PositionCommand>::SharedPtr
      sub_position_cmd_;

  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_activate_payload_;
};
void NMPCControlNodelet::quadOdomCallback(
    const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
  quad_position_ << odom_msg->pose.pose.position.x,
      odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z;
  quad_velocity_ << odom_msg->twist.twist.linear.x,
      odom_msg->twist.twist.linear.y, odom_msg->twist.twist.linear.z;
}

void NMPCControlNodelet::activate_payload_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  use_nmpc_payload_ = request->data;
  response->success = true;
  response->message = use_nmpc_payload_ ? "Payload NMPC Planner active"
                                        : "Standard TRPY active";
  RCLCPP_INFO(this->get_logger(), "Switching controller mode: %s",
              response->message.c_str());
}

void NMPCControlNodelet::payloadOdomCallback(
    const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
  Eigen::Matrix<double, kStateSize, 1> state(
      Eigen::Matrix<double, kStateSize, 1>::Zero());

  frame_id_ = odom_msg->header.frame_id;
  state(0) = odom_msg->pose.pose.position.x;
  state(1) = odom_msg->pose.pose.position.y;
  state(2) = odom_msg->pose.pose.position.z;

  state(3) = odom_msg->twist.twist.linear.x;
  state(4) = odom_msg->twist.twist.linear.y;
  state(5) = odom_msg->twist.twist.linear.z;

  Eigen::Vector3d cable_dir(0.0, 0.0, -1.0);
  Eigen::Vector3d cable_r(0.0, 0.0, 0.0);

  const Eigen::Vector3d payload_position = state.segment<3>(0);
  const Eigen::Vector3d payload_velocity = state.segment<3>(3);
  const Eigen::Vector3d a = payload_position - quad_position_;
  const Eigen::Vector3d a_dot = payload_velocity - quad_velocity_;
  const double norm_a = a.norm();
  const double dot_a = a.dot(a);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d projection = I - (a * a.transpose()) / dot_a;
  const Eigen::Vector3d n_dot = (1.0 / norm_a) * projection * a_dot;

  cable_dir = a / norm_a;
  cable_r = cable_dir.cross(n_dot);

  state.segment<3>(6) = cable_dir;
  state.segment<3>(9) = cable_r;

  // Tension and cable angular acceleration are not measured: use predicted
  // values from previous NMPC solve, with a hover fallback.
  state(12) = mass_payload_ * gravity_;
  state.segment<3>(13).setZero();
  const Eigen::Matrix<double, kStateSize, 1> predicted_state =
      controller_.getPredictedState();
  if (predicted_state.allFinite() && predicted_state(12) > 1e-6) {
    state(12) = predicted_state(12);
    state.segment<3>(13) = predicted_state.segment<3>(13);
  }

  const double stamp_sec =
      static_cast<double>(odom_msg->header.stamp.sec) +
      static_cast<double>(odom_msg->header.stamp.nanosec) * 1e-9;
  controller_.setState(state, stamp_sec);
}

void NMPCControlNodelet::referenceCallback(
    const quadrotor_msgs::msg::PositionCommand::SharedPtr reference_msg) {
  Eigen::Matrix<double, kStateSize, kSamples> reference_states =
      Eigen::Matrix<double, kStateSize, kSamples>::Zero();
  Eigen::Matrix<double, kInputSize, kSamples> reference_inputs =
      Eigen::Matrix<double, kInputSize, kSamples>::Zero();

  const Eigen::Vector3d n_eq(0.0, 0.0, -1.0);
  const Eigen::Vector3d r_eq(0.0, 0.0, 0.0);
  const double thrust_eq = mass_payload_ * gravity_;

  const size_t n_points = reference_msg->points.size();

  if (n_points == 0) {
    for (int i = 0; i < kSamples; ++i) {
      reference_states(0, i) = reference_msg->position.x;
      reference_states(1, i) = reference_msg->position.y;
      reference_states(2, i) = reference_msg->position.z;
      reference_states(3, i) = reference_msg->velocity.x;
      reference_states(4, i) = reference_msg->velocity.y;
      reference_states(5, i) = reference_msg->velocity.z;
      reference_states.block<3, 1>(6, i) = n_eq;
      reference_states.block<3, 1>(9, i) = r_eq;
      reference_states(12, i) = thrust_eq;
      reference_states.block<3, 1>(13, i).setZero();

      // Inputs are [tension_dot, r_ddot] references.
      reference_inputs.col(i).setZero();
    }
    RCLCPP_WARN_THROTTLE(this->get_logger(), clock_, 1000,
                         "[PayloadOlanner] Obtaining only one point reference");
  } else {
    for (int i = 0; i < kSamples; ++i) {
      const size_t idx = std::min(static_cast<size_t>(i), n_points - 1U);
      const auto &point = reference_msg->points[idx];

      reference_states(0, i) = point.position.x;
      reference_states(1, i) = point.position.y;
      reference_states(2, i) = point.position.z;
      reference_states(3, i) = point.velocity.x;
      reference_states(4, i) = point.velocity.y;
      reference_states(5, i) = point.velocity.z;

      reference_states(6, i) = point.cable_direction.x;
      reference_states(7, i) = point.cable_direction.y;
      reference_states(8, i) = point.cable_direction.z;

      reference_states(9, i) = point.cable_r.x;
      reference_states(10, i) = point.cable_r.y;
      reference_states(11, i) = point.cable_r.z;

      reference_states(12, i) = point.tension;

      reference_states(13, i) = point.cable_r_dot.x;
      reference_states(14, i) = point.cable_r_dot.y;
      reference_states(15, i) = point.cable_r_dot.z;

      reference_inputs.col(i).setZero();
    }
  }

  controller_.setReferenceStates(reference_states);
  controller_.setReferenceInputs(reference_inputs);
  if (use_nmpc_payload_) {
    run();
  } else {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "[NMPC Payload Planner] Waiting to switch to payload planner ");
  }
}

void NMPCControlNodelet::run() {
  const int acados_status = controller_.run();

  switch (acados_status) {
  case 1:
    if (_aux_initial) {
      RCLCPP_WARN(
          this->get_logger(),
          "[NMPC Payload Planner] acados failure: could not find a solution.");
      _optimization_error = true;
      return;
    }
    break;
  case 2:
    RCLCPP_WARN(
        this->get_logger(),
        "[NMPC Payload Planner] acados maxiter: maximum iterations reached.");
    _optimization_error = true;
    return;
  case 3:
    RCLCPP_WARN(
        this->get_logger(),
        "[NMPC Payload Planner] acados minstep: minimum QP step reached.");
    _optimization_error = true;
    return;
  case 4:
    RCLCPP_WARN(this->get_logger(),
                "[NMPC Payload Planner] acados qp failure.");
    _optimization_error = true;
    return;
  default:
    break;
  }

  const Eigen::Matrix<double, kStateSize, 1> pred_state =
      controller_.getPredictedState();
  const Eigen::Matrix<double, kInputSize, 1> pred_input =
      controller_.getPredictedInput();

  if (!pred_state.allFinite() || !pred_input.allFinite()) {
    RCLCPP_WARN(this->get_logger(),
                "[NMPC Payload Planner] NaN/Inf in current solution.");
    _optimization_error = true;
    _aux_initial = true;
    return;
  }

  _optimization_error = false;
  _aux_initial = false;
  publishPrediction();
  publishReference();
  publishDesiredQuadrotorCommand();
}

void NMPCControlNodelet::publishPrediction() {
  const auto predicted_states = controller_.getPredictedStates();

  nav_msgs::msg::Path path_msg;
  path_msg.header.stamp = this->now();
  path_msg.header.frame_id = frame_id_;
  path_msg.poses.reserve(kSamples);

  for (int i = 0; i < kSamples; ++i) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path_msg.header;
    pose.pose.position.x = predicted_states(0, i);
    pose.pose.position.y = predicted_states(1, i);
    pose.pose.position.z = predicted_states(2, i);
    pose.pose.orientation.w = 1.0;
    path_msg.poses.push_back(pose);
  }

  pub_pred_traj_->publish(path_msg);
}

void NMPCControlNodelet::publishReference() {
  const auto reference_states = controller_.getReferenceStates();

  nav_msgs::msg::Path path_msg;
  path_msg.header.stamp = this->now();
  path_msg.header.frame_id = frame_id_;
  path_msg.poses.reserve(kSamples);

  for (int i = 0; i < kSamples; ++i) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path_msg.header;
    pose.pose.position.x = reference_states(0, i);
    pose.pose.position.y = reference_states(1, i);
    pose.pose.position.z = reference_states(2, i);
    pose.pose.orientation.w = 1.0;
    path_msg.poses.push_back(pose);
  }
  pub_ref_traj_->publish(path_msg);
}

void NMPCControlNodelet::publishDesiredQuadrotorCommand() {
  const auto predicted_states = controller_.getPredictedStates();

  quadrotor_msgs::msg::PositionCommand position_cmd_msg;
  position_cmd_msg.header.stamp = this->now();
  position_cmd_msg.header.frame_id = frame_id_;
  position_cmd_msg.planner_type =
      quadrotor_msgs::msg::PositionCommand::PAYLOAD_PLANNER;
  position_cmd_msg.yaw = 0.0;
  position_cmd_msg.yaw_dot = 0.0;

  const int first_state_idx = std::min(1, kSamples - 1);
  const Eigen::Vector3d quad_pos =
      quadrotorPositionFromPayloadState(predicted_states.col(first_state_idx));
  const Eigen::Vector3d quad_vel =
      quadrotorVelocityFromPayloadState(predicted_states.col(first_state_idx));
  const Eigen::Vector3d quad_acc = quadrotorAccelerationFromPayloadState(
      predicted_states.col(first_state_idx));
  const double tension = predicted_states(12, first_state_idx);
  const Eigen::Vector3d direction =
      predicted_states.col(first_state_idx).segment<3>(6);
  const Eigen::Vector3d cable_force = tension * direction;

  position_cmd_msg.position.x = quad_pos(0);
  position_cmd_msg.position.y = quad_pos(1);
  position_cmd_msg.position.z = quad_pos(2);

  position_cmd_msg.velocity.x = quad_vel(0);
  position_cmd_msg.velocity.y = quad_vel(1);
  position_cmd_msg.velocity.z = quad_vel(2);

  position_cmd_msg.acceleration.x = quad_acc(0);
  position_cmd_msg.acceleration.y = quad_acc(1);
  position_cmd_msg.acceleration.z = quad_acc(2);

  position_cmd_msg.cable_force.x = cable_force(0);
  position_cmd_msg.cable_force.y = cable_force(1);
  position_cmd_msg.cable_force.z = cable_force(2);

  position_cmd_msg.points.reserve(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    const int state_idx = std::min(i + 1, kSamples - 1);
    const auto state_i = predicted_states.col(state_idx);

    const Eigen::Vector3d quad_position =
        quadrotorPositionFromPayloadState(state_i);
    const Eigen::Vector3d quad_velocity =
        quadrotorVelocityFromPayloadState(state_i);
    const Eigen::Vector3d quad_acceleration =
        quadrotorAccelerationFromPayloadState(state_i);

    quadrotor_msgs::msg::TrajectoryPoint point;
    point.position.x = quad_position(0);
    point.position.y = quad_position(1);
    point.position.z = quad_position(2);
    point.velocity.x = quad_velocity(0);
    point.velocity.y = quad_velocity(1);
    point.velocity.z = quad_velocity(2);
    point.acceleration.x = quad_acceleration(0);
    point.acceleration.y = quad_acceleration(1);
    point.acceleration.z = quad_acceleration(2);
    point.tension = state_i(12);
    point.cable_r_dot.x = state_i(13);
    point.cable_r_dot.y = state_i(14);
    point.cable_r_dot.z = state_i(15);
    position_cmd_msg.points.push_back(point);
  }

  pub_desired_quadrotor_->publish(position_cmd_msg);
}

Eigen::Vector3d NMPCControlNodelet::quadrotorPositionFromPayloadState(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> &state) const {
  const Eigen::Vector3d payload_position = state.segment<3>(0);
  const Eigen::Vector3d cable_direction = state.segment<3>(6);
  return payload_position - (cable_length_ * cable_direction);
}

Eigen::Vector3d NMPCControlNodelet::quadrotorVelocityFromPayloadState(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> &state) const {
  const Eigen::Vector3d payload_velocity = state.segment<3>(3);
  const Eigen::Vector3d cable_direction = state.segment<3>(6);
  const Eigen::Vector3d cable_angular_velocity = state.segment<3>(9);
  return payload_velocity -
         cable_length_ * cable_angular_velocity.cross(cable_direction);
}

Eigen::Vector3d NMPCControlNodelet::quadrotorAccelerationFromPayloadState(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> &state) const {
  const Eigen::Vector3d cable_direction = state.segment<3>(6);
  const Eigen::Vector3d cable_angular_velocity = state.segment<3>(9);
  const double tension = state(12);
  const Eigen::Vector3d cable_angular_acceleration = state.segment<3>(13);

  const double safe_mass = mass_payload_;
  const Eigen::Vector3d e3(0.0, 0.0, 1.0);
  const Eigen::Vector3d payload_linear_acceleration =
      -(tension / safe_mass) * cable_direction - gravity_ * e3;
  const Eigen::Vector3d input_angular_acc_cable =
      -cable_length_ * cable_angular_acceleration.cross(cable_direction);
  const Eigen::Vector3d cable_angular_velocity_aux =
      cable_angular_velocity.cross(cable_direction);
  const Eigen::Vector3d angular_velocity_cable =
      -cable_length_ * cable_angular_velocity.cross(cable_angular_velocity_aux);
  return payload_linear_acceleration + input_angular_acc_cable +
         angular_velocity_cable;
}

} // namespace planner_payload_nodelet

RCLCPP_COMPONENTS_REGISTER_NODE(planner_payload_nodelet::NMPCControlNodelet)
