#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <planner_payload_cpp/nmpc_planner_jerk.h>
#include <quadrotor_msgs/msg/position_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace planner_payload_jerk_nodelet {
class NMPCControlJerkNodelet : public rclcpp::Node {
public:
  NMPCControlJerkNodelet(const rclcpp::NodeOptions &options)
      : Node("nmpc_control_jerk_nodelet", options), frame_id_("world") {
    this->declare_parameter("mass_payload", 0.12);
    this->declare_parameter("gravity", 9.81);
    this->declare_parameter("cable_length", 0.76);
    this->declare_parameter("cable_signal_filter_alpha", 0.2);
    this->declare_parameter<std::vector<double>>(
        "nmpc.Q_jerk", std::vector<double>{210., 210., 210., 1., 1., 1., 5., 5.,
                                           5., 1., 1., 1., 20., 20., 20.});
    this->declare_parameter<std::vector<double>>(
        "nmpc.Q_jerk_e",
        std::vector<double>{210., 210., 210., 1., 1., 1., 5., 5., 5., 1., 1.,
                            1., 20., 20., 20.});
    this->declare_parameter<std::vector<double>>(
        "nmpc.R_jerk", std::vector<double>{0.5, 0.5, 0.5});

    this->get_parameter("mass_payload", mass_payload_);
    this->get_parameter("gravity", gravity_);
    this->get_parameter("cable_length", cable_length_);
    this->get_parameter("cable_signal_filter_alpha",
                        cable_signal_filter_alpha_);

    Q_param_ = this->get_parameter("nmpc.Q_jerk").as_double_array();
    Q_e_param_ = this->get_parameter("nmpc.Q_jerk_e").as_double_array();
    R_param_ = this->get_parameter("nmpc.R_jerk").as_double_array();

    RCLCPP_INFO(this->get_logger(), "[Jerk NMPC] mass_payload: %.4f",
                mass_payload_);
    RCLCPP_INFO(this->get_logger(), "[Jerk NMPC] gravity: %.4f", gravity_);
    RCLCPP_INFO(this->get_logger(), "[Jerk NMPC] cable_length: %.4f",
                cable_length_);
    RCLCPP_INFO(this->get_logger(),
                "[Jerk NMPC] cable_signal_filter_alpha: %.4f",
                cable_signal_filter_alpha_);
    RCLCPP_INFO(this->get_logger(), "[Jerk NMPC] Q_jerk: %s",
                this->get_parameter("nmpc.Q_jerk").value_to_string().c_str());
    RCLCPP_INFO(this->get_logger(), "[Jerk NMPC] Q_jerk_e: %s",
                this->get_parameter("nmpc.Q_jerk_e").value_to_string().c_str());
    RCLCPP_INFO(this->get_logger(), "[Jerk NMPC] R_jerk: %s",
                this->get_parameter("nmpc.R_jerk").value_to_string().c_str());

    controller_.setMass(mass_payload_);
    controller_.setGravity(gravity_);
    controller_.setWeightMatrices(Q_param_, Q_e_param_, R_param_);

    auto qos_profile = rclcpp::SensorDataQoS();
    pub_ref_traj_ = this->create_publisher<nav_msgs::msg::Path>(
        "payload_reference_path_jerk", 1);
    pub_pred_traj_ = this->create_publisher<nav_msgs::msg::Path>(
        "payload_predicted_path_jerk", 1);
    pub_desired_quadrotor_ =
        this->create_publisher<quadrotor_msgs::msg::PositionCommand>(
            "payload_planner_quadrotor_cmd", 1);
    pub_measured_cable_direction_ =
        this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
            "measured_cable_direction", 10);
    pub_measured_cable_r_ =
        this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
            "measured_cable_angular_velocity", 10);
    pub_filtered_cable_direction_ =
        this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
            "filtered_cable_direction", 10);
    pub_filtered_cable_r_ =
        this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
            "filtered_cable_angular_velocity", 10);

    sub_payload_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "payload/odom", qos_profile,
        std::bind(&NMPCControlJerkNodelet::payloadOdomCallback, this,
                  std::placeholders::_1));
    sub_quad_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "odom", qos_profile,
        std::bind(&NMPCControlJerkNodelet::quadOdomCallback, this,
                  std::placeholders::_1));
    sub_position_cmd_ =
        this->create_subscription<quadrotor_msgs::msg::PositionCommand>(
            "position_cmd", 1,
            std::bind(&NMPCControlJerkNodelet::referenceCallback, this,
                      std::placeholders::_1));
    srv_activate_payload_ = this->create_service<std_srvs::srv::SetBool>(
        "activate_payload_nmpc_controller",
        std::bind(&NMPCControlJerkNodelet::activate_payload_callback, this,
                  std::placeholders::_1, std::placeholders::_2));
  }

private:
  static Eigen::Vector3d vector3(const geometry_msgs::msg::Vector3 &msg) {
    return Eigen::Vector3d(msg.x, msg.y, msg.z);
  }

  static Eigen::Vector3d point3(const geometry_msgs::msg::Point &msg) {
    return Eigen::Vector3d(msg.x, msg.y, msg.z);
  }

  void publishMeasuredCableSignals(const builtin_interfaces::msg::Time &stamp,
                                   const Eigen::Vector3d &cable_direction,
                                   const Eigen::Vector3d &cable_r) {
    geometry_msgs::msg::Vector3Stamped direction_msg;
    direction_msg.header.stamp = stamp;
    direction_msg.header.frame_id = frame_id_;
    direction_msg.vector.x = cable_direction(0);
    direction_msg.vector.y = cable_direction(1);
    direction_msg.vector.z = cable_direction(2);
    pub_measured_cable_direction_->publish(direction_msg);

    geometry_msgs::msg::Vector3Stamped cable_r_msg;
    cable_r_msg.header.stamp = stamp;
    cable_r_msg.header.frame_id = frame_id_;
    cable_r_msg.vector.x = cable_r(0);
    cable_r_msg.vector.y = cable_r(1);
    cable_r_msg.vector.z = cable_r(2);
    pub_measured_cable_r_->publish(cable_r_msg);

    if (!filtered_cable_initialized_) {
      filtered_cable_direction_ = cable_direction;
      filtered_cable_r_ = cable_r;
      filtered_cable_initialized_ = true;
    } else {
      filtered_cable_direction_ =
          (1.0 - cable_signal_filter_alpha_) * filtered_cable_direction_ +
          cable_signal_filter_alpha_ * cable_direction;
      const double dir_norm = filtered_cable_direction_.norm();
      if (dir_norm > 1e-6) {
        filtered_cable_direction_ /= dir_norm;
      } else {
        filtered_cable_direction_ = Eigen::Vector3d(0.0, 0.0, -1.0);
      }
      filtered_cable_r_ =
          (1.0 - cable_signal_filter_alpha_) * filtered_cable_r_ +
          cable_signal_filter_alpha_ * cable_r;
    }

    geometry_msgs::msg::Vector3Stamped filtered_direction_msg;
    filtered_direction_msg.header = direction_msg.header;
    filtered_direction_msg.vector.x = filtered_cable_direction_(0);
    filtered_direction_msg.vector.y = filtered_cable_direction_(1);
    filtered_direction_msg.vector.z = filtered_cable_direction_(2);
    pub_filtered_cable_direction_->publish(filtered_direction_msg);

    geometry_msgs::msg::Vector3Stamped filtered_cable_r_msg;
    filtered_cable_r_msg.header = cable_r_msg.header;
    filtered_cable_r_msg.vector.x = filtered_cable_r_(0);
    filtered_cable_r_msg.vector.y = filtered_cable_r_(1);
    filtered_cable_r_msg.vector.z = filtered_cable_r_(2);
    pub_filtered_cable_r_->publish(filtered_cable_r_msg);
  }

  Eigen::Vector3d quadrotorPositionFromPayloadState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> &state)
      const {
    return state.segment<3>(0) - cable_length_ * state.segment<3>(6);
  }

  Eigen::Vector3d quadrotorVelocityFromPayloadState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> &state)
      const {
    return state.segment<3>(3) -
           cable_length_ * state.segment<3>(9).cross(state.segment<3>(6));
  }

  Eigen::Vector3d quadrotorAccelerationFromPayloadState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> &state)
      const {
    return state.segment<3>(12);
  }

  Eigen::Vector3d cableAngularAccelerationFromState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> &state)
      const {
    const Eigen::Vector3d e3(0.0, 0.0, 1.0);
    return -(1.0 / cable_length_) *
           state.segment<3>(6).cross(state.segment<3>(12) + gravity_ * e3);
  }

  double tensionFromState(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> &state)
      const {
    const Eigen::Vector3d e3(0.0, 0.0, 1.0);
    const Eigen::Vector3d n = state.segment<3>(6);
    const Eigen::Vector3d omega = state.segment<3>(9);
    const Eigen::Vector3d a_q = state.segment<3>(12);
    return mass_payload_ *
           (cable_length_ * omega.squaredNorm() - n.dot(a_q + gravity_ * e3));
  }

  double tensionDotFromStateInput(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> &state,
      const Eigen::Ref<const Eigen::Matrix<double, kInputSizeJerk, 1>> &input)
      const {
    const Eigen::Vector3d e3(0.0, 0.0, 1.0);
    const Eigen::Vector3d n = state.segment<3>(6);
    const Eigen::Vector3d omega = state.segment<3>(9);
    const Eigen::Vector3d a_q = state.segment<3>(12);
    const Eigen::Vector3d j_q = input;
    const Eigen::Vector3d n_dot = omega.cross(n);
    const Eigen::Vector3d omega_dot = cableAngularAccelerationFromState(state);
    return mass_payload_ * (2.0 * cable_length_ * omega.dot(omega_dot) -
                            n_dot.dot(a_q + gravity_ * e3) - n.dot(j_q));
  }

  Eigen::Vector3d cableAngularJerkFromStateInput(
      const Eigen::Ref<const Eigen::Matrix<double, kStateSizeJerk, 1>> &state,
      const Eigen::Ref<const Eigen::Matrix<double, kInputSizeJerk, 1>> &input)
      const {
    const Eigen::Vector3d e3(0.0, 0.0, 1.0);
    const Eigen::Vector3d n = state.segment<3>(6);
    const Eigen::Vector3d omega = state.segment<3>(9);
    const Eigen::Vector3d a_q = state.segment<3>(12);
    const Eigen::Vector3d j_q = input;
    const Eigen::Vector3d n_dot = omega.cross(n);
    return -(1.0 / cable_length_) *
           (n_dot.cross(a_q + gravity_ * e3) + n.cross(j_q));
  }

  void quadOdomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    quad_position_ << odom_msg->pose.pose.position.x,
        odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z;
    quad_velocity_ << odom_msg->twist.twist.linear.x,
        odom_msg->twist.twist.linear.y, odom_msg->twist.twist.linear.z;
  }

  void activate_payload_callback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    use_nmpc_payload_ = request->data;
    response->success = true;
    response->message = use_nmpc_payload_ ? "Payload jerk NMPC active"
                                          : "Payload jerk NMPC inactive";
    RCLCPP_INFO(this->get_logger(), "[Jerk NMPC] %s",
                response->message.c_str());
  }

  void payloadOdomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    Eigen::Matrix<double, kStateSizeJerk, 1> state =
        Eigen::Matrix<double, kStateSizeJerk, 1>::Zero();
    frame_id_ = odom_msg->header.frame_id;

    state(0) = odom_msg->pose.pose.position.x;
    state(1) = odom_msg->pose.pose.position.y;
    state(2) = odom_msg->pose.pose.position.z;

    state(3) = odom_msg->twist.twist.linear.x;
    state(4) = odom_msg->twist.twist.linear.y;
    state(5) = odom_msg->twist.twist.linear.z;

    // Compute cable directions and angular velocity
    const Eigen::Vector3d payload_position = state.segment<3>(0);
    const Eigen::Vector3d payload_velocity = state.segment<3>(3);

    const Eigen::Vector3d a = payload_position - quad_position_;
    const Eigen::Vector3d a_dot = payload_velocity - quad_velocity_;
    const double norm_a = a.norm();
    const double dot_a = std::max(a.dot(a), 1e-12);
    const Eigen::Matrix3d projection =
        Eigen::Matrix3d::Identity() - (a * a.transpose()) / dot_a;
    const Eigen::Vector3d n_dot = (1.0 / norm_a) * projection * a_dot;
    const Eigen::Vector3d cable_dir = a / norm_a;
    const Eigen::Vector3d cable_r = cable_dir.cross(n_dot);
    publishMeasuredCableSignals(odom_msg->header.stamp, cable_dir, cable_r);
    state.segment<3>(6) = filtered_cable_direction_;
    state.segment<3>(9) = filtered_cable_r_;

    // Update acceleration based on the predictions
    const auto predicted_state = controller_.getPredictedState();
    if (predicted_state.allFinite()) {
      state.segment<3>(12) = predicted_state.segment<3>(12);
    }

    const double stamp_sec =
        static_cast<double>(odom_msg->header.stamp.sec) +
        static_cast<double>(odom_msg->header.stamp.nanosec) * 1e-9;
    controller_.setState(state, stamp_sec);
  }

  void referenceCallback(
      const quadrotor_msgs::msg::PositionCommand::SharedPtr reference_msg) {
    Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk> reference_states =
        Eigen::Matrix<double, kStateSizeJerk, kSamplesJerk>::Zero();
    Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk> reference_inputs =
        Eigen::Matrix<double, kInputSizeJerk, kSamplesJerk>::Zero();

    const Eigen::Vector3d n_eq(0.0, 0.0, -1.0);
    const Eigen::Vector3d jerk_eq(0.0, 0.0, 0.0);
    const Eigen::Vector3d r_eq = Eigen::Vector3d::Zero();

    const size_t n_points = reference_msg->points.size();
    for (int i = 0; i < kSamplesJerk; ++i) {
      if (n_points == 0) {
        reference_states(0, i) = reference_msg->position.x;
        reference_states(1, i) = reference_msg->position.y;
        reference_states(2, i) = reference_msg->position.z;
        reference_states(3, i) = reference_msg->velocity.x;
        reference_states(4, i) = reference_msg->velocity.y;
        reference_states(5, i) = reference_msg->velocity.z;
        reference_states.block<3, 1>(6, i) = n_eq;
        reference_states.block<3, 1>(9, i) = r_eq;
        reference_states.block<3, 1>(12, i) =
            vector3(reference_msg->acceleration);
        reference_inputs.col(i) = vector3(reference_msg->jerk);
      } else {
        const auto &point =
            reference_msg
                ->points[std::min(static_cast<size_t>(i), n_points - 1U)];
        reference_states(0, i) = point.position.x;
        reference_states(1, i) = point.position.y;
        reference_states(2, i) = point.position.z;
        reference_states(3, i) = point.velocity.x;
        reference_states(4, i) = point.velocity.y;
        reference_states(5, i) = point.velocity.z;

        Eigen::Vector3d n_ref = vector3(point.cable_direction);
        reference_states.block<3, 1>(6, i) = n_ref;
        reference_states.block<3, 1>(9, i) = vector3(point.cable_r);

        Eigen::Vector3d a_ref = point3(point.acceleration_quad);
        reference_states.block<3, 1>(12, i) = a_ref;

        // we can use the jerk on the quadrotor
        reference_inputs.col(i) = jerk_eq;
      }
    }

    controller_.setReferenceStates(reference_states);
    controller_.setReferenceInputs(reference_inputs);
    if (use_nmpc_payload_) {
      run();
    } else {
      RCLCPP_INFO_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "[Jerk NMPC] Waiting to switch to planning. "
          "Received reference, but use_nmpc_payload_ is false.");
    }
  }

  void run() {
    const int acados_status = controller_.run();
    if (acados_status != 0) {
      RCLCPP_WARN(this->get_logger(), "[Jerk NMPC] acados returned %d",
                  acados_status);
      return;
    }
    publishPrediction();
    publishReference();
    publishDesiredQuadrotorCommand();
  }

  void publishPrediction() {
    const auto predicted_states = controller_.getPredictedStates();
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = frame_id_;
    for (int i = 0; i < kSamplesJerk; ++i) {
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

  void publishReference() {
    const auto reference_states = controller_.getReferenceStates();
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = frame_id_;
    for (int i = 0; i < kSamplesJerk; ++i) {
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

  void publishDesiredQuadrotorCommand() {
    const auto predicted_states = controller_.getPredictedStates();
    const auto predicted_inputs = controller_.getPredictedInputs();

    quadrotor_msgs::msg::PositionCommand msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id_;
    msg.planner_type = quadrotor_msgs::msg::PositionCommand::PAYLOAD_PLANNER;

    const int first_state_idx = std::min(1, kSamplesJerk - 1);
    const auto first_state = predicted_states.col(first_state_idx);
    const auto first_input = predicted_inputs.col(0);
    const Eigen::Vector3d quad_pos =
        quadrotorPositionFromPayloadState(first_state);
    const Eigen::Vector3d quad_vel =
        quadrotorVelocityFromPayloadState(first_state);
    const Eigen::Vector3d quad_acc =
        quadrotorAccelerationFromPayloadState(first_state);
    const Eigen::Vector3d cable_r_dot =
        cableAngularAccelerationFromState(first_state);
    const double tension = tensionFromState(first_state);
    const Eigen::Vector3d cable_force = tension * first_state.segment<3>(6);

    msg.position.x = quad_pos(0);
    msg.position.y = quad_pos(1);
    msg.position.z = quad_pos(2);
    msg.velocity.x = quad_vel(0);
    msg.velocity.y = quad_vel(1);
    msg.velocity.z = quad_vel(2);
    msg.acceleration.x = quad_acc(0);
    msg.acceleration.y = quad_acc(1);
    msg.acceleration.z = quad_acc(2);
    msg.jerk.x = first_input(0);
    msg.jerk.y = first_input(1);
    msg.jerk.z = first_input(2);
    msg.tension = tension;
    msg.cable_direction.x = first_state(6);
    msg.cable_direction.y = first_state(7);
    msg.cable_direction.z = first_state(8);
    msg.cable_r_dot.x = cable_r_dot(0);
    msg.cable_r_dot.y = cable_r_dot(1);
    msg.cable_r_dot.z = cable_r_dot(2);
    msg.cable_force.x = cable_force(0);
    msg.cable_force.y = cable_force(1);
    msg.cable_force.z = cable_force(2);

    for (int i = 0; i < kSamplesJerk; ++i) {
      const int state_idx = std::min(i + 1, kSamplesJerk - 1);
      const int input_idx = std::min(i, kSamplesJerk - 1);
      const auto state_i = predicted_states.col(state_idx);
      const auto input_i = predicted_inputs.col(input_idx);
      const Eigen::Vector3d quad_position =
          quadrotorPositionFromPayloadState(state_i);
      const Eigen::Vector3d quad_velocity =
          quadrotorVelocityFromPayloadState(state_i);
      const Eigen::Vector3d quad_acceleration =
          quadrotorAccelerationFromPayloadState(state_i);
      const Eigen::Vector3d quad_jerk = input_i;
      const Eigen::Vector3d cable_r_dot_i =
          cableAngularAccelerationFromState(state_i);
      const Eigen::Vector3d cable_r_dot_dot_i =
          cableAngularJerkFromStateInput(state_i, input_i);

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
      point.jerk.x = quad_jerk(0);
      point.jerk.y = quad_jerk(1);
      point.jerk.z = quad_jerk(2);
      point.cable_direction.x = state_i(6);
      point.cable_direction.y = state_i(7);
      point.cable_direction.z = state_i(8);
      point.cable_r.x = state_i(9);
      point.cable_r.y = state_i(10);
      point.cable_r.z = state_i(11);
      point.tension = tensionFromState(state_i);
      point.cable_r_dot.x = cable_r_dot_i(0);
      point.cable_r_dot.y = cable_r_dot_i(1);
      point.cable_r_dot.z = cable_r_dot_i(2);
      point.tension_dot = tensionDotFromStateInput(state_i, input_i);
      point.cable_r_dot_dot.x = cable_r_dot_dot_i(0);
      point.cable_r_dot_dot.y = cable_r_dot_dot_i(1);
      point.cable_r_dot_dot.z = cable_r_dot_dot_i(2);
      msg.points.push_back(point);
    }

    pub_desired_quadrotor_->publish(msg);
  }

  NMPCControlJerk controller_;
  std::string frame_id_;
  double mass_payload_{0.1128};
  double gravity_{9.81};
  double cable_length_{0.88};
  double cable_signal_filter_alpha_{0.2};
  std::vector<double> Q_param_;
  std::vector<double> Q_e_param_;
  std::vector<double> R_param_;
  Eigen::Vector3d quad_position_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d quad_velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d filtered_cable_direction_{Eigen::Vector3d(0.0, 0.0, -1.0)};
  Eigen::Vector3d filtered_cable_r_{Eigen::Vector3d::Zero()};
  bool filtered_cable_initialized_{false};
  bool use_nmpc_payload_{false};

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_ref_traj_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_pred_traj_;
  rclcpp::Publisher<quadrotor_msgs::msg::PositionCommand>::SharedPtr
      pub_desired_quadrotor_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr
      pub_measured_cable_direction_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr
      pub_measured_cable_r_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr
      pub_filtered_cable_direction_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr
      pub_filtered_cable_r_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      sub_payload_odometry_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_quad_odometry_;
  rclcpp::Subscription<quadrotor_msgs::msg::PositionCommand>::SharedPtr
      sub_position_cmd_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_activate_payload_;
};

} // namespace planner_payload_jerk_nodelet

RCLCPP_COMPONENTS_REGISTER_NODE(
    planner_payload_jerk_nodelet::NMPCControlJerkNodelet)
