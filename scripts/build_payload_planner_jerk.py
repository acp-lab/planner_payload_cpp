#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import casadi as ca
import numpy as np
import yaml
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def yaml_to_dict(path_to_yaml):
    with open(path_to_yaml, "r", encoding="utf-8") as stream:
        parsed_yaml = yaml.safe_load(stream)
    if "/**" in parsed_yaml:
        parsed_yaml = parsed_yaml["/**"]["ros__parameters"]
    return parsed_yaml


class PayloadPlannerJerkBuilder:
    def __init__(self, params):
        self.weight_cable_direction = 1.0
        self.weight_r = 1.0
        self.weight_accel = 1.0
        self.weight_jerk = 0.1
        self.weight_orthogonality = 1.0
        self.norm_constraint_slack_weight = 10.0
        self.unit_vector_norm_tol = 1e-3

        self.t_N = float(params["nmpc"]["horizon_time"])
        self.N_prediction = int(params["nmpc"]["horizon_steps"])
        self.ts = self.t_N / self.N_prediction

        self.mass = float(params["mass_payload"])
        self.gravity = float(params["gravity"])
        self.length = float(params["cable_length"])
        self.e3 = ca.DM([0.0, 0.0, 1.0])

        self.kp_min = 100.0
        self.kv_min = 5.0

        pos_0 = np.array([0.0, 0.0, 0.47], dtype=np.double)
        vel_0 = np.zeros((3,), dtype=np.double)
        wrench_0 = np.array([0.0, 0.0, self.mass * self.gravity], dtype=np.double)
        tension_0 = np.linalg.norm(wrench_0)
        self.tension_min = 0.1 * tension_0
        self.tension_max = 10.0 * tension_0
        n_init = -wrench_0 / tension_0
        r_init = np.zeros((3,), dtype=np.double)
        a_q_init = np.zeros((3,), dtype=np.double)

        self.x_0 = np.hstack((pos_0, vel_0, n_init, r_init, a_q_init))
        self.u_equilibrium = np.zeros((3,), dtype=np.double)

        self.jerk_max = np.array([120.0, 120.0, 120.0], dtype=np.double)
        self.jerk_min = -self.jerk_max

        self.u_min = self.jerk_min.copy()
        self.u_max = self.jerk_max.copy()

        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]

        # Print Nominalm states
        print("Nominal states and control actions")
        print(self.x_0)
        print(self.u_equilibrium)

        print("Max and Min tension internal Constraints")
        print(self.tension_min)
        print(self.tension_max)

        print("Wrench on the cable")
        print(wrench_0)

        print("Tension on the cable")
        print(tension_0)

        self.project_root = Path(__file__).resolve().parents[1]
        self.code_export_directory = self.project_root / "c_generated_code_jerk"
        self.json_file = self.project_root / "acados_ocp_planner_payload_jerk.json"

        self.ocp = self.solver(self.x_0)

        AcadosOcpSolver(self.ocp, build=True, generate=True)

    def payloadModel(self) -> AcadosModel:
        model_name = "planner_payload_jerk"

        p_x = ca.MX.sym("p_x")
        p_y = ca.MX.sym("p_y")
        p_z = ca.MX.sym("p_z")
        x_p = ca.vertcat(p_x, p_y, p_z)

        vx_p = ca.MX.sym("vx_p")
        vy_p = ca.MX.sym("vy_p")
        vz_p = ca.MX.sym("vz_p")
        v_p = ca.vertcat(vx_p, vy_p, vz_p)

        nx_1 = ca.MX.sym("nx_1")
        ny_1 = ca.MX.sym("ny_1")
        nz_1 = ca.MX.sym("nz_1")
        n1 = ca.vertcat(nx_1, ny_1, nz_1)

        rx_1 = ca.MX.sym("rx_1")
        ry_1 = ca.MX.sym("ry_1")
        rz_1 = ca.MX.sym("rz_1")
        r1 = ca.vertcat(rx_1, ry_1, rz_1)

        ax_q = ca.MX.sym("ax_q")
        ay_q = ca.MX.sym("ay_q")
        az_q = ca.MX.sym("az_q")
        a_q = ca.vertcat(ax_q, ay_q, az_q)

        x = ca.vertcat(x_p, v_p, n1, r1, a_q)

        jx_q = ca.MX.sym("jx_q")
        jy_q = ca.MX.sym("jy_q")
        jz_q = ca.MX.sym("jz_q")
        j_q = ca.vertcat(jx_q, jy_q, jz_q)
        u = j_q

        linear_velocity = v_p
        gravity_vec = self.gravity * self.e3
        linear_acceleration = (
            -gravity_vec
            + n1 * ca.dot(n1, (a_q + gravity_vec))
            - self.length * ca.dot(r1, r1) * n1
        )
        n1_dot = ca.cross(r1, n1)
        r1_dot = -(1.0 / self.length) * ca.cross(n1, (a_q + gravity_vec))
        a_q_dot = j_q

        f_expl = ca.vertcat(
            linear_velocity, linear_acceleration, n1_dot, r1_dot, a_q_dot
        )

        nx = x.shape[0]
        x_dot = ca.MX.sym("x_dot", nx, 1)
        f_impl_expr = x_dot - f_expl

        ref_params = ca.MX.sym("ref_params", nx + u.shape[0], 1)
        cost_params = ca.MX.sym("cost_params", nx + nx + u.shape[0], 1)

        model = AcadosModel()
        model.x = x
        model.xdot = x_dot
        model.x_dot = x_dot
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl_expr
        model.u = u
        model.p = ca.vertcat(ref_params, cost_params)
        model.name = model_name
        return model

    def solver(self, x0):
        model = self.payloadModel()

        ocp = AcadosOcp()
        ocp.model = model
        ocp.name = model.name
        ocp.code_gen_options.code_export_directory = str(self.code_export_directory)
        ocp.code_gen_options.json_file = self.json_file.name

        nx = model.x.size()[0]
        nu = model.u.size()[0]

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p

        # Split states of the system
        x_p = x[0:3]
        v_p = x[3:6]
        n1 = x[6:9]
        r1 = x[9:12]
        a_q = x[12:15]

        # Control actions of the system
        j_q = u[0:3]

        # Split desired states of the system
        x_p_d = p[0:3]
        v_p_d = p[3:6]
        n1_d = p[6:9]
        r1_d = p[9:12]
        a_q_d = p[12:15]

        # Desired Jerk of the quadrotor
        j_q_d = p[15:18]

        error_position = x_p - x_p_d
        error_velocity = v_p - v_p_d

        error_n1 = ca.cross(n1_d, n1)

        tangent_projector = ca.MX.eye(3) - n1 @ n1.T

        r_error = r1 - tangent_projector @ r1_d

        a_q_error = a_q_d - a_q
        j_q_error = j_q_d - j_q

        orthogonality_error = ca.dot(n1, r1)
        tension_expr = self.mass * (
            self.length * ca.dot(r1, r1) - ca.dot(n1, (a_q + self.gravity * self.e3))
        )

        self.Q = ca.MX.zeros(3, 3)
        self.Q[0, 0] = 1.0
        self.Q[1, 1] = 1.0
        self.Q[2, 2] = 20.0

        lyapunov_position = 100.0 * self.kp_min * (
            error_position.T @ self.Q @ error_position
        ) + 0.5 * self.kv_min * self.mass * (error_velocity.T @ error_velocity)

        ocp.model.cost_expr_ext_cost = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_r * (r_error.T @ r_error)
            + self.weight_accel * (a_q_error.T @ a_q_error)
            + self.weight_jerk * (j_q_error.T @ j_q_error)
            + self.weight_orthogonality * (orthogonality_error**2)
        )
        ocp.model.cost_expr_ext_cost_e = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_r * (r_error.T @ r_error)
            + self.weight_accel * (a_q_error.T @ a_q_error)
            + self.weight_orthogonality * (orthogonality_error**2)
        )

        ref_params = np.hstack((self.x_0, self.u_equilibrium))
        cost_params = np.zeros((nx + nx + nu,), dtype=np.double)
        ocp.parameter_values = np.concatenate([ref_params, cost_params])

        ocp.constraints.constr_type = "BGH"
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int32)
        ocp.constraints.x0 = x0

        ocp.model.con_h_expr = ca.vertcat(
            ca.dot(n1, n1),
            tension_expr,
        )
        nh = 2
        nsh = nh
        ocp.cost.zl = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.cost.Zl = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.cost.zu = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.cost.Zu = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.constraints.lh = np.array(
            [1.0 - self.unit_vector_norm_tol, self.tension_min],
            dtype=np.double,
        )
        ocp.constraints.uh = np.array(
            [1.0 + self.unit_vector_norm_tol, self.tension_max],
            dtype=np.double,
        )
        ocp.constraints.lsh = np.zeros((nsh,))
        ocp.constraints.ush = np.zeros((nsh,))
        ocp.constraints.idxsh = np.array(range(nsh), dtype=np.int32)

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = self.N_prediction
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.sim_method_newton_iter = 20
        ocp.solver_options.sim_method_newton_tol = 1e-10
        ocp.solver_options.levenberg_marquardt = 1.0
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 2
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N
        ocp.solver_options.N_horizon = self.N_prediction
        ocp.solver_options.regularize_method = "CONVEXIFY"

        return ocp


def main(params):
    PayloadPlannerJerkBuilder(params)
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_payload_planner_jerk.py <config.yaml>")
    path_to_yaml = os.path.abspath(sys.argv[1])
    params = yaml_to_dict(path_to_yaml)
    print(params)
    main(params)
