#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from casadi import Function
from scipy.spatial.transform import Rotation as R
import time
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker
import yaml
import os
import sys

def yaml_to_dict(path_to_yaml):
  with open(path_to_yaml, 'r') as stream:
    try:
      parsed_yaml = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  if '/**' in parsed_yaml:
    parsed_yaml = parsed_yaml['/**']['ros__parameters']
  return parsed_yaml

class PayloadControlMujocoNode():
    def __init__(self, params):
        self.weight_cable_direction = float(10)
        self.weight_tension = float(10)
        self.weight_rdot = float(10)
        #self.norm_constraint_slack_weight = float(0.1)
        #self.unit_vector_norm_tol = float(1e-3)

        # Time Definition
        self.final = 15

        # Prediction nodes and nonuniform time grid for NMPC
        self.t_N = params['nmpc']['horizon_time']
        self.N_prediction = params['nmpc']['horizon_steps']

        self.ts = self.t_N/self.N_prediction
        #self.time_step_growth = 3.0  # Last step is ~3x the first before normalization.
        #step_profile = np.linspace(1.0, self.time_step_growth, self.N_prediction)
        #self.t_steps = self.t_N * step_profile / np.sum(step_profile)
        #self.shooting_nodes = np.concatenate(([0.0], np.cumsum(self.t_steps)))
        #self.ts = float(self.t_steps[0])
        print("Check time parameters")
        print(self.ts)
        print(self.N_prediction)
        print(self.t_N)
        #print(self.t_steps)

        # Internal parameters defintion
        self.robot_num = 1
        self.mass = params['mass_payload']
        self.gravity = params['gravity']


        # Control gains
        c1 = 1
        kp_min = 100
        self.kp_min = kp_min
        self.kv_min = 10
        self.c1 = c1
        
        # Cable length
        self.length = params['cable_length']
        self.e3 = ca.DM([0, 0, 1])

        print("Check payload mass, gravity and cable length parameters")
        print(self.mass)
        print(self.gravity)
        print(self.length)

        # Position of the system payload
        pos_0 = np.array([0.0, 0.0, 0.47], dtype=np.double)
        # Linear velocity of the payload
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        
        # Initial Wrench
        # This is just the mass of the payload an gravity
        Wrench0 = np.array([0, 0, (self.mass)*self.gravity])

        # Init Tension of the cables so we can get initial cable direction
        # Init state payload
        self.init = np.hstack((pos_0, vel_0))

        # Compute the initial tension based on the the Wrench
        self.tensions_init = np.linalg.norm(Wrench0, axis=0)

        # Compute the cable direction initial condition
        self.n_init = -Wrench0/self.tensions_init

        # Compute the cable initial angular velocity
        self.r_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)

        # Init states for the optimizer: [p, v, n, r, tension, r_dot]
        self.r_dot_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)
        self.x_0 = np.hstack((pos_0, vel_0, self.n_init, self.r_init, np.array([self.tensions_init]), self.r_dot_init))

        # Init control actions (rates): [tension_dot, r_ddot]
        self.u_equilibrium = np.zeros((1 + 3*self.robot_num, ), dtype=np.double)
        
        # check equilibirum
        print(self.x_0)
        print(self.u_equilibrium)

        # Maximum and minimun control actions
        self.tension_min = 0.8*self.tensions_init
        self.tension_max = 5.0*self.tensions_init

        self.r_dot_max = np.array([6.0, 6.0, 6.0]*self.robot_num, dtype=np.double)
        self.r_dot_min = -self.r_dot_max
        self.tension_dot_max = 10.0*self.tensions_init
        self.tension_dot_min = -self.tension_dot_max
        self.r_ddot_max = np.array([12.0, 12.0, 12.0]*self.robot_num, dtype=np.double)
        self.r_ddot_min = -self.r_ddot_max

        # Control bounds are on rates [tension_dot, r_ddot]
        self.u_min =  np.hstack((self.tension_dot_min, self.r_ddot_min))
        self.u_max =  np.hstack((self.tension_dot_max, self.r_ddot_max))

        # Define state dimension and control action
        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]

        self.ocp = self.solver(self.x_0)
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_" + self.ocp.model.name + ".json", build= True, generate= True)


    def payloadModel(self)->AcadosModel:
        # Model Name
        model_name = "planner_payload"

        #position 
        p_x = ca.MX.sym('p_x')
        p_y = ca.MX.sym('p_y')
        p_z = ca.MX.sym('p_z')
        x_p = ca.vertcat(p_x, p_y, p_z)
    
        #linear vel
        vx_p = ca.MX.sym("vx_p")
        vy_p = ca.MX.sym("vy_p")
        vz_p = ca.MX.sym("vz_p")   
        v_p = ca.vertcat(vx_p, vy_p, vz_p)

        # Cable kinematics
        nx_1 = ca.MX.sym('nx_1')
        ny_1 = ca.MX.sym('ny_1')
        nz_1 = ca.MX.sym('nz_1')
        n1 = ca.vertcat(nx_1, ny_1, nz_1)

        # Cable kinematics
        rx_1 = ca.MX.sym('rx_1')
        ry_1 = ca.MX.sym('ry_1')
        rz_1 = ca.MX.sym('rz_1')
        r1 = ca.vertcat(rx_1, ry_1, rz_1)
        
        # Augmented states: [payload dynamics, tension, cable angular acceleration state]
        t_1 = ca.MX.sym("t_1")
        rdotx_1 = ca.MX.sym("rdotx_1")
        rdoty_1 = ca.MX.sym("rdoty_1")
        rdotz_1 = ca.MX.sym("rdotz_1")
        r1_dot = ca.vertcat(rdotx_1, rdoty_1, rdotz_1)

        # Full states of the system (16 x 1)
        x = ca.vertcat(x_p, v_p, n1, r1, t_1, r1_dot)
        
        # Control actions are rates: [tension_dot, r_ddot]
        t_1_dot_cmd = ca.MX.sym("t_1_dot_cmd")
        rddotx_1_cmd = ca.MX.sym("rddotx_1_cmd")
        rddoty_1_cmd = ca.MX.sym("rddoty_1_cmd")
        rddotz_1_cmd = ca.MX.sym("rddotz_1_cmd")

        r1_ddot_cmd = ca.vertcat(rddotx_1_cmd, rddoty_1_cmd, rddotz_1_cmd)

        # Vector of control actions
        u = ca.vertcat(t_1_dot_cmd, rddotx_1_cmd, rddoty_1_cmd, rddotz_1_cmd)

        # Linear Dynamics
        linear_velocity = v_p
        cross_angular_payload = ca.cross(r1, n1)
        linear_acceleration = -(1/(self.mass))*t_1*n1 - self.gravity*self.e3

        # Angular dynamics
        # Cable Kinematics
        n1_dot = ca.cross(r1, n1)

        r1_ddot = r1_ddot_cmd

        # Explicit Dynamics
        f_expl = ca.vertcat(linear_velocity, linear_acceleration, n1_dot, r1_dot, t_1_dot_cmd, r1_ddot)

        nx = x.shape[0]
        nu = u.shape[0]

        x_dot = ca.MX.sym('x_dot', nx, 1)

        f_implicit_expr = x_dot - f_expl

        ref_params = ca.MX.sym('ref_params', nx + nu, 1)
        cost_params = ca.MX.sym('cost_params', nx + nx + nu, 1)

        # Dynamics
        model = AcadosModel()
        model.x = x
        # acados_template expects `xdot`; keep both for compatibility across versions.
        model.xdot = x_dot
        model.x_dot = x_dot
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_implicit_expr
        model.u = u
        model.p = ca.vertcat(ref_params, cost_params)
        model.name = model_name
        return model

    def solver(self, x0):
        # get dynamical model
        model = self.payloadModel()
        
        # Optimal control problem
        ocp = AcadosOcp()
        ocp.model = model

        # Get size of the system
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL" 

        # some variables
        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p

        # Split states of the system
        x_p = x[0:3]
        v_p = x[3:6]
        n1 = x[6:9]
        r1 = x[9:12]
        t_1 = x[12]
        r1_dot = x[13:16]

        # Split control actions (rates)
        t_dot_cmd = u[0]
        r_ddot_cmd = u[1:4]

        # Get desired states of the system
        x_p_d = p[0:3]
        v_p_d = p[3:6]
        n1_d = p[6:9]
        r1_d = p[9:12]
        
        # Desired Control Actions 
        t_d = p[12]
        r_dot_d = p[13:16]
        
        # Error of linear dynamics
        error_position_quad = x_p - x_p_d
        error_velocity_quad = v_p - v_p_d

        # Cost functions
        lyapunov_position = 100*(1/2)*self.kp_min*error_position_quad.T@error_position_quad + self.kv_min*(1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad

        # Error cable direction
        error_n1 = ca.cross(n1_d, n1)
        I = ca.MX.eye(3)

        r_error = r1 - (I - n1@n1.T)@r1_d
        r_dot_error = r1_dot - (I - n1@n1.T)@r_dot_d

        # Cost Function control actions
        tension_error = t_d - t_1
        
        # Enforce the velocity is orthogonal
        orthogonality_error = ca.dot(n1, r1)

        ocp.model.cost_expr_ext_cost = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_cable_direction * (r_error.T @ r_error)
            + self.weight_tension * (tension_error * tension_error)
            + self.weight_rdot * (r_dot_error.T @ r_dot_error)
            + 0.2 * (t_dot_cmd * t_dot_cmd)
            + 0.2 * (r_ddot_cmd.T @ r_ddot_cmd)
        )
        ocp.model.cost_expr_ext_cost_e = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_cable_direction * (r_error.T @ r_error)
            + self.weight_tension * (tension_error * tension_error)
            + self.weight_rdot * (r_dot_error.T @ r_dot_error)
        )

        ref_params = np.hstack((self.x_0, self.u_equilibrium))
        cost_params = np.zeros((nx + nx + nu, ), dtype=np.double)
        ocp.parameter_values = np.concatenate([ref_params, cost_params])

        ocp.constraints.constr_type = 'BGH'

        # Set constraints
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        # Keep augmented tension and r_dot states within physical limits.
        #ocp.constraints.idxbx = np.array([12, 13, 14, 15], dtype=np.int32)
        #ocp.constraints.lbx = np.hstack((self.tension_min, self.r_dot_min))
        #ocp.constraints.ubx = np.hstack((self.tension_max, self.r_dot_max))
        ocp.constraints.x0 = x0

        # Softly enforce ||n1|| ~= 1 to improve robustness against numerical drift.
        #ocp.model.con_h_expr = ca.vertcat(ca.dot(n1, n1))
        #nh = 1
        #nsbx = 0
        #nsh = nh
        #ns = nsh + nsbx
        #ocp.cost.zl = self.norm_constraint_slack_weight * np.ones((ns, ))
        #ocp.cost.Zl = self.norm_constraint_slack_weight * np.ones((ns, ))
        #ocp.cost.zu = self.norm_constraint_slack_weight * np.ones((ns, ))
        #ocp.cost.Zu = self.norm_constraint_slack_weight * np.ones((ns, ))
        #ocp.constraints.lh = np.array([1.0 - self.unit_vector_norm_tol])
        #ocp.constraints.uh = np.array([1.0 + self.unit_vector_norm_tol])
        #ocp.constraints.lsh = np.zeros((nsh, ))
        #ocp.constraints.ush = np.zeros((nsh, ))
        #ocp.constraints.idxsh = np.array(range(nsh), dtype=np.int32)

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
        ocp.solver_options.qp_solver_cond_N = self.N_prediction
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  

        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.sim_method_num_stages = 4  # IRK-GL4: 4 stages for accuracy
        ocp.solver_options.sim_method_num_steps = 2  # Number of integration steps
        ocp.solver_options.sim_method_newton_iter = 20  # Newton iterations for convergence
        ocp.solver_options.sim_method_newton_tol = 1e-10  # Newton iterations for convergence

        #ocp.solver_options.time_steps = self.t_steps
        #ocp.solver_options.shooting_nodes = self.shooting_nodes
        ocp.solver_options.levenberg_marquardt = 1e-6

        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 2
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N
        ocp.solver_options.N_horizon = self.N_prediction
        ocp.solver_options.regularize_method = 'NO_REGULARIZE'  
        #ocp.solver_options.levenberg_marquardt = 10.0

        ocp.solver_options.timeout_max_time = 1*1e-3
        ocp.solver_options.timeout_heuristic = "ZERO"
        return ocp


def main(params):
    payload_node = PayloadControlMujocoNode(params)
    return None

if __name__ == '__main__':
    path_to_yaml = os.path.abspath(sys.argv[1])
    params = yaml_to_dict(path_to_yaml)
    print(params)
    main(params)
