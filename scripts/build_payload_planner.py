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

class PayloadControlMujocoNode():
    def __init__(self):
        self.weight_cable_direction = float(0.1)
        self.weight_tension = float(0.5)
        self.weight_rdot = float(10.0)
        self.weight_orthogonality = float(10.0)
        self.norm_constraint_slack_weight = float(100.0)
        self.unit_vector_norm_tol = float(1e-3)

        # Time Definition
        self.ts = float(0.05)
        self.final = 15

        # Prediction Node of the NMPC formulation
        self.t_N = float(2.5)
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]
        print(self.N_prediction)

        # Internal parameters defintion
        self.robot_num = 1
        self.mass = 0.11
        self.gravity = 9.81


        # Quadrotor paramaters
        self.mass_quad = 1.05
        self.inertia_quad = np.array([[0.00345398, 0.0, 0.0], [0.0, 0.00179687, 0.0], [0.0, 0.0, 0.00179676]], dtype=np.double)

        # Control gains
        c1 = 1
        kp_min = 100
        self.kp_min = kp_min
        self.kv_min = 5
        self.c1 = c1
        
        # Cable length
        self.length = 0.88
        self.e3 = ca.DM([0, 0, 1])

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

        # Init states for the optimizer
        self.x_0 = np.hstack((pos_0, vel_0, self.n_init, self.r_init))

        # Init Control Actions or equilibirum
        self.r_dot_init = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)
        self.u_equilibrium = np.hstack((self.tensions_init, self.r_dot_init))
        
        # check equilibirum
        print(self.x_0)
        print(self.u_equilibrium)

        # Maximum and minimun control actions
        self.tension_min = 0.8*self.tensions_init
        self.tension_max = 8.0*self.tensions_init

        self.r_dot_max = np.array([6.0, 6.0, 6.0]*self.robot_num, dtype=np.double)
        self.r_dot_min = -self.r_dot_max

        self.u_min =  np.hstack((self.tension_min, self.r_dot_min))
        self.u_max =  np.hstack((self.tension_max, self.r_dot_max))

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
        
        # Full states of the system (12 x 1)
        x = ca.vertcat(x_p, v_p, n1, r1)
        
        # Control actions of the system
        t_1_cmd = ca.MX.sym("t_1_cmd")
        rx_1_cmd = ca.MX.sym("rx_1_cmd")
        ry_1_cmd = ca.MX.sym("ry_1_cmd")
        rz_1_cmd = ca.MX.sym("rz_1_cmd")

        r1_cmd = ca.vertcat(rx_1_cmd, ry_1_cmd, rz_1_cmd) 

        # Vector of control actions
        u = ca.vertcat(t_1_cmd, rx_1_cmd, ry_1_cmd, rz_1_cmd)

        # Linear Dynamics
        linear_velocity = v_p
        cross_angular_payload = ca.cross(r1, n1)
        linear_acceleration = -(1/(self.mass))*t_1_cmd*n1 - self.gravity*self.e3

        # Angular dynamics
        # Cable Kinematics
        n1_dot = ca.cross(r1, n1)

        r1_dot = (r1_cmd)

        # Explicit Dynamics
        f_expl = ca.vertcat(linear_velocity, linear_acceleration, n1_dot, r1_dot)

        nx = x.shape[0]
        nu = u.shape[0]

        ref_params = ca.MX.sym('ref_params', nx + nu, 1)
        cost_params = ca.MX.sym('cost_params', nx + nx + nu, 1)

        # Dynamics
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
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

        # Split control actions
        t_cmd = u[0]
        r_dot_cmd = u[1:4]

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
        r_error = r1_d - r1

        # Cost Function control actions
        tension_error = t_d - t_cmd
        r_dot_error = r_dot_d - r_dot_cmd 
        
        # Enforce the velocity is orthogonal
        orthogonality_error = ca.dot(n1, r1)

        ocp.model.cost_expr_ext_cost = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_tension * (tension_error * tension_error)
            + self.weight_rdot * (r_dot_error.T @ r_dot_error)
            + self.weight_orthogonality * (orthogonality_error**2)
        )
        ocp.model.cost_expr_ext_cost_e = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_orthogonality * (orthogonality_error**2)
        )

        ref_params = np.hstack((self.x_0, self.u_equilibrium))
        cost_params = np.array([210., 210., 210., 1., 1., 1., 5., 5., 5., 1., 1., 1., 210., 210., 210., 1., 1., 1., 5., 5., 5., 1., 1., 1., 0.5, 0.1, 0.1, 0.1])
        ocp.parameter_values = np.concatenate([ref_params, cost_params])

        ocp.constraints.constr_type = 'BGH'

        # Set constraints
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.x0 = x0

        # Softly enforce ||n1|| ~= 1 to improve robustness against numerical drift.
        ocp.model.con_h_expr = ca.vertcat(ca.dot(n1, n1))
        nh = 1
        nsbx = 0
        nsh = nh
        ns = nsh + nsbx
        ocp.cost.zl = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.cost.Zl = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.cost.zu = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.cost.Zu = self.norm_constraint_slack_weight * np.ones((ns, ))
        ocp.constraints.lh = np.array([1.0 - self.unit_vector_norm_tol])
        ocp.constraints.uh = np.array([1.0 + self.unit_vector_norm_tol])
        ocp.constraints.lsh = np.zeros((nsh, ))
        ocp.constraints.ush = np.zeros((nsh, ))
        ocp.constraints.idxsh = np.array(range(nsh), dtype=np.int32)

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" 
        ocp.solver_options.qp_solver_cond_N = self.N_prediction
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N
        ocp.solver_options.N_horizon = self.N_prediction
        ocp.solver_options.regularize_method = 'NO_REGULARIZE'  
        ocp.solver_options.levenberg_marquardt = 10.0

        return ocp


def main():
    payload_node = PayloadControlMujocoNode()
    return None

if __name__ == '__main__':
    main()

