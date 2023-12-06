#!/usr/bin/env python

from jax import numpy as jp
import jax
import time

import signal
import rospy
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float32
from unitree_legged_msgs.msg import LowCmd, LowState
from unitree_legged_sdk import UNITREE_LEGGED_SDK

g_request_shutdown = False

def sigIntHandler(signal, frame):
    global g_request_shutdown
    g_request_shutdown = True

#### PD Funcitons ####
"""The decorator below @jax.jit, jit-compiles the function.
You can also use the following to jit-compile a function:

    pd_control_jit = jax.jit(pd_control)"""

@jax.jit
def pd_control(q: jp.ndarray,
               dq: jp.ndarray,
               q_des: jp.ndarray,
               Kp: jp.ndarray,
               Kd: jp.ndarray) -> jp.ndarray:

    """PD control law for a single joint.
    Args:
        q (numpy.ndarray): current joint angle.
        qd (numpy.ndarray): current joint velocity.
        q_des (numpy.ndarray): desired joint angle.
        Kp (numpy.ndarray): proportional gain (stiffness).
        Kd (numpy.ndarray): derivative gain (damping).
    Returns:
        numpy.ndarray: joint torque.
    """
    tau = Kp*(q_des - q) - Kd*(dq)
    return tau

def initialize_pd():
    """Run dummy data through jit-compiled functions to initialize them"""
    pd_control(jp.zeros(1), jp.zeros(1), jp.zeros(1), jp.zeros(1), jp.zeros(1))


THIGH_OFFSET = jp.array[0.08]
"""constant: the length of the thigh motor"""
LEG_OFFSET_X = jp.array[0.1881]
"""constant: x distance from the robot COM to the leg base."""
LEG_OFFSET_Y = jp.array[0.04675]
"""constant: y distance from the robot COM to the leg base."""
THIGH_LENGTH = jp.array[0.213]
"""constant: length of the thigh and also the length of the calf"""
STANDING_JOINT_ANGLES = jp.array([0.0, 0.67, -1.3])
"""constant: the joint angles for a leg while standing"""
LOWER_JOINT_LIMITS = jp.array([-0.802851, -1.0472, -2.69653])
"""constant: the lower joint angle limits for a leg"""
UPPER_JOINT_LIMITS = jp.array([0.802851, 4.18879, -0.916298])
"""constant: the upper joint angle limits for a leg"""
MOTOR_TORQUE_LIMIT = jp.array[33.5]
"""constant: the torque limit for the motors"""

#hard code these for entire script
FR = jp.ndarray[-THIGH_OFFSET[0], LEG_OFFSET_X[0], -LEG_OFFSET_Y[0]]
FL = jp.ndarray[THIGH_OFFSET[0], LEG_OFFSET_X[0], LEG_OFFSET_Y[0]]
RR = jp.ndarray[-THIGH_OFFSET[0], -LEG_OFFSET_X[0], -LEG_OFFSET_Y[0]]
RL = jp.ndarray[THIGH_OFFSET[0], -LEG_OFFSET_X[0], LEG_OFFSET_Y[0]]



#### impedence functions
@jax.jit
def forward_kinematics(offset: jp.ndarray, q: jp.ndarray) -> jp.ndarray:
    # Returns the position of the foot in the body frame centered on the
    # trunk, given the joint angles; (3,)

    # Arguments:
    # offset (jp.ndarray): see global variables
    # q (jp.ndarray): the joint angles of a leg; (3,)
 
    d = offset[0]
    fx = offset[1]
    fy = offset[2]

    length =  THIGH_LENGTH

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    p0 = fx - length*jp.sin(q2 + q3) - length*jp.sin(q2)
    p1 = (
        fy
        + d*jp.cos(q1)
        + length*jp.cos(q2)*jp.sin(q1)
        + length*jp.cos(q2)*jp.cos(q3)*jp.sin(q1)
        - length*jp.sin(q1)*jp.sin(q2)*jp.sin(q3)
    )
    p2 = (
        d*jp.sin(q1)
        - length*jp.cos(q1)*jp.cos(q2)
        - length*jp.cos(q1)*jp.cos(q2)*jp.cos(q3)
        + length*jp.cos(q1)*jp.sin(q2)*jp.sin(q3)
    )
    p = jp.stack([p0, p1, p2], axis=0)
    return p

def initialize_forward_kinematics():
    forward_kinematics(jp.zeros(3), jp.zeros(3))


@jax.jit
def jacobian(offset: jp.ndarray, q: jp.ndarray) -> jp.ndarray:
    # get the jacobian of the leg

    # Arguments:
    #     offset (jp.ndarray): see global variables
    #     q (jp.ndarray): the joint angles of a leg; (3,)

    # Returns:
    #     jp.ndarray: the jacobian of the leg, (3, 3)
    

    # d = jax.lax.select(leg in ['FR', 'RR'],
    #                     -THIGH_OFFSET,
    #                     THIGH_OFFSET)
    d = offset[0]
    
    length =  THIGH_LENGTH

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    J00 = 0.
    J01 = -length*(jp.cos(q2 + q3) + jp.cos(q2))
    J02 = -length*jp.cos(q2 + q3)
    J10 = (
        length*jp.cos(q1)*jp.cos(q2)
        - d*jp.sin(q1)
        + length*jp.cos(q1)*jp.cos(q2)*jp.cos(q3)
        - length*jp.cos(q1)*jp.sin(q2)*jp.sin(q3)
    )
    J11 = -length*jp.sin(q1)*(jp.sin(q2 + q3) + jp.sin(q2))
    J12 = -length*jp.sin(q2 + q3)*jp.sin(q1)
    J20 = (
        d*jp.cos(q1)
        + length*jp.cos(q2)*jp.sin(q1)
        + length*jp.cos(q2)*jp.cos(q3)*jp.sin(q1)
        - length*jp.sin(q1)*jp.sin(q2)*jp.sin(q3)
    )
    J21 = length*jp.cos(q1)*(jp.sin(q2 + q3) + jp.sin(q2))
    J22 = length*jp.sin(q2 + q3)*jp.cos(q1)

    J = jp.stack([
        jp.stack([J00, J01, J02], axis=0),
        jp.stack([J10, J11, J12], axis=0),
        jp.stack([J20, J21, J22], axis=0)
    ], axis=0)

    return J

def initialize_jacobian():
    jacobian(jp.zeros(3), jp.zeros(3))


@jax.jit
def foot_vel(offset: jp.ndarray, q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
    """Returns the linear velocity of the foot in the body frame; (3,)

    Arguments:
        offset (jp.ndarray): see global variables
        q (jp.jp.ndarray): the joint angles of a leg; (3,)
        qd (jp.jp.ndarray): the joint speeds of a leg; (3,)
    """
    J =  jacobian(offset, q)
    vel = jp.matmul(J, qd)
    return vel

def initialize_foot_vel():
    foot_vel(jp.zeros(3), jp.zeros(3), jp.zeros(3))


def standing_foot_positions() -> jp.ndarray:
    """Returns the positions of the feet in the body frame when the robot
    is standing; (12,)"""

    return jp.concatenate([
        forward_kinematics(FR, STANDING_JOINT_ANGLES),
        forward_kinematics(FL, STANDING_JOINT_ANGLES),
        forward_kinematics(RR, STANDING_JOINT_ANGLES),
        forward_kinematics(RL, STANDING_JOINT_ANGLES),
        ])

@jax.jit
def imp_control(Kp: jp.ndarray, Kd: jp.ndarray, p_des: jp.ndarray,
                q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Impedance control for the go1 robot. All arguments will need to be
        formatted in the order FR, FL, RR, RL.

        Arguments:
            Kp: proportional gain; shape (12,); Kpx, Kpy, Kpz for each leg
            Kd: derivative gain; shape (12, ); Kdx, Kdy, Kdz for each leg
            p_des: desired position of the feet, specified in the body frame;
                shape (12,)
            obs: observations
        """

        # desired positions of the feet
        p_des_FR = p_des[0:3]
        p_des_FL = p_des[3:6]
        p_des_RR = p_des[6:9]
        p_des_RL = p_des[9:12]

        # joint angles
        q_FR = q[0:3]
        q_FL = q[3:6]
        q_RR = q[6:9]
        q_RL = q[9:12]

        # joint speeds
        qd_FR = qd[0:3]
        qd_FL = qd[3:6]
        qd_RR = qd[6:9]
        qd_RL = qd[9:12]

        # estimate the current positions of the feet
        p_FR = forward_kinematics(FR, q_FR)
        p_FL = forward_kinematics(FL, q_FL)
        p_RR = forward_kinematics(RR, q_RR)
        p_RL = forward_kinematics(RL, q_RL)

        # estimate the current velocities of the feet
        pd_FR = foot_vel(FR, q_FR, qd_FR)
        pd_FL = foot_vel(FL, q_FL, qd_FL)
        pd_RR = foot_vel(RR, q_RR, qd_RR)
        pd_RL = foot_vel(RL, q_RL, qd_RL)

        # jacobians for each leg
        J_FR = jacobian(FR, q_FR)
        J_FL = jacobian(FL, q_FL)
        J_RR = jacobian(RR, q_RR)
        J_RL = jacobian(RL, q_RL)

        # compute the torques
        u_FR = jp.matmul(jp.transpose(J_FR),
                         Kp[0:3]*(p_des_FR - p_FR) - Kd[0:3]*pd_FR)
        u_FL = jp.matmul(jp.transpose(J_FL),
                         Kp[3:6]*(p_des_FL - p_FL) - Kd[3:6]*pd_FL)
        u_RR = jp.matmul(jp.transpose(J_RR),
                         Kp[6:9]*(p_des_RR - p_RR) - Kd[6:9]*pd_RR)
        u_RL = jp.matmul(jp.transpose(J_RL),
                         Kp[9:12]*(p_des_RL - p_RL) - Kd[9:12]*pd_RL)

        u = jp.concatenate([u_FR, u_FL, u_RR, u_RL])
        return u

def initialize_impedence():
    imp_control(jp.zeros(12), jp.zeros(12), jp.zeros(12), jp.zeros(12), jp.zeros(12))



class LowCommunication:
    def __init__(self):
        rospy.init_node("low_communication", log_level=rospy.INFO)
        rospy.on_shutdown(self.shutdown)
        
        self.lcmd = LowCmd()
        self.lcmd.head = [0xFE, 0xEF]
        self.lcmd.levelFlag = UNITREE_LEGGED_SDK.LOWLEVEL
        self.lcmd.reserve = 0

        #used for simple torque commands tests
        self.lcmd_pub = rospy.Publisher("low_cmd", LowCmd, queue_size=100)
        self.lcmd_sub = rospy.Subscriber("cmd_torque", Wrench, self.lcmdTorqueCallback)

        #subscribers for angle commands and angle state
        self.lcmd_angle_sub = rospy.Subscriber("cmd_angle", Float32, self.lcmdAngleCallback)
        self.lstate_sub = rospy.Subscriber("low_state", LowState, self.lstateCallback)

        self.lcmd_position_sub = rospy.Subscriber("cmd_position", Float32, self.lcmdPositionCallback)

        self.q = jp.array([])
        self.dq = jp.array([])


        rospy.sleep(1.0)
        self.start_motor()
        rospy.loginfo("Low communication ready")
    
    def shutdown(self):
        self.stop_motors()
        rospy.sleep(1.0)
        rospy.loginfo("Stopping and exiting...")
        rospy.signal_shutdown("User requested shutdown.")
        print("Exited.")

    def lcmdTorqueCallback(self, msg):
        tau = [msg.torque.z]
        self.publishCommand(tau)




    #message for angle command is float in radians
    def lcmdAngleCallback(self, msg):
        q_des = [msg]
        self.publishAngleCommand(q_des)
    #subscribed to entire low state, pulling front right angle and speed  for feedback
    def lstateCallback(self, state):
        self.q[0] = [state[0].MotorState.q]
        self.dq[0] = [state[0].MotorState.dq]
        rospy.loginfo(self.q[0])
        rospy.loginfo(self.dq[0])

    #may switch this to standing position using provided function
    def lcmdPositionCallback(self, msg):
        p_des = [msg]
        self.publishPositionCommand(p_des)
        #safety is in publish command -> only sends one torque command for now




    def start_motor(self):
        self.lcmd.motorCmd[0].mode = 0x0A
        self.lcmd_pub.publish(self.lcmd)
        rospy.loginfo("Robot started")

    def stop_motors(self):
        for i in range(4):
            self.lcmd.motorCmd[i * 3 + 0].tau = 0
            self.lcmd.motorCmd[i * 3 + 1].tau = 0
            self.lcmd.motorCmd[i * 3 + 2].tau = 0

        self.lcmd_pub.publish(self.lcmd)
        rospy.loginfo("Robot stopped")

   

    def publishAngleCommand(self, q_des):
        # publishes a torque based on a current and desired angle with PD control
        # for one joint
        """Run PD control loop."""

        while not rospy.is_shutdown():
            # get desired state/angle
            q_des = jp.array(q_des)

            # get current state/angle
            q = self.q
            dq = self.dq

            if len(self.q) == 0 or len(self.dq) == 0:
                print(f"fail")
                exit

            # compute control
            Kp = jp.array(1.0)
            Kd = jp.array(0.1)

            tau = pd_control(q, dq, q_des, Kp, Kd)
            tau = jp.tau.astype("float32")

            # publish control
            self.publishCommand(tau[0])



    def publishPositionCommand(self, p_des):
        # publishes a torque based on a current and desired angle with PD control
        # for one joint
        """Run impedence control loop."""

        while not rospy.is_shutdown():
            # get desired state/angle
            p_des = jp.array(p_des)

            # get current state/angle
            q = self.q
            dq = self.dq

            if len(self.q) == 0 or len(self.dq) == 0:
                print(f"fail")
                exit

            # compute control
            Kp = jp.array(1.0)
            Kd = jp.array(0.1)

            u = imp_control(Kp, Kd, p_des, q, dq)
            tau = u[0]
            tau = jp.tau.astype("float32")

            # publish control
            self.publishCommand(tau[0])



# this publishes a torque command
    def publishCommand(self, command):
        v = self.clamp(command[0], -2.0, 2.0)
        self.lcmd.motorCmd[0].tau = v
        self.lcmd_pub.publish(self.lcmd)

# general clamp function
    def clamp(self, n, lower, upper):
        return max(lower, min(n, upper))






if __name__ == "__main__":
    initialize_pd()
    initialize_forward_kinematics()
    initialize_jacobian()
    initialize_foot_vel()
    initialize_impedence()

    signal.signal(signal.SIGINT, sigIntHandler)
    comm = LowCommunication()
    rate = rospy.Rate(100.0)

    while not g_request_shutdown and not rospy.is_shutdown():
        rospy.spin_once()
        rate.sleep()

    comm.shutdown()
