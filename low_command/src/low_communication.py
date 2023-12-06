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

"""The decorator below @jax.jit, jit-compiles the function.
You can also use the following to jit-compile a function:

    pd_control_jit = jax.jit(pd_control)"""

def initialize():
    """Run dummy data through jit-compiled functions to initialize them"""
    pd_control(jp.zeros(1), jp.zeros(1), jp.zeros(1), jp.zeros(1), jp.zeros(1))

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




def sigIntHandler(signal, frame):
    global g_request_shutdown
    g_request_shutdown = True

class LowCommunication:
    def __init__(self):
        rospy.init_node("low_communication", log_level=rospy.INFO)
        rospy.on_shutdown(self.shutdown)
        
        self.lcmd = LowCmd()
        self.lcmd.head = [0xFE, 0xEF]
        self.lcmd.levelFlag = UNITREE_LEGGED_SDK.LOWLEVEL
        self.lcmd.reserve = 0

        self.lcmd_pub = rospy.Publisher("low_cmd", LowCmd, queue_size=100)
        self.lcmd_sub = rospy.Subscriber("cmd_torque", Wrench, self.lcmdTorqueCallback)



        self.lcmd_angle_sub = rospy.Subscriber("cmd_angle", Float32, self.lcmdAngleCallback)
        self.lstate_sub = rospy.Subscriber("low_state", LowState, self.lstateCallback)


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






    def lcmdAngleCallback(self, msg):
        q_des = [msg]
        self.publishAngleCommand(q_des)

    def lstateCallback(self, state):
        self.q[0] = [state[0].MotorState.q]
        self.dq[0] = [state[0].MotorState.dq]
        rospy.loginfo(self.q[0])
        rospy.loginfo(self.dq[0])







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
        """Run PD control loop."""

        while True:
            # get desired state
            q_des = jp.array(q_des)

            # get current state)
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





# this publishes a torque command
    def publishCommand(self, command):
        v = self.clamp(command[0], -2.0, 2.0)
        self.lcmd.motorCmd[0].tau = v
        self.lcmd_pub.publish(self.lcmd)
# general clamp function
    def clamp(self, n, lower, upper):
        return max(lower, min(n, upper))

if __name__ == "__main__":
    initialize()
    signal.signal(signal.SIGINT, sigIntHandler)
    comm = LowCommunication()
    rate = rospy.Rate(100.0)

    while not g_request_shutdown and not rospy.is_shutdown():
        rospy.spin_once()
        rate.sleep()

    comm.shutdown()
