// This node is used to relay messsages from the cmd_vel topic to the high_cmd
// topic in order to communicate with the Go1 quadruped.
// NOT TRUE ANYMORE

#include <iostream>
#include <signal.h>

#include <ros/ros.h>
//#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Wrench.h"
//#include <unitree_legged_msgs/HighCmd.h>
#include <unitree_legged_msgs/LowCmd.h>
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <unitree_legged_msgs/LowState.h>

using namespace UNITREE_LEGGED_SDK;

// Flag for whether shutdown is requested
sig_atomic_t volatile g_request_shutdown = 0;

void sigIntHandler(int sig)
{
  g_request_shutdown = 1;
}

class LowCommunication
{
  public:
    LowCommunication()
    {
      ROS_INFO_STREAM("Starting low communication");

      //param init:

      lcmd_.head[0] = 0xFE;        // don't change this.....exists in LowCmd, might be wrong setting
      lcmd_.head[1] = 0xEF;        // don't change this.....exists in LowCmd, might be wrong setting
      lcmd_.levelFlag = LOWLEVEL; // use HIGHLEVEL for high-level control.....use LOWLEVEL for low-level control
      //cmd_.mode = 0;              // 0: idle, 2: walk.....DNE in LowCmd
      //cmd_.gaitType = 0;          // .....DNE in LowCmd.....0: idle, 1: trot, 2: trot running, 3: climb stair, 4: trot obstacle
      //cmd_.speedLevel = 0;        // .....DNE in LowCmd.....not used (only for mode=3) 
      //cmd_.footRaiseHeight = 0;   // f.....DNE in LowCmd.....oot height while walking, delta value
      //cmd_.bodyHeight = 0;        // .....DNE in LowCmd.....don't change this
      //cmd_.euler[0] = 0;          // .....DNE in LowCmd.....euler angles are for mode=1 only
      //cmd_.euler[1] = 0;
      //cmd_.euler[2] = 0;
      //cmd_.velocity[0] = 0.0f;    // .....DNE in LowCmd.....forward speed, -1 ~ +1 m/s for gaitType = 1; upto 3 m/s for gaitType = 2
      //cmd_.velocity[1] = 0.0f;    // .....DNE in LowCmd.....side speed
      //cmd_.yawSpeed = 0.0f;       // .....DNE in LowCmd.....rotation speed, rad/s
      
      //from Unitree values may need to be debugged:
      //for(int i=0; i<4; i++){
        //hip
      //  lcmd_.motorCmd[i*3+0].mode = 0x0A;
      //  lcmd_.motorCmd[i*3+0].Kp = 7;
      //  lcmd_.motorCmd[i*3+0].dq = 0;
      //  lcmd_.motorCmd[i*3+0].Kd = 3;
      //  lcmd_.motorCmd[i*3+0].tau = 0;
        //thigh
      //  lcmd_.motorCmd[i*3+1].mode = 0x0A;
      //  lcmd_.motorCmd[i*3+1].Kp = 18;
      //  lcmd_.motorCmd[i*3+1].dq = 0;
      //  lcmd_.motorCmd[i*3+1].Kd = 8;
      //  lcmd_.motorCmd[i*3+1].tau = 0;
        //calf
      //  lcmd_.motorCmd[i*3+2].mode = 0x0A;
      //  lcmd_.motorCmd[i*3+2].Kp = 30;
      //  lcmd_.motorCmd[i*3+2].dq = 0;
      //  lcmd_.motorCmd[i*3+2].Kd = 15;
      //  lcmd_.motorCmd[i*3+2].tau = 0;
  //  }
  //  for(int i=0; i<12; i++){
  //      lcmd_.motorCmd[i].q = lstate_.motorState[i].q;
  //  }

      lcmd_.reserve = 0;           // don't change this

      //end of param init

      const auto queue_size = 100;
      lcmd_pub_ = nh_.advertise<unitree_legged_msgs::LowCmd>("low_cmd", queue_size);                        //good
      lcmd_sub_ = nh_.subscribe("cmd_torque", queue_size, &LowCommunication::lcmdTorqueCallback, this);     //good

      ros::Duration(1.0).sleep();
      start_motor();
      ROS_INFO_STREAM("low communication ready");
    }

    void shutdown()
    {
      stop_motors();
      ros::Duration(1.0).sleep();
      ROS_INFO_STREAM("Stopping and exiting..." );
      ros::shutdown();
      std::cout << "Exited." << std::endl;
    }

  private:
    ros::NodeHandle nh_;
    ros::Publisher lcmd_pub_;
    ros::Subscriber lcmd_sub_;
    unitree_legged_msgs::LowCmd lcmd_;
    unitree_legged_msgs::LowState lstate_;


    void lcmdTorqueCallback(geometry_msgs::Wrench msg)
    {
      std::vector<double> command{msg.torque.z};
      publishCommand(command);
    }



    void start_motor() //motors
    {
      lcmd_.motorCmd[0].mode = 0x0A;
      //lcmd_.motorCmd[0].tau = 0;

    //  for(int i=0; i<4; i++){
    //    //hip
    //    lcmd_.motorCmd[i*3+0].mode = 0x0A;
    //    lcmd_.motorCmd[i*3+0].tau = 0;
        //thigh
    //    lcmd_.motorCmd[i*3+1].mode = 0x0A;
    //    lcmd_.motorCmd[i*3+1].tau = 0;
        //calf
    //    lcmd_.motorCmd[i*3+2].mode = 0x0A;
    //    lcmd_.motorCmd[i*3+2].tau = 0;
    //}


      lcmd_pub_.publish(lcmd_);
      ROS_INFO_STREAM("Robot started");
    }



  //done
    void stop_motors()
    {
      for(int i=0; i<4; i++){
        //hip
        lcmd_.motorCmd[i*3+0].tau = 0;
        //lcmd_.motorCmd[i*3+0].mode = 0x00;
        
        //thigh
        lcmd_.motorCmd[i*3+1].tau = 0;
        //lcmd_.motorCmd[i*3+1].mode = 0x00;
        //calf
        lcmd_.motorCmd[i*3+2].tau = 0;
        //lcmd_.motorCmd[i*3+2].mode = 0x00;
        
    }
      lcmd_pub_.publish(lcmd_);
      ROS_INFO_STREAM("Robot stopped");
    }





    void publishCommand(std::vector<double> command)
    {
      double v = clamp(command[0], -2.0, 2.0); //max and min torque values....torques in N*m......from URDF+testing
      // double w = clamp(command[1], -3.0, 3.0); //.........superfluous

      lcmd_.motorCmd[0].tau = command[0];

      lcmd_pub_.publish(lcmd_);
    }

    // Clamps n between the supplied lower and upper limits........not sure what this is doing
    double clamp(double n, double lower, double upper) {
      return std::max(lower, std::min(n, upper));
    }
};






auto main(int argc, char **argv) -> int
{
  ros::init(argc, argv, "low_communication", ros::init_options::NoSigintHandler);
  signal(SIGINT, sigIntHandler);

  LowCommunication comm;

  ros::Rate rate(100.0);

  // main loop which exits when the g_request_shutdown flag is true
  while (!g_request_shutdown && ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }

  // shutdown gracefully if node is interrupted with ctrl+C
  comm.shutdown();

  return 0;
}
