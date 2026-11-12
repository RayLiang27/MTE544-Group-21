import sys

from utilities import Logger

from rclpy.time import Time

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from rclpy.node import Node
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom

from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

from rclpy import init, spin, spin_once

import numpy as np
import message_filters

rawSensors=0
kalmanFilter=1
odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)
imu_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)

class localization(Node):
    
    def __init__(self, type, dt, loggerName="robotPose.csv", loggerHeaders=["odom_x", "odom_y", "imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w", "kf_x", "kf_y", "stamp"]):
        loggerHeaders=["odom_x", "odom_y", "odom_theta", "imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx", "kf_w", "kf_th", "kf_x", "kf_y", "stamp"]
        # loggerHeaders=["odom_x", "odom_y", "odom_theta", "kf_th", "kf_x", "kf_y", "stamp"]
        # loggerHeaders=["odom_x", "odom_y", "odom_theta", "imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w", "kf_th", "kf_x", "kf_y", "stamp"]

        super().__init__("localizer")

        self.loc_logger=Logger( loggerName , loggerHeaders)
        self.pose=None
        
        if type==rawSensors:
            self.initRawSensors()
        elif type==kalmanFilter:
            self.initKalmanfilter(dt)
        else:
            print("We don't have this type for localization", sys.stderr)
            return  

    def initRawSensors(self):
        self.create_subscription(odom, "/odom", self.odom_callback, qos_profile=odom_qos)
        
    def initKalmanfilter(self, dt):
        
        # TODO Part 3: Set up the quantities for the EKF (hint: you will need the functions for the states and measurements)
        
        x= np.zeros(6) # [x, y, theta, w, v, vdot], initialized all to 0

        # Sim
        # Q= np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # expected shape: (6, 6)
        # R= np.diag([0.5, 0.5, 0.5, 0.5]) # expected shape: (4, 4)
        # P= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # initial covariance expected shape: (6, 6)

        # Real robot
        # try 1: 
        # Q= np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # expected shape: (6, 6)
        # R= np.diag([0.5, 0.5, 0.5, 0.5]) # expected shape: (4, 4)
        # P= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # initial covariance expected shape: (6, 6)

        # try 2:
        # Q= np.diag([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) # expected shape: (6, 6)
        # R= np.diag([0.5, 0.5, 0.5, 0.5]) # expected shape: (4, 4)
        # P= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # initial covariance expected shape: (6, 6)

        # try 3:
        # Q= np.diag([0.8, 0.8, 0.8, 0.8, 0.8, 0.8]) # expected shape: (6, 6)
        # R= np.diag([0.5, 0.5, 0.5, 0.5]) # expected shape: (4, 4)
        # P= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # initial covariance expected shape: (6, 6)

        # try 4: 
        Q= np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # expected shape: (6, 6)
        R= np.diag([0.5, 0.5, 0.5, 0.5]) # expected shape: (4, 4)
        P= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # initial covariance expected shape: (6, 6)

        # try 5: 
        # Q= np.diag([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]) # expected shape: (6, 6)
        # R= np.diag([0.85, 0.85, 0.85, 0.85]) # expected shape: (4, 4)
        # P= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # initial covariance expected shape: (6, 6)

        # point 
        # Q= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # expected shape: (6, 6)
        # R= np.diag([0.1, 0.1, 0.1, 0.1]) # expected shape: (4, 4)
        # P= np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # initial covariance expected shape: (6, 6)
        
        self.kf=kalman_filter(P,Q,R, x, dt)
        
        # TODO Part 3: Use the odometry and IMU data for the EKF
        self.odom_sub=message_filters.Subscriber(self, odom, "/odom", qos_profile=odom_qos)
        self.imu_sub=message_filters.Subscriber(self, Imu, "/imu", qos_profile=imu_qos)
        
        time_syncher=message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size=10, slop=0.1)
        time_syncher.registerCallback(self.fusion_callback)
    
    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):
        
        # TODO Part 3: Use the EKF to perform state estimation
        # Take the measurements
        # your measurements are the linear velocity and angular velocity from odom msg
        # and linear acceleration in x and y from the imu msg
        # the kalman filter should do a proper integration to provide x,y and filter ax,ay
        z=np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z, imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y]) # [v, w, ax, ay]
        
        # Implement the two steps for estimation
        self.kf.predict()
        self.kf.update(z)
        
        # Get the estimate
        xhat=self.kf.get_states()

        # Update the pose estimate to be returned by getPose
        self.pose=np.array(np.append(xhat[:3], odom_msg.header.stamp))

        # TODO Part 4: log your data
        # ALL
        self.loc_logger.log_values(list([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, euler_from_quaternion(odom_msg.pose.pose.orientation)]) + list(z[2:]) + [xhat[5], (xhat[3]*xhat[4]), xhat[4], xhat[3], xhat[2], xhat[0], xhat[1], Time.from_msg(odom_msg.header.stamp).nanoseconds])

        # self.loc_logger.log_values(list([euler_from_quaternion(odom_msg.pose.pose.orientation)]) + list(z[2:]) + [xhat[5], (xhat[3]*xhat[4]), xhat[4], xhat[3], xhat[2], xhat[0], xhat[1], Time.from_msg(odom_msg.header.stamp).nanoseconds])

        # ORIGINAL
        # self.loc_logger.log_values(list(z[2:]) + [xhat[5], (xhat[3]*xhat[4]), xhat[4], xhat[3], xhat[0], xhat[1], Time.from_msg(odom_msg.header.stamp).nanoseconds])

        #add odom xy
        # self.loc_logger.log_values(list([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y]) + list(z[2:]) + [xhat[5], (xhat[3]*xhat[4]), xhat[4], xhat[3], xhat[0], xhat[1], Time.from_msg(odom_msg.header.stamp).nanoseconds])

        # self.loc_logger.log_values(list([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, euler_from_quaternion(odom_msg.pose.pose.orientation)]) + [xhat[2], xhat[0], xhat[1], Time.from_msg(odom_msg.header.stamp).nanoseconds])
      
    def odom_callback(self, pose_msg):
        
        self.pose=[ pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    euler_from_quaternion(pose_msg.pose.pose.orientation),
                    Time.from_msg(pose_msg.header.stamp).nanoseconds]

    # Return the estimated pose
    def getPose(self):
        return self.pose


if __name__=="__main__":
    
    init()
    
    LOCALIZER=localization()
    
    spin(LOCALIZER)
