#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import TwistStamped



if __name__ == "__main__":
    node = rospy.init_node("node")
    pub = rospy.Publisher(
        name = "cmd_vel",
        data_class=TwistStamped,
        queue_size = 10
        )

    twistMessage = TwistStamped()
    twistMessage.header.frame_id = "odom"
    
    twistMessage.twist.linear.x = 0.1
    twistMessage.twist.angular.z = 0.1


    while not rospy.is_shutdown():
        # twistMessage.header.stamp.secs = rospy.Time.now().secs
        # twistMessage.header.stamp.nsecs = rospy.Time.now().nsecs

        pub.publish(twistMessage)
        rospy.sleep(1)