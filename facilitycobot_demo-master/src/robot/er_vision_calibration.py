#!/usr/bin/env python3

#ros
import rospy

#ros messages
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo

from robot.utils.method_logger import log_method

@log_method()
class vision_class:
    """[summary]
    This class is used to calibrate the transformation from the camera to the robot TCP.
    """

    @log_method()
    def __init__(self):
        """[summary]
        The topics from the realsense camera is listened to, whereafter the messages are published on ER's topics
        """

        self.sub_camera_info     = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.__camera_info)
        self.sub_image_raw       = rospy.Subscriber('/camera/color/image_raw', Image, self.__image_raw_receiver)
        self.sub_compressed      = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.__compressed_receiver)

        self.pub_camera_info     = rospy.Publisher('/camera/camera_info', CameraInfo)
        self.pub_image_raw       = rospy.Publisher('/camera/image_raw', Image)
        self.pub_compressed      = rospy.Publisher('/camera/image_raw/compressed', CompressedImage)

    def __camera_info(self,msg):
        """[summary]
        Publishes the camera info message on ER's topics
        """
        self.pub_camera_info.publish(msg)

    def __image_raw_receiver(self, msg):
        """[summary]
        Publishes the image raw message on ER's topics
        """
        self.pub_image_raw.publish(msg)

    def __compressed_receiver(self, msg):
        """[summary]
        Publishes the compressed image message on ER's topics
        """
        self.pub_compressed.publish(msg)


if __name__ == "__main__":
    try:

        print("\n  VERSION 3  \n")
        print("Starting new node: ur")
        rospy.init_node('vision')

        test_obj = vision_class()



        print("press 'enter' to escape")
        temp = input()


        if True:
            rospy.loginfo("----------------------- Code finished -----------------------")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")