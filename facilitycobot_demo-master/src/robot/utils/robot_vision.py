#!/usr/bin/env python3

# ros modules
import rospy
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection3DArray

from vision_msgs.msg import Detection3D

from std_msgs.msg import Float32MultiArray

from robot.utils.method_logger import log_method


# 0 - cracker box
# 2 - bleach cleanser
# 3 - bowl
@log_method()
class object_pose_estimation:
    """[summary]
    This class is used to listen to the rostopic publishing the object poses. 
    """

    @log_method()
    def __init__(self):
        """[summary]
        Initalizer for the vision class. This class is used to listen to the rostopic publishing the object poses.
        """

        self.pose_sub = rospy.Subscriber('/mvprbpf_ros/tracked_pose', ObjectHypothesisWithPose, self.__object_pose_callback)
        self.confidence_sub = rospy.Subscriber('/mvprbpf_ros/predictive_confidences', Float32MultiArray, self.__confidence_interval_callback)
        
        self.detectron_confidence_sub = rospy.Subscriber('/detectron2_ros/result', Detection3DArray, self.__detectron_result_callback)

        self.__predictive_confidence_score = 0 # Predictive confidence is set to 0, to convey no confidence
        self.__pose_msg = None
        self.__detectron_object_detection = None

    @log_method()
    def get_object_pose(self):
        """[summary]
        will get the latest value from the topic, and delete it.

        returns:
            [list]: [description] returns a list of the object pose and the object id
        """

        if not(self.__pose_msg == None):
            x = self.__pose_msg.pose.pose.position.x
            y = self.__pose_msg.pose.pose.position.y
            z = self.__pose_msg.pose.pose.position.z
            rx = self.__pose_msg.pose.pose.orientation.x
            ry = self.__pose_msg.pose.pose.orientation.y
            rz = self.__pose_msg.pose.pose.orientation.z
            rw = self.__pose_msg.pose.pose.orientation.w

            object_id = self.__pose_msg.id
            object_pose = [x,y,z,rx,ry,rz,rw]

            self.__pose_msg = None # Remove after reading

            return  [object_pose, object_id]
        
        else:
            return None

    @log_method()
    def get_predictive_confidence(self):
        """[summary]
        Gets the latest update from the predictive confidence interval score

        returns:
            [Float64]
        """


        predictive_confidence_interval_score = 0

        if not(self.__predictive_confidence_score == 0):
            predictive_confidence_interval_score = self.__predictive_confidence_score.data[0]

            self.__predictive_confidence_score = 0 # remove after reading
        

        return predictive_confidence_interval_score

    @log_method()
    def delete_object_pose_estimation(self):
        """[summary]
        Deletes the current object pose estimation, incase the robot is now searching another place.
        """

        self.__predictive_confidence_score = 0 
        self.__pose_msg = None

    @log_method()
    def get_detectron_object_detection(self, object_id = -1, score_thresholds = [0,0,0,0,0]):
        """[summary]
        Gets the latest update from the detectron object detection

        args:
            object_id [int]: is the id of the object to be detected, with default, then the highest confidence object is chosen
            score_thresholds [descriptions]: is a list of thresholds before returning detectron detection, The list sequence is given as
                                            [cracker box, mustard bottle, bleach cleaner, bowl, mug]

        returns:
            [list]: [description] returns a list of the object pose and the object id
        """

        test_detection = self.__detectron_object_detection

        if test_detection != None:

            if object_id == -1: # if no object id is given, then the highest confidence object is chosen
                test_detection.sort(reverse=True)
                return test_detection[0]

            object_with_object_id = [x for x in test_detection if x[1] == object_id]

            if object_with_object_id.__len__() > 0:
                object_with_object_id.sort(reverse=True)

                if object_with_object_id[0][0] > score_thresholds[object_id]:
                    return object_with_object_id[0]
                else:
                    return None

        return None
    
    @log_method()
    def delete_detectron_object_detection(self):
        """[summary]
        Deletes the current object detection from the detectron, in case the robot is now searching another place.
        """

        self.__detectron_object_detection = None

    def __detectron_result_callback(self, msg):
        """[summary]
        Detects messages from /detectron2_ros/result, containing all object results seen in the camera. 
        This function returns the highest probability object detection.

        updates the self variable "self.__detectron_object_detection"
        """

        object_confidence_results = []
        
        if msg.detections.__len__() > 0:
            for detections in msg.detections:
                
                object_confidence_results.append([detections.results[0].score, detections.results[0].id])

            self.__detectron_object_detection = object_confidence_results
 
    def __object_pose_callback(self, msg):
        """[summary]
        Detects if updates are sent in the rostopic /mvprbpf_ros/tracked_pose, and adds them to self.msg
        """

        self.__pose_msg = msg

    def __confidence_interval_callback(self, msg):
        """[summary]
        Detects if updates are sent in the rostopic /mvprbpf_ros/predictive_confidences and adds them to self.predictive_confi
        """
        
        self.__predictive_confidence_score = msg