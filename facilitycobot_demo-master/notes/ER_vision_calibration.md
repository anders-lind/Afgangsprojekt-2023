Connect to the ER for camera calibration.

Remove xavier from the router. Make sure the ER computer is connected to a network that the PC can see.

Set the following as the ros master and ros ip (ROS_IP might be different for others, as this is my computers IP address)
export ROS_MASTER_URI=http://192.168.11.30:11311 && export ROS_IP=192.168.11.106

where 192.168.11.30 is the IP of the ER webinterface
and 192.168.11.106 is the IP of the computer.

Check the connection to ER’s network by executing the command:
-	rostopic list
Should output the following list of topics
 
There is more topics than these.

Alternatively, check the web interface by entering the IP (192.168.11.30) into the web interface. Remember to disconnect the xavier, as it would take the IP address spot from ER.
Web interface: 
 


Launch the realsense camera, by using the command:
-	roslaunch realsense2_camera rs_aligned_depth.launch
Output should look like something akin the following.
 

Check if it works by using rviz in another terminal, remember to export the correct ROS IP in the new terminal. Find the camera feed by going to
Add -> by topic -> camera -> color -> image_raw -> image
 

Afterwards feed information from
-	camera/color /camera_info
-	camera/color/image_raw
-	camera/color/image_raw/compressed

into the ER’s topics
-	camera/camera_info
-	camera/image_raw
-	camera/image_raw/compressed

The code supplied in the file ER_vision.py does this.


When the camera feed is visible in the ER website, then the calibration can begin. Keep adding different poses of the robot looking at the chessboard. Add around 30 images to get a solid calibration
The add command in ER will run for about 20 seconds for each image
 
 



Setting up realsense camera for ros

The realsense ROS setup is based on the guide followed by this github repository:
IntelRealSense/realsense-ros: Intel(R) RealSense(TM) ROS Wrapper for Depth Camera (github.com)

I choose the legacy ROS1 wrapper, as I work with ROS1. 
 
It deals with both the installation of the realsense camera packages, and the ROS wrapper.



