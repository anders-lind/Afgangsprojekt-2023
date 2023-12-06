Python api for mir move base control

The API works by using a mir bridge given in the github repository:
 DFKI-NI/mir_robot: ROS support for the MiR Robots. This is a community project to use the MiR Robots with ROS. It is not affiliated with Mobile Industrial Robots. (github.com)
This should be downloaded as source into your catkin workspace whereafter running catkin_make.

Afterwards, run the command
$ rosrun mir_driver mir.launch
In the terminal (This must be done from your computer IP, NOT from the mir IP address). It will take a while for it too connect, but when it is connected, you will see many “tf_static” transform messages being given. What should be waited for, is that the roscore on your computer shows all topics from the MIR robot (the mir bridge gives all topics from the MIR robot to the Rosmaster on the computer).
Then afterwards. The mir robot can be controlled as normal using move base action (do not go down the MirMoveBaseAction route, it is a scam).
Make sure to synchronize the time between the robot and the computer, as the messages are timestamped, and a large time offset would likely cause an error. 

The code for running the python mir move API is within the address:
/facilitycobot_demo/src/tests/test_mir_navigation.py
This command starts a node and runs the mir program on it.

