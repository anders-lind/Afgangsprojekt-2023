# FacilityCobot demo
FacilityCobot demo executer

# Run

## Run MiR navigation tests

Install DFKI MiR package
```
sudo apt install ros-noetic-mir-robot
```
Run following commands

```
rosrun mir_driver mir.launch
roslaunch facilitycobot_demo test_mir_navigation.launch
```

ROS_IP and ROS Master must be set too computer IP, NOT from the mir IP address (essentially don’t change ROS_IP and ROS_MASTER_URI). It will take a while for it to connect, but when it is connected, it will print “tf_static” transform messages.

