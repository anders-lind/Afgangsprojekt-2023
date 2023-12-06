from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

print("**************************Setup**********************************")

d = generate_distutils_setup(
   packages=['facilitycobot_demo', 'robot'],
   package_dir={'':'src'}
)

setup(**d)

