#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates

class Moving():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def moving(self):
        n=0
        while not rospy.is_shutdown():
            obstacle = ModelState()
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'obstacle3' :
                    obstacle.model_name = 'obstacle3'
                    obstacle.pose = model.pose[i]
                    obstacle.twist = Twist()
                    if n > 2500 and n < 4500:
                        obstacle.twist.linear.x = -0.8
                    elif n > 0 and n < 2000:
                        obstacle.twist.linear.x = 0.8
                    else:
                        obstacle.twist.linear.x = 0.0
                    obstacle.twist.angular.z = 0
                    self.pub_model.publish(obstacle)
                    time.sleep(0.1)
                n+=1
                if n>5000:
                    n = n-5000


def main():
    rospy.init_node('moving_obstacle3')
    moving = Moving()

if __name__ == '__main__':
    main()
