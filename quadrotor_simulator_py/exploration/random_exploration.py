#!/usr/bin/env python
import numpy as np
import yaml
import math
import random

from quadrotor_simulator_py.utils import Rot3
from quadrotor_simulator_py.utils import Pose
from quadrotor_simulator_py.sensor_simulator import *
from quadrotor_simulator_py.map_tools import *

class RandomExplorer:

    def __init__(self, yaml_file, og, sensor_simulator, pose):
        self.curr_pose = pose

        with open(yaml_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YamlError as exc:
                print(exc)

        self._Tbc = Pose()

        self._Tbc.set_translation(np.array([data['depth_camera']['offset']['x'],
                                           data['depth_camera']['offset']['y'],
                                           data['depth_camera']['offset']['z']]))

        self._Tbc.set_rotation(Rot3().from_euler_zyx([data['depth_camera']['offset']['roll'],
                                                     data['depth_camera']['offset']['pitch'],
                                                     data['depth_camera']['offset']['yaw']]).R)

        self._depth_points = sensor_simulator.depth_points

        self._og = og

        self._sensor_simulator = sensor_simulator

        self._range_max = self.sensor_simulator.range_max

    @property
    def Tbc(self):
        return self._Tbc

    @property
    def depth_points(self):
        return self._depth_points

    @property
    def sensor_simulator(self):
        return self._sensor_simulator

    @property
    def range_max(self):
        return self._range_max

    @property
    def og(self):
        return self._og

    def evaluate_view_reward(self, pose):
        return random.random()

    def collision(self, p):
        if self.og.point_in_grid(Point().from_numpy(p)):
            index = self.og.point2index(Point().from_numpy(p))
            if self.og.unknown(self.og.data[index]) or self.og.occupied(self.og.data[index]):
                return True
            else:
                return False
        else:
            return True

    def create_new_pose(self, pose, p2):
        p1 = pose.translation()

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dy, dx)

        R = Rot3.from_euler_zyx([0, 0, angle_rad])
        
        new_pose = Pose()
        new_pose.set_translation(p2)
        new_pose.set_rotation(R.R)
        return new_pose


    def get_next_sensing_action(self, pose):

        action_dict = {}

        p = pose.translation()
        c = self.og.point2cell(Point(p[0], p[1], p[2]))

        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                for z in range(-1, 2, 1):

                    # The following two lines are a hack to keep the
                    # z coordinate greater than zero and avoid numerical instability
                    # when raytracing through the floor. The sensor observation is too
                    # sparse to get a good view of the floor so it may not perceive it well.
                    # If I make the sensor observation denser, the code takes forever to run.
                    # For now this hack is good enough but I will fix it in the future.
                    if p[2] < 2:
                        z = 0

                    e = p + np.array([x * self.og.resolution,
                                      y * self.og.resolution,
                                      z * self.og.resolution])
                    if self.collision(e):
                        continue

                    new_pose = self.create_new_pose(pose, e)
                    reward = self.evaluate_view_reward(new_pose)
                    action_dict[reward] = new_pose

        if len(action_dict) == 0:

            # Keep position the same, but change rotation
            angle_rad = math.atan2(random.random(), random.random())
            R = Rot3.from_euler_zyx([0, 0, angle_rad])
            new_pose = Pose()
            new_pose.set_translation(pose.translation())
            new_pose.set_rotation(R.R)
            return new_pose
        else:
            highest_key = max(action_dict)
            highest_value = action_dict[highest_key]
            return highest_value

