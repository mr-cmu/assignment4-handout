#!/usr/bin/env python
import numpy as np
import yaml
import math
import random

from quadrotor_simulator_py.utils import Rot3
from quadrotor_simulator_py.utils import Pose
from quadrotor_simulator_py.sensor_simulator import *
from quadrotor_simulator_py.map_tools import *

class StudentExplorer:

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
        # TODO: Assignment 4
        return 0.0

    def collision(self, p):
        if self.og.point_in_grid(Point().from_numpy(p)):
            index = self.og.point2index(Point().from_numpy(p))
            if self.og.unknown(self.og.data[index]) or self.og.occupied(self.og.data[index]):
                return True
            else:
                return False
        else:
            return True

    def get_next_sensing_action(self, pose):
        # TODO: Assignment 4
        return pose

