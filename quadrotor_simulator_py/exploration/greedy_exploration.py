#!/usr/bin/env python
import numpy as np
import yaml
import math
import random

from quadrotor_simulator_py.utils import Rot3
from quadrotor_simulator_py.utils import Pose
from quadrotor_simulator_py.sensor_simulator import *
from quadrotor_simulator_py.map_tools import *

class GreedyExplorer:

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
        R = pose.get_so3()
        t = pose.translation()

        unknown_idxs = []
        for idx, ray in enumerate(self.depth_points):
            rotated_ray = (R  @ ray.reshape((3,1))).reshape(3)

            # Make sure the direction is normalized
            ray_dir = np.array(rotated_ray.T, dtype=np.float32)
            ray_dir /= np.linalg.norm(ray_dir)

            e = ray_dir * self.range_max + t

            if self.og.point_in_grid(Point().from_numpy(e)) and self.og.point_in_grid(Point().from_numpy(t)):
               success, raycells = self.og.get_raycells(Point().from_numpy(t), Point().from_numpy(e))

               for cell in raycells:
                   if self.og.unknown(self.og.data[self.og.cell2index(cell)]):
                       unknown_idxs.append(self.og.cell2index(cell))

        return len(set(unknown_idxs))

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
