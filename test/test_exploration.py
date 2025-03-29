import os
import numpy as np
import math
import sys
import open3d as o3d
import yaml
from tqdm import tqdm

sys.path.append('../')

from quadrotor_simulator_py.map_tools import *
from quadrotor_simulator_py.sensor_simulator import *
from quadrotor_simulator_py.utils import *
from quadrotor_simulator_py.visualizer import *
from quadrotor_simulator_py.exploration import *

def count_cells(og):
    unknown = 0
    occupied = 0
    free = 0
    for index in tqdm(range(len(og.data)), desc="Evaluating performance"):
        if og.unknown(og.data[index]):
            unknown+=1
        elif og.occupied(og.data[index]):
            occupied+=1
        else:
            free+=1
    return (unknown, occupied, free)

def test_exploration(config, environment_name, meshfile,
                     exploration_type="random", visualize=True, save=False):

    sensor_simulator = SensorSimulator(config, meshfile)
    og = OccupancyGrid(config)
    viz_elements = []
    
    with open(config, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YamlError as exc:
            print(exc)
    trimmed_range_max = data['depth_camera']['trimmed_range_max']
    
    # Create an initial pose
    Twb = Pose()
    t = np.array([14.0, 0.0, 5.0])
    rpy = np.array([math.pi, 0.0, 0.0])
    R = Rot3().from_euler_zyx([rpy[2], rpy[1], rpy[0]])
    Twb.set_translation(t)
    Twb.set_rotation(R.R)

    # Select exploration approach
    explorer = None
    if exploration_type == "random":
        explorer = RandomExplorer(config, og, sensor_simulator, Twb)
    elif exploration_type == "greedy":
        explorer = GreedyExplorer(config, og, sensor_simulator, Twb)
    else:
        explorer = StudentExplorer(config, og, sensor_simulator, Twb)
        
    # Iterate 20 exploration steps
    for i in tqdm(range(20), desc=exploration_type + " exploration progress"):
        world_frame_points = sensor_simulator.ray_mesh_intersect(Twb)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(world_frame_points)
    
        Twc = Twb.compose(sensor_simulator.Tbc)
   
        for w in world_frame_points:
            og.add_ray(Point().from_numpy(Twc.translation()),
                       Point().from_numpy(w), trimmed_range_max)

        if visualize:
            viz_elements.append(point_cloud)
            triad = visualize_sensor_triad(Twc)
            viz_elements.append(triad)

            line_list = line_segments_numpy(Twc.translation(), world_frame_points)
            viz_elements = viz_elements + line_list

            mesh = visualize_mesh(meshfile)
            viz_elements.append(mesh)
            o3d.visualization.draw_geometries(viz_elements)

            if save: # save screenshots you can use to make a video
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                for e in viz_elements:
                    vis.add_geometry(e)
            
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image("./images/" + str(i).zfill(5) + ".png")
                vis.destroy_window()

        Twb = explorer.get_next_sensing_action(Twb)

    unknown, occupied, free = count_cells(og)
    return occupied + free

if __name__ == "__main__":
    random_score = test_exploration("../config/exploration.yaml", "environment1", "../mesh/environment1.ply",
                                    exploration_type="random", visualize=True, save=False)
    greedy_score = test_exploration("../config/exploration.yaml", "environment1", "../mesh/environment1.ply",
                                    exploration_type="greedy", visualize=True, save=False)
    student_score = test_exploration("../config/exploration.yaml", "environment1", "../mesh/environment1.ply",
                                    exploration_type="student", visualize=True, save=False)

    print('Random score: ' + str(random_score))
    print('Greedy score: ' + str(greedy_score))
    print('Student score: ' + str(student_score))

    if (student_score > greedy_score) and (student_score > random_score):
        print('Test passed')
    else:
        print('Test failed')
