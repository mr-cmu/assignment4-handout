"""Classes for Robotic Exploration planning, 16-761: Mobile Robot Algorithms Laboratory
"""

from typing import Tuple
import numpy as np
import math
from copy import copy
from mapper_py.data_structures.grid import Cell, Point, Grid3D
from mapper_py.data_structures.sensor import Sensor
from robot import PointRobotState

from scipy.spatial import KDTree


class SimplePrimitive:
    """A single motion primitive depicting the starting position and the direction of motion.
    """

    def __init__(self, start: PointRobotState, d: Cell):
        """Both start position state and the direction are required for initialization."""
        """
        Attributes:
            dir: (Cell) Direction specified in difference in cell coordinates. For example,
                    moving towards a block to the right means a dir of Cell(0, 1).
            start: (PointRobotState) Starting state of the primitive. Also called the root node of the primitive.
        """
        self.dir = d
        self.start = start

    def end(self, step: int = 1):
        """Compute the end state of the primitive depending on the number of cells (step)

        Args:
            step: (int) Number of cells to step along the direction self.dir. Default is 1.

        Returns:
            e: (PointRobotState) The end point of this primitive, at 'step' length away from start.
        """
        # Initialize using the start state
        e = PointRobotState(copy(self.start.p), copy(
            self.start.c), copy(self.start.res))
        # Add steps along this primitive's direction
        e.update_by(self.dir * step)
        return e

    def __str__(self) -> str:
        return f"Start: {self.start.p}, Direction: {self.dir}"


class PrimitiveLibrary:
    """A library/list of simple motion primitives for a point robot operating on an occupancy grid.

    Attributes:
        prim_lib: (list of Cell) Possible directions that the robot can move
                    from a given position in an 8-connected way.
    """
    prim_lib = []
    prim_lib.append(Cell(-1, 0, 0))
    prim_lib.append(Cell(1, 0, 0))
    prim_lib.append(Cell(0, -1, 0))
    prim_lib.append(Cell(0, 1, 0))

    prim_lib.append(Cell(-1, -1, 0))
    prim_lib.append(Cell(1, -1, 0))
    prim_lib.append(Cell(-1, 1, 0))
    prim_lib.append(Cell(1, 1, 0))

    prim_lib.append(Cell(0, 0, 1))

    prim_lib.append(Cell(-1, 0, 1))
    prim_lib.append(Cell(1, 0, 1))
    prim_lib.append(Cell(0, -1, 1))
    prim_lib.append(Cell(0, 1, 1))
    prim_lib.append(Cell(-1, -1, 1))
    prim_lib.append(Cell(1, -1, 1))
    prim_lib.append(Cell(-1, 1, 1))
    prim_lib.append(Cell(1, 1, 1))

    prim_lib.append(Cell(0, 0, -1))

    prim_lib.append(Cell(-1, 0, -1))
    prim_lib.append(Cell(1, 0, -1))
    prim_lib.append(Cell(0, -1, -1))
    prim_lib.append(Cell(0, 1, -1))
    prim_lib.append(Cell(-1, -1, -1))
    prim_lib.append(Cell(1, -1, -1))
    prim_lib.append(Cell(-1, 1, -1))
    prim_lib.append(Cell(1, 1, -1))

    def __init__(self, library: list = []):
        """Initialize the list of motion primitives"""
        """
        Attributes:
            library: (list of SimplePrimitive) All possible motion primitives in this motion primitive library.
        """
        self.library = library

    @staticmethod
    def generate_primitive(state: PointRobotState, d: Cell):
        """Generate a single primitive in direction `d` from the state `state`."""
        return SimplePrimitive(state, d)

    @staticmethod
    def generate_primitives(state: PointRobotState, deltas: list):
        """Generate all primitives from the state `state` in directions `deltas`."""
        L = []
        for d in deltas:
            s = PrimitiveLibrary.generate_primitive(state, d)
            L.append(s)
        return L

    @classmethod
    def twentysix_connected(cls, state: PointRobotState):
        """Generate 26-connected motion primitive library from the state `state`."""
        return cls(library=PrimitiveLibrary.generate_primitives(state,
                                                                PrimitiveLibrary.prim_lib))
    
    def __str__(self) -> str:
        return f"Primitive Library: {self.library}"


class ExplorationPlanner:
    """Base class for a motion planner within an exploration system."""

    def __init__(self, map: Grid3D, state: PointRobotState):
        """Initialize the map being explored, current robot state, and the MPL"""
        """
        Attributes:
            map: (Grid3D) The map, as explored so far. Initially, everything is unknown in this map.
            state: (PointRobotState) The state of the robot.
            mpl: (PrimitiveLibrary) The motion primitive library generated from the state of the robot.
        """
        self.map = map
        self.state = state
        self.mpl = PrimitiveLibrary.twentysix_connected(self.state)

    def update_map(self, map: Grid3D):
        """Update the map after an observation has been incorported by an external mapper."""
        self.map = map

    def update_state(self, state: PointRobotState):
        """Update the state and the MPL after the robot moves as a consequence of taking an action."""
        self.state = state
        self.mpl = PrimitiveLibrary.twentysix_connected(self.state)

    def is_feasible(self, mp: SimplePrimitive):
        """Make sure that the motion primitive `mp` is `feasible`. This is the `collision avoidance` aspect
        of an exploration planner.

        Since the robot is constrained to move one cell at a time, the conditions for feasibility are:
        1. The end point of the mp (step = 1) should be inside the occupancy grid.
        2. The end point of the mp (step = 1) should not be an unknown cell.
        3. The end point of the mp (step = 1) should not be an occupied cell.

        Args:
            mp: (SimplePrimitive) The motion primitive to be checked for feasibility

        Returns:
            (Boolean) True if the primitive is feasible for the robot to follow; False otherwise
        """
        # TODO: Assignment 4, Task 2.1
        raise NotImplementedError

    def selection_policy(self):
        """By default, the exploration planner takes a random but safe action.

        Go through every action in the self.mpl and check if it is feasible. Then, out of
        the feasible actions pick one random action and return it. If there is no feasible
        action, return None. A quick look at `self.take_action` might help understand how this
        function is used.

        Returns:
            mp: (SimplePrimitive or None) Chosen random action (or None)
        """
        # TODO: Assignment 4, Task 2.2
        raise NotImplementedError

    def take_action(self):
        sel_mp = self.selection_policy()
        #print("Selected MP: ", sel_mp)
        if sel_mp is not None:
            next_c = sel_mp.start.c + sel_mp.dir
            next_state = PointRobotState.from_cell(next_c, sel_mp.start.res)
            self.update_state(next_state)


class FrontierPlanner(ExplorationPlanner):
    def __init__(self, map: Grid3D, state: PointRobotState):
        super().__init__(map, state)
        # TODO: Assignment 4, Task 3.1

    def selection_policy(self):
        """Select a feasible motion primitive out of self.mpl using a frontier-based exploration method.

        Returns:
            mp: (SimplePrimitive or None) Chosen action (or None)
        """
        # TODO: Assignment 4, Task 3.1
        raise NotImplementedError

class MIPlanner(ExplorationPlanner):
    def __init__(self, map: Grid3D, state: PointRobotState):
        super().__init__(map, state)
        self.sensor = Sensor(max_range=2.0, max_height=2.0, num_rays=50)

    def compute_mi(self, pos: Point) -> Tuple[float, bool]:
        """Compute the mutual information (MI) gain by observing at the input position `pos` in the map.
        In other words, for all sensor rays emitted from the given position, sum up the information gain
        from all the observed cells.
        Hints: 
            - use self.map.cell_entropy(cell) to calculate the entropy of a cell in the map.
            - since we assume a perfect sensor model, the current entropy of the cell is the same as 
                the information gain (i.e. observation eliminates all entropy).

        Args:
            pos (Point): the robot's position in the map.

        Returns:
            Tuple[float, bool]: Return (1) the information gain and (2) a boolean indicating whether the 
                input position is valid.
        """
        pass

    def selection_policy(self):
        """Select a feasible motion primitive out of self.mpl using a information-theoretic exploration method.

        Returns:
            mp: (SimplePrimitive or None) Chosen action (or None)
        """
        # TODO: Assignment 4, Task 4.2
        raise NotImplementedError