import numpy as np

# Type of planner
POINT_PLANNER=0; TRAJECTORY_PLANNER=1

# parabola formula
def parabola(x):
    return x**2

# sigmoid formula
def sigmoid(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


class planner:
    def __init__(self, type_):

        self.type=type_

    # Decide which planner to use
    def plan(self, goalPoint=[-1.0, -1.0]):
        
        if self.type==POINT_PLANNER:
            return self.point_planner(goalPoint)
        
        elif self.type==TRAJECTORY_PLANNER:
            return self.trajectory_planner()

    # Return just a point for point planner
    def point_planner(self, goalPoint):
        x = goalPoint[0]
        y = goalPoint[1]
        return x, y

    # TODO Part 6: Implement the trajectories here
    def trajectory_planner(self):
        # parabola
        # x_parabola = np.linspace(0, 1.5, 15)
        # out_space = np.column_stack((x_parabola, parabola(x_parabola)))

        # sigmoid
        x_sigmoid = np.linspace(0, 2.5, 25)
        out_space = np.column_stack((x_sigmoid, sigmoid(x_sigmoid)))

        # the return should be a list of trajectory points: [ [x1,y1], ..., [xn,yn]]
        # cast numpy array back into list for output trajectory points
        return list(out_space)

