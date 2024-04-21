import matplotlib.pyplot as plt
import numpy as np


def derivatives(X_k, u_k):
    # function: derivatives
    # inputs:
    #   X_k - robot state at time k
    #   u_k - control inputs
    # returns:
    #   X_dot_k - vector of state derivatives
    X_dot_k = [(u_k[0] * np.cos(X_k[2])), (u_k[0] * np.sin(X_k[2])), (u_k[1])]
    return X_dot_k


def vehicle_model(X_k, u_k, dt):
    # function: vehicle_model
    # inputs:
    #   X_k - robot state at time k
    #   u_k - control input
    #   dt - time since last update
    # returns:
    #   X_xp1 - robot state at next time k+1
    dX_k = derivatives(X_k, u_k)
    X_kp1 = [(X_k[0] + dt * dX_k[0]), (X_k[1] + dt * dX_k[1]), (X_k[2] + dt * dX_k[2])]
    return X_kp1


def relative_sensor_positions(X_k, R, theta):
    # function: relative_sensor_positions
    # inputs:
    #   X_k - vehicle state at time k
    #   R - range from centre of robot to sensor
    #   theta - angle from centre of robot to sensor
    # returns:
    #   r_r, r_l - the relative positions of the right and left sensors compared to the robot's centre
    r_r = [(R * np.cos(X_k[2] - theta)), (R * np.sin(X_k[2] - theta))]
    r_l = [(R * np.cos(X_k[2] + theta)), (R * np.sin(X_k[2] + theta))]
    return r_r, r_l


def control_vector(v_r, v_l, d):
    # function: control_vector
    # inputs:
    #   v_r - velocity of the right wheel
    #   v_l - velocity of the left wheel
    #   d - width of robot
    # returns:
    #   u_k - vector of control inputs
    v = (v_r + v_l) / 2
    psi_dot = (v_r - v_l) / (2 * d)
    u_k = [v, psi_dot]
    return u_k


def ellipse_func(x, y, x_0, y_0, a, b):
    # function: ellipse_func
    # inputs:
    #   x - x position of sensor
    #   y - y position of sensor
    #   x_0 - x position of centre of ellipse
    #   y_0 - y position of centre of ellipse
    #   a - major axis of ellipse
    #   b - minor axis of ellipse
    # return:
    #   F - F = ((x-x_0)^2)/a^2 + ((y-y_0)^2)/b^2 -1
    F = np.square((x - x_0) / a) + np.square((y - y_0) / b) - 1

    return F


def classify_ellipse_point(x, y, x_0, y_0, a, b):
    # function: classify_ellipse_point
    # inputs:
    #   x - x position of sensor
    #   y - y position of sensor
    #   x_0 - x position of centre of ellipse
    #   y_0 - y position of centre of ellipse
    #   a - major axis of ellipse
    #   b - minor axis of ellipse
    # return:
    #   < 0 if point is inside ellipse
    #   0 if point is on ellipse
    #   > 0 if point is outside of ellipse
    F = ellipse_func(x, y, x_0, y_0, a, b)

    return F


def simulation(v_r, v_l, robot_d, X):
    Xs = []
    dt = 1  # timestep (s)
    t_start = 0  # time start (s)
    t_end = 10  # time end (s)
    ts = np.arange(t_start, t_end, dt)  # List of evenly spaced timestamps
    for t in ts:
        u = control_vector(v_r, v_l, robot_d)
        X = vehicle_model(X, u, dt)
        Xk = np.concatenate(([t], X, u, [v_r, v_l]), axis=0)
        Xs.append(Xk)

    return np.array(Xs), t_end, v_l, v_r


if __name__ == '__main__':
    ####################################################################################
    #                            Simulation Parameters                                 #
    ####################################################################################

    # Distance between COM and the wheel rotation axis
    d = 0.04  # m
    # Maximum wheel speed
    r_wheel = 0.02  # Wheel radius (m)
    omega = (24 * np.pi) / 5  # Maximum wheel angular velocity (rad/s)
    v_max = omega * r_wheel  # m/s

    X = np.zeros(3)  # Vehicle initial pose of COM

    # Run simulation where both wheels are driven at v_max
    sim_data, sim_2c_t_end, v_l, v_r = simulation(v_max, v_max, d, X)

    # Pull data to pass into plotting function
    sim_x_values = sim_data[:, 1] * 100
    sim_y_values = sim_data[:, 2] * 100

    # Create new figure with a single subplot
    fig, ax = plt.subplots()
    plot, = ax.plot(sim_x_values, sim_y_values, 'c--', label="Center Of Mass Path")
    # Set plot parameters
    plt.rcParams['savefig.dpi'] = 300

    # Adjust plot limits
    plt.xlim(min(sim_x_values), max(sim_x_values))
    plt.ylim(-1, 1)

    # Add labels with font
    plt.xlabel("x (cm)", fontname='Times New Roman')
    plt.ylabel("y (cm)", fontname='Times New Roman')
    subtitle = "t = {}s with v max = {}cm/s, v_l = {}cm/s, v_r = {}cm/s, d = {}cm".format(sim_2c_t_end,
                                                                    round(v_max * 100, 2),
                                                                    round(v_l * 100, 2), round(v_r * 100, 2), round(d * 100, 2))

    plt.title(label="Question 2c", fontdict={'family': 'serif', 'color': 'black', 'size': 14}, pad=14, weight='bold')
    plt.suptitle(t=subtitle, y=0.05,fontdict={'family': 'serif', 'color': 'black'}, size='small')
    plt.legend(loc='upper right', fontsize=6)

    # Make space for bottom text
    fig.subplots_adjust(bottom=0.17)

    # Set the font to Times New Roman for the axis numbers
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    fig.savefig('question-2c-plot.png')
