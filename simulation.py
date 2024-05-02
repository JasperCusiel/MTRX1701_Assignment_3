import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import transforms


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


def sensor_positions(X_k, R, theta):
    # function: sensor_positions
    # inputs:
    #   X_k - vehicle state at time k
    #   R - range from centre of robot to sensor
    #   theta - angle from centre of robot to sensor
    # returns:
    #   s_r, s_l - the positions of the right and left sensors
    r_r, r_l = relative_sensor_positions(X_k, R, theta)

    s = [X_k[0], X_k[1]]

    s_r = [(s[0] + r_r[0]), (s[1] + r_r[1])]
    s_l = [(s[0] + r_l[0]), (s[1] + r_l[1])]

    return s_r, s_l


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


def track_model(x, y, x_0, y_0, a, b, T):
    # function: track_model
    # inputs:
    #   x - x position of sensor
    #   y - y position of sensor
    #   x_0 - x position of centre of ellipse
    #   y_0 - y position of centre of ellipse
    #   a - major axis of ellipse
    #   b - minor axis of ellipse
    #   T - thickness of ellipse
    # return:
    #   < 0 if point is inside ellipse of thickness T
    #   0 if point is on ellipse of thickness T
    #   > 0 if point is outside of ellipse of thickness T
    d_inner = classify_ellipse_point(x, y, x_0, y_0, (a - (T / 2)), (b - (T / 2)))
    d_outer = classify_ellipse_point(x, y, x_0, y_0, (a + (T / 2)), (b + (T / 2)))

    if d_outer > 0:
        signed_distance = d_outer
    elif d_inner < 0:
        signed_distance = d_inner
    else:
        signed_distance = 0

    return signed_distance


def control_model(d_r, d_l, v_max, distance_between_sensors, use_proportional = False):
    # function: control_model
    # inputs:
    #   d_r - distance from right sensor to track
    #   d_l - distance from left sensor to track
    #   v_max - maximum velocity
    #   distance_between_sensors - physical distance between the sensors on the robot
    #   use_proportional - set True to use the proportional controller
    kp = 0.055      # Proportional constant

    if use_proportional:
        v_r = v_max - (kp * v_max * (d_l / distance_between_sensors))
        v_l = v_max - (kp * v_max * (d_r / distance_between_sensors))
    else:
        if d_l == 0:
            v_l = 0
            v_r = v_max
        elif d_r == 0:
            v_r = 0
            v_l = v_max
        else:
            v_r = v_max
            v_l = v_max

    return v_r, v_l


def simulate(ts, dt, X, robot_d, r, th, x0, y0, a, b, T, vmax, distance_between_sensors):
    Xs = []
    for t in ts:
        s_r, s_l = sensor_positions(X, r, th)
        d_r = track_model(s_r[0], s_r[1], x0, y0, a, b, T)
        d_l = track_model(s_l[0], s_l[1], x0, y0, a, b, T)
        v_r, v_l = control_model(d_r, d_l, vmax, distance_between_sensors, True)

        u = control_vector(v_r, v_l, robot_d)

        X = vehicle_model(X, u, dt)
        # store all the variables so you can prepare figures
        Xk = np.concatenate(([t], X, u, s_r, s_l, [d_r, d_l, v_r, v_l]), axis=0)
        Xs.append(Xk)

    return np.array(Xs)


if __name__ == '__main__':
    ####################################################################################
    #                            Simulation Parameters                                 #
    ####################################################################################
    dt = 0.02  # timestep (s)
    t_start = 0  # time start (s)
    t_end = 10   # time end (s)
    t = np.arange(t_start, t_end, dt)  # List of evenly spaced timestamps

    # Robot physical dimensions
    robot_length = 0.105  # m
    robot_width = 0.07  # m
    """
      Distance from the bottom left of the robot to the center of mass
          +------------------+
          |                  |
    COM_offset_y   COM       |
          |                  |
         (xy)- COM_offset_x -+ """

    COM_offset_x = 0.04  # m
    COM_offset_y = robot_width / 2  # m

    # Robot sensor positions
    sensor_to_COM_distance = 0.055  # m
    distance_between_sensors = 0.021  # m
    r = np.sqrt(sensor_to_COM_distance ** 2 + (distance_between_sensors / 2) ** 2)  # m
    th = np.arctan2(distance_between_sensors / 2, sensor_to_COM_distance)  # rad

    # Distance between COM and the wheel rotation axis
    d = 0.04  # m
    # Maximum wheel speed
    r_wheel = 0.02  # Wheel radius (m)
    omega = (24 * np.pi) / 5  # Maximum wheel angular velocity (rad/s)
    v_max = omega * r_wheel  # m/s

    # Vehicle vectors
    u = np.zeros(2)
    X = np.zeros(3)  # Vehicle initial pose of COM

    # Track ellipse parameters
    ellipse_a = 0.18  # m
    ellipse_b = 0.1  # m
    ellipse_origin_x = 0  # m
    ellipse_origin_y = ellipse_b  # m
    ellipse_thickness = 0.015  # m

    ####################################################################################
    #                                  Plot Setup                                      #
    ####################################################################################

    # Create new figure with a single subplot
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    plt.rcParams["font.family"] = "serif"
    plot, = ax.plot([], [], 'c--', label="Center Of Mass Path")
    subtitle = "t = {}s with v = {}cm/s, d = {}cm, w = {}cm, l = {}cm, Ellipse a = {}cm & b = {}cm".format(t_end, round(v_max * 100, 2), round(d * 100, 2), round(robot_width * 100, 2), round(robot_length * 100, 2), ellipse_a * 100, ellipse_b * 100)
    plt.suptitle(t=subtitle, y=0.05, size='small')
    fig.subplots_adjust(bottom=0.17)

    # Set plot limits
    plt.xlim(-20, 20)
    plt.ylim(-10, 25)

    # Add labels with font
    plt.xlabel("x (cm)", fontname='Times New Roman')
    plt.ylabel("y (cm)", fontname='Times New Roman')

    plt.title(label="Robot Center Of Mass Path", fontdict={'family': 'serif', 'color': 'black', 'size': 10}, weight='bold', pad=20)

    # Set the font to Times New Roman for axis numbers
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    ####################################################################################
    #                                Run Simulation                                    #
    ####################################################################################

    sim_data = simulate(t, dt, X, d, r, th, ellipse_origin_x, ellipse_origin_y, ellipse_a, ellipse_b, ellipse_thickness,
                        v_max, distance_between_sensors)

    # Pull data from simulation and store in lists to make it easier to access
    sim_x_values = sim_data[:, 1]
    sim_y_values = sim_data[:, 2]
    sim_heading_angles = sim_data[:, 3]
    sim_sensor_right_position_x = sim_data[:, 6]
    sim_sensor_right_position_y = sim_data[:, 7]
    sim_sensor_left_position_x = sim_data[:, 8]
    sim_sensor_left_position_y = sim_data[:, 9]

    plot_x_data = []
    plot_y_data = []

    ####################################################################################
    #                             Add Graphics to Plot                                 #
    ####################################################################################

    # Add in two ellipses to draw track outline
    outer_ellipse = patches.Ellipse((ellipse_origin_x * 100, ellipse_origin_y * 100),
                                    (ellipse_a * 200 + (ellipse_thickness * 100 / 2)),
                                    (ellipse_b * 200 + (ellipse_thickness * 100 / 2)), edgecolor='None', fc='black',
                                    lw=2)
    ax.add_patch(outer_ellipse)

    inner_ellipse = patches.Ellipse((ellipse_origin_x * 100, ellipse_origin_y * 100),
                                    (ellipse_a * 200 - (ellipse_thickness * 100 / 2)),
                                    (ellipse_b * 200 - (ellipse_thickness * 100 / 2)), edgecolor='None', fc='white',
                                    lw=2)
    ax.add_patch(inner_ellipse)

    # Add in robot outline as a rectangle
    robot_body = Rectangle(xy=(0, 0), width=(robot_length * 100), height=(robot_width * 100),
                           facecolor='none', edgecolor='black')
    ax.add_patch(robot_body)

    # Add sensor dots
    left_sensor = patches.Circle(
        (float(sim_sensor_left_position_x[0] * 100), float(sim_sensor_left_position_y[0] * 100)), radius=0.1,
        color='orange', label='Left Sensor')
    ax.add_patch(left_sensor)
    right_sensor = patches.Circle(
        (float(sim_sensor_right_position_x[0] * 100), float(sim_sensor_right_position_y[0] * 100)), radius=0.1,
        color='green', label='Right Sensor')
    ax.add_patch(right_sensor)

    # Add center of mass marker
    com_marker, = ax.plot(*(0, 0), marker="+", markersize=10, color='black', label="Center Of Mass")

    plt.legend(loc='upper right', handles=[left_sensor, right_sensor, com_marker, plot], fontsize=6)

    ####################################################################################
    #                              Animation Creation                                  #
    ####################################################################################

    # FFMpeg Writer Settings
    metadata = dict(title='Robot Simulation', artist='Jasper Cusiel')
    writer = FFMpegWriter(fps=50, metadata=metadata)

    with writer.saving(fig, 'simulation.mp4', 300):
        for i in range(1, len(sim_x_values)):
            plot_x_data.append(sim_x_values[i] * 100)
            plot_y_data.append(sim_y_values[i] * 100)

            # Move center of mass marker
            com_marker.set_data([sim_x_values[i] * 100], [sim_y_values[i] * 100])

            # Move sensor dots
            right_sensor.set(center=(sim_sensor_right_position_x[i] * 100, sim_sensor_right_position_y[i] * 100))
            left_sensor.set(center=(sim_sensor_left_position_x[i] * 100, sim_sensor_left_position_y[i] * 100))

            # Move robot body rectangle
            robot_body.set_x(float(sim_x_values[i] * 100) - COM_offset_x * 100)
            robot_body.set_y(float((sim_y_values[i] * 100) - COM_offset_y * 100))

            # Rotate rectangle
            t = transforms.Affine2D().rotate_deg_around(robot_body.get_x() + COM_offset_x * 100,
                                                        robot_body.get_y() + COM_offset_y * 100,
                                                        float(np.degrees(sim_heading_angles[i - 1])))
            robot_body.set_transform(t + ax.transData)

            # Add data point to plot
            plot.set_data(plot_x_data, plot_y_data)
            writer.grab_frame()