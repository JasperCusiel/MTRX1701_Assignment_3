import numpy as np
import matplotlib
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
    #   v_l - velcocity of the left wheel
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


def control_model(d_r, d_l, v_max):
    # function: control_model
    # inputs:
    #   d_r - distance from right sensor to track
    #   d_l - distance from left sensor to track
    #   v_max - maximum velocity
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


def simulate(ts, dt, X, robot_d, r, th, x0, y0, a, b, T, vmax):
    Xs = []
    for t in ts:
        # Your code here
        s_r, s_l = sensor_positions(X, r, th)
        d_r = track_model(s_r[0], s_r[1], x0, y0, a, b, T)
        d_l = track_model(s_l[0], s_l[1], x0, y0, a, b, T)
        v_r, v_l = control_model(d_r, d_l, vmax)

        u = control_vector(v_r, v_l, robot_d)

        X = vehicle_model(X, u, dt)
        # store all the variables so you can prepare figures
        Xk = np.concatenate(([t], X, u, s_r, s_l, [d_r, d_l, v_r, v_l]), axis=0)
        Xs.append(Xk)

    return np.array(Xs)


if __name__ == '__main__':

    X = np.zeros(3)  # vehicle initial pose
    dt = 1  # s
    t_start = 0  # s
    t_end = 180  # s
    t = np.arange(t_start, t_end, dt)  # timesteps
    robot_length = 0.067  # m
    robot_width = 0.06  # m
    r = np.sqrt(robot_length ** 2 + (robot_width / 2) ** 2)  # m
    th = np.arctan2(robot_width / 2, robot_length)  # rad
    d = robot_width / 2  # m
    r_wheel = 0.025  # m
    omega = 0.25  # rad
    v_max = omega * r_wheel  # m/s
    print(v_max)
    # vehicle command
    u = np.zeros(2)  # m/s

    # ellipse parameters
    ellipse_a = 0.125  # m
    ellipse_b = 0.075  # m
    ellipse_origin_x = 0  # m
    ellipse_origin_y = ellipse_b  # m
    ellipse_thickness = 0.015  # m

    fig, ax = plt.subplots()  # Create a new figure with a single subplot
    plot, = ax.plot([], [], 'b-')
    plt.xlim(-18, 18)
    plt.ylim(-10, 25)

    metadata = dict(title='Movie', artist='jasper')
    writer = FFMpegWriter(fps=60, metadata=metadata)

    sim_data = simulate(t, dt, X, d, r, th, ellipse_origin_x, ellipse_origin_y, ellipse_a, ellipse_b, ellipse_thickness, v_max)

    sim_x_values = sim_data[:, 1]
    sim_y_values = sim_data[:, 2]
    sim_heading_angles = sim_data[:, 3]

    plot_x_data = []
    plot_y_data = []

    # add in two ellipses to draw path
    outer_ellipse = patches.Ellipse((ellipse_origin_x * 100, ellipse_origin_y * 100),
                                    (ellipse_a * 200 + (ellipse_thickness * 100 / 2)),
                                    (ellipse_b * 200 + (ellipse_thickness * 100 / 2)), edgecolor='None', fc='black', lw=2)
    ax.add_patch(outer_ellipse)

    inner_ellipse = patches.Ellipse((ellipse_origin_x * 100, ellipse_origin_y * 100),
                                    (ellipse_a * 200 - (ellipse_thickness * 100 / 2)),
                                    (ellipse_b * 200 - (ellipse_thickness * 100 / 2)), edgecolor='None', fc='white', lw=2)
    ax.add_patch(inner_ellipse)

    # add in robot outline rectangle
    robot_body = Rectangle(xy=(0, -(robot_width*100/2)), width=(robot_length * 100), height=(robot_width * 100), facecolor='none', edgecolor='black')
    ax.add_patch(robot_body)

    with writer.saving(fig, 'simulation.mp4', 200):
        for i in range(0, len(sim_x_values)):
            plot_x_data.append(sim_x_values[i] * 100)
            plot_y_data.append(sim_y_values[i] * 100)
            # if i > 60:
            #     # removes the first item to give the trailing line effect
            #     plot_x_data.pop(0)
            #     plot_y_data.pop(0)
            # Rotate Rectangle
            t = transforms.Affine2D().rotate_deg_around(robot_body.get_x(),
                                                        robot_body.get_y() + robot_body.get_height() / 2,
                                                        float(np.degrees(sim_heading_angles[i])))
            robot_body.set_transform(t + ax.transData)
            # Move rectangle to new plot point
            robot_body.set_xy((float((sim_x_values[i] * 100)), float((sim_y_values[i] * 100)-(robot_width*100/2))))
            plot.set_data(plot_x_data, plot_y_data)
            writer.grab_frame()
