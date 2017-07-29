#!/usr/bin/env python
"""
Agree & Pursue Algorithm
Distributed Control of Robotic Networks.
http://coordinationbook.info
Page: 156
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.animation import FuncAnimation


class Robots:
    """
    uid:   robot ID. [list]
    dirm:  direction of movement. [List]
    pos:   angle location. [list]
    rcomm: communication range.
    k:     proportional constant for control.
    """
    def __init__(self, uid, dirm, pos, rcomm, k):
        self.uid    = uid.copy()
        self.dirm   = dirm.copy()
        self.pos    = pos.copy()

        self.max_id = uid.copy()
        self._rcomm = rcomm
        self._kctrl = k
        # communication buffer
        self.msg = [[] for _ in self.uid]  # null msgs for each robot
        # control
        self.u_ctrl = [0 for _ in self.uid]
        # evolution
        self.pos_b  = [self.pos.copy()]
        self.dirm_b = [self.dirm.copy()]
        self.u_b    = [self.u_ctrl.copy()]

    def broadcast(self):
        """
        Send message (theta, dir, max_id)
        """
        nr      = len(self.uid)
        msg_tmp = [[] for _ in range(nr)]
        # send
        for r_tx in range(nr):
            for r_rx in range(r_tx + 1, nr):
                dr = abs(self.pos[r_tx] - self.pos[r_rx])
                if dr <= self._rcomm:
                    # message format: (position, direction, id)
                    msg_tmp[r_rx].append((self.pos[r_tx], self.dirm[r_tx], self.max_id[r_tx]))
                    msg_tmp[r_tx].append((self.pos[r_rx], self.dirm[r_rx], self.max_id[r_rx]))

        # update
        self.msg = msg_tmp.copy()

    def stf(self):
        """
        state-transition function.
        """
        nr = len(self.uid)
        for i in range(nr):
            new_id  = None
            new_dir = None
            for msg in self.msg[i]:
                pos_rcvd = msg[0]
                dir_rcvd = msg[1]
                id_rcvd  = msg[2]
                if (id_rcvd > self.max_id[i]) and ((dist_cc(self.pos[i], pos_rcvd) <= self._rcomm and dir_rcvd == 'c') or (dist_c(self.pos[i], pos_rcvd) <= self._rcomm and dir_rcvd == 'cc')):
                    new_id  = id_rcvd
                    new_dir = dir_rcvd

            if new_id is not None:
                self.max_id[i] = new_id
                self.dirm[i]   = new_dir

        # log data
        self.dirm_b.append(self.dirm.copy())

    def ctl(self):
        """
        control function
        Try to match the velocity of the nearest neighbor.
        """
        nr = len(self.uid)
        for i in range(nr):
            d_tmp = self._rcomm
            u_tmp = (1 if self.dirm[i] == 'cc' else -1) * self._kctrl * d_tmp
            for msg in self.msg[i]:
                pos_rcvd = msg[0]
                pos      = self.pos[i]
                if self.dirm[i] == 'cc' and dist_cc(pos, pos_rcvd) < d_tmp:
                    d_tmp = dist_cc(pos, pos_rcvd)
                    u_tmp = self._kctrl * d_tmp
                elif self.dirm[i] == 'c' and dist_c(pos, pos_rcvd) < d_tmp:
                    d_tmp = dist_c(pos, pos_rcvd)
                    u_tmp = -(self._kctrl * d_tmp)

            self.u_ctrl[i] = u_tmp

        # log data
        self.u_b.append(self.u_ctrl.copy())

    def move(self):
        """
        Move the robot.
        Because the ctl function is data-sampled, this does not need a dt. Assume the same time interval: 1 simulation step.
        """
        nr = len(self.uid)
        for i in range(nr):
            tmp = self.pos[i] + self.u_ctrl[i]
            # wrap position: [-pi, pi]
            self.pos[i] = (tmp + np.pi) % (2*np.pi) - np.pi

        # log data
        self.pos_b.append(self.pos.copy())

    def Simulate(self, step_phy):
        self.broadcast()
        self.stf()
        self.ctl()
        self.move()


def dist_c(orig, dest):
    """
    Clockwise angular distance
    """
    dpi = 2 * np.pi
    dist = (orig - dest) % dpi
    assert dist >= 0
    return dist


def dist_cc(orig, dest):
    """
    CounterClockwise angular distance
    """
    dpi = 2 * np.pi
    dist = (dest - orig) % dpi
    assert dist >= 0
    return dist


def create_robots(nrobot, rcomm, k):
    # create robots in the unit circle.
    # Assume robots with no body.
    uid = [i for i in range(nrobot)]
    dirm = np.random.choice(['c', 'cc'], size=nrobot).tolist()
    pos = np.random.uniform(-np.pi, np.pi, nrobot).tolist()
    return Robots(uid, dirm, pos, rcomm, k)


def plot_robots(robots, line):
    """
    Plot the robots using scatter
    """
    sc_area = np.pi * (7.5)**2
    rx      = np.cos(robots.pos)
    ry      = np.sin(robots.pos)
    colors  = ['red' if x == 'cc' else 'blue' for x in robots.dirm]
    line.axes.scatter(rx, ry, alpha=1, linewidths=2, s=sc_area, c='white', edgecolors=colors)
    line.axes.set_xlim(-1.25, 1.25)
    line.axes.set_ylim(-1.25, 1.25)
    line.axes.set_aspect('equal')
    line.axes.get_xaxis().set_visible(False)
    line.axes.get_yaxis().set_visible(False)
    line.axes.set_frame_on(False)

    # Do the bloody legend.
    # I #$% miss MATLAB
    dir_legend = ['CounterClockwise', 'Clockwise']
    circ       = [mp.Circle((0, 0), fc='r'), mp.Circle((0, 0), fc='b')]
    line.axes.legend(circ, dir_legend)


def update(frame, robots, line):
    """
    Updater for the plot animation.
    """
    robots.Simulate(10)
    # plot
    line.axes.clear()
    plot_robots(robots, line)
    return line,


def main():
    # ----------------------------------------------------------------------------------------------
    # Simulation parameters
    nr     = 45
    rcomm  = (2*np.pi)/(0.9*nr)
    kprop  = 3/16
    robots = create_robots(nr, rcomm, kprop)
    nsteps = 500

    # ----------------------------------------------------------------------------------------------
    # create plot
    mpl.rcParams['toolbar'] = 'None'
    fig  = plt.figure(figsize=(7.5, 7.5))
    line, = plt.plot([], [])
    plot_robots(robots, line)
    plt.tight_layout()
    plt.axis('off')
    plt.suptitle('Agree & Pursue algorithm')

    # ----------------------------------------------------------------------------------------------
    # Animation
    animation = FuncAnimation(fig, update, fargs=(robots, line), frames=nsteps, interval=60, repeat=False)
    if False:
        animation.save("apl_{0}.gif".format(nr), dpi=80, writer='imagemagick')
    plt.show()


if __name__ == '__main__':
    main()

# Local Variables:
# flycheck-flake8-maximum-line-length: 120
# flycheck-flake8rc: ".flake8rc"
# End:
