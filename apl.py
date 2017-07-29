#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

"""
Agree & Pursue Algorithm
Distributed Control of Robotic Networks.
http://coordinationbook.info
Page: 156
"""


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
        # evolution
        self.pos_t  = [self.pos.copy()]
        self.dirm_t = [self.dirm.copy()]
        # comm
        self.msg = [[] for _ in self.uid]  # null msgs for each robot
        # control
        self.u_ctrl = [0 for _ in self.uid]

    def broadcast(self):
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
        self.dirm_t.append(self.dirm.copy())

    def ctl(self):
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

    def move(self, dt):
        # d_thetha = u_control
        # so: d_pos = theta + dt * d_thetha
        nr = len(self.uid)
        for i in range(nr):
            self.pos[i] = self.pos[i] + self.u_ctrl[i] * dt
        # self.pos_t.append(self.pos.copy())

    def Simulate(self, nstep=2000, subdiv_step=60):
        for i in range(1, nstep):
            self.broadcast()
            self.stf()
            self.ctl()
            for _ in range(subdiv_step):
                self.move(1.0 / subdiv_step)  # each step is a unit of "time".


def dist_c(orig, dest):
    dpi = 2 * np.pi
    dist = (orig - dest) % dpi
    assert dist >= 0
    return dist


def dist_cc(orig, dest):
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


def plot_robots(robots, radius=1):
    # circle (working space)
    _th = np.arange(0, 2 * np.pi, 0.01)
    x = radius * np.cos(_th)
    y = radius * np.sin(_th)

    # Crear figures for the "robots"
    sc_area = np.pi * (7.5)**2
    rx = radius * np.cos(robots.pos)
    ry = radius * np.sin(robots.pos)
    colors = ['red' if x == 'cc' else 'blue' for x in robots.dirm]  # R = CounterClockwise, B = Clockwise

    # do the plot
    plt.figure()
    plt.scatter(rx, ry, alpha=1, linewidths=2, s=sc_area, c='white', edgecolors=colors)
    for i in range(len(rx)):
        plt.text(rx[i], ry[i], str(i))
    plt.plot(x, y, alpha=0.5)
    plt.axis('equal')
    plt.axis('off')


def plot_direction(nr, robots):
    data = np.array(robots.dirm_t)
    x = np.arange(data.shape[0])
    plt.figure()
    for i in range(nr):
        plt.plot(x, [1 if x == 'cc' else -1 for x in data[:, i]])


if __name__ == '__main__':
    nr     = 20
    rcomm  = (2*np.pi)/(0.9*nr)
    kprop  = 3/16
    robots = create_robots(nr, rcomm, kprop)
    rb = robots.__dict__  # stupid spyder
    nstep  = 1000

    plot_robots(robots)
    robots.Simulate(nstep=nstep)
    plot_robots(robots)
    plot_direction(nr, robots)
    plt.show()

# Local Variables:
# flycheck-flake8-maximum-line-length: 120
# flycheck-flake8rc: ".flake8rc"
# End:
