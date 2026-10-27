#!/usr/bin/env python

from numpy import hstack

class Trajectory(object):

    def __init__(self, q, qd, qdd, duration):
        self.duration = duration
        self.q = q
        self.qd = qd
        self.qdd = qdd

    def state(self, t):
        return hstack([self.q(t), self.qd(t)])

    @property
    def start_state(self):
        return self.state(0.)

    @property
    def end_state(self):
        return self.state(self.duration)

    @staticmethod
    def bezier_interpolate(state0, state1, T=1.):
        """Bezier interpolation between (q0, qd0) and (q1, qd1)."""
        n = len(state0) / 2
        q0, q1 = state0[:n], state1[:n]
        qd0, qd1 = state0[n:], state1[n:]
        Dq = q1 - q0
        c3 = (- 2 * Dq / T + qd0 + qd1) / T ** 2
        c2 = (3 * Dq / T - qd1 - 2 * qd0) / T
        c1 = qd0
        c0 = q0
        traj = Trajectory(
            q=lambda t: c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0,
            qd=lambda t: 3 * c3 * t ** 2 + 2 * c2 * t + c1,
            qdd=lambda t: 6 * c3 * t + 2 * c2,
            duration=T)
        return traj

    @staticmethod
    def soc_interpolate(state0, state1):
        """Discrete-acceleration compliant interpolation for 1-DOF systems."""
        q0, q1 = state0[:1], state1[:1]
        qd0, qd1 = state0[1:], state1[1:]
        Delta_q = q1 - q0
        Delta_qd = qd1 - qd0
        qd_avg = .5 * (qd0 + qd1)
        Delta_t = float(Delta_q / qd_avg)
        qdd_disc = Delta_qd / Delta_t
        if Delta_t < 0:
            return None
        traj = Trajectory(
            q=lambda t: q0 + min(t, Delta_t) * (
                qd0 + .5 * min(t, Delta_t) * qdd_disc),
            qd=lambda t: qd0 + min(t, Delta_t) * qdd_disc,
            qdd=lambda t: qdd_disc * (t < Delta_t),
            duration=Delta_t)
        return traj
