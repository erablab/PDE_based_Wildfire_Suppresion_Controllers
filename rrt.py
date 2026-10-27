#!/usr/bin/env python

import numpy
import pylab
import sys

from trajectory import Trajectory


class RRT(object):

    class Node(object):

        def __init__(self, state, parent, traj_goal=None, traj_duration=None):
            self.parent = parent
            self.state = state
            self.traj_duration = traj_duration
            self.traj_goal = traj_goal

    def __init__(self, robot, init_state, goal_state, state_interpolator,
                 steer_to_goal_every, k=10):
        self.goal_state = numpy.array(goal_state)
        self.init_state = init_state
        self.interpolate = state_interpolator
        self.k = k
        self.nb_iter = 0
        self.dist_to_goal = robot.state_dist(init_state, goal_state)
        self.plot_data = [(0, self.dist_to_goal)]
        self.nodes = [self.Node(self.init_state, parent=None)]
        self.robot = robot
        self.steer_to_goal_every = steer_to_goal_every
        self.solution = None
        self.state_dist = robot.state_dist

    def add_node(self, node):
        self.nodes.append(node)
        self.dist_to_goal = min(
            self.dist_to_goal,
            self.state_dist(node.state, self.goal_state))

    def update_plot_data(self):
        if self.dist_to_goal < self.plot_data[-1][1] - 1e-10:
            self.plot_data.append((self.nb_iter, self.dist_to_goal))

    def run(self, nb_iter):
        for itnum in range(nb_iter):
            if self.solution is not None:
                return
            self.step(self.robot.sample_state())

    def step(self, sampled_state):
        if self.solution is not None:
            return
        self.nb_iter += 1
        #sys.stderr.write("RRT iteration %6d (%5d nodes)\n" % (self.nb_iter, len(self.nodes)))
        target_state = sampled_state
        ext_node = self.extend(target_state)
        if ext_node is None:
            self.update_plot_data()
            return
        self.add_node(ext_node)
        goal_node = None
        if self.nb_iter % self.steer_to_goal_every == 0:
            goal_node = self.steer(ext_node, self.goal_state)
        if goal_node is not None:
            self.add_node(goal_node)
            if self.state_dist(goal_node.state, self.goal_state) < 1e-3:
                #sys.stderr.write("Solution found!\n")
                self.solution = self.get_trajectory_list(goal_node)
                self.plot_data.append((self.nb_iter, 0.))
                return
        self.update_plot_data()

    def extend(self, target_state):
        # Fast k-NN selection for the current planner:
        # Robot2D_SpaceCBF.state_dist() uses position-only Euclidean distance
        # in normal mode, so we can vectorize this part.
        target_xy = numpy.asarray(target_state[:2], dtype=float)

        n_nodes = len(self.nodes)
        k = min(self.k, n_nodes)
        if k <= 0:
            return None

        node_xy = numpy.array([node.state[:2] for node in self.nodes], dtype=float)  # (N, 2)
        d2 = numpy.sum((node_xy - target_xy) ** 2, axis=1)

        if k < n_nodes:
            idx = numpy.argpartition(d2, k - 1)[:k]
        else:
            idx = numpy.arange(n_nodes)

        k_nearest_nodes = [self.nodes[i] for i in idx]

        trials = [self.steer(node, target_state) for node in k_nearest_nodes]
        trials = [x for x in trials if x is not None]
        if not trials:
            return None

        # pick the reached node closest to target
        def trial_dist(node):
            dx = node.state[0] - target_xy[0]
            dy = node.state[1] - target_xy[1]
            return dx * dx + dy * dy

        return min(trials, key=trial_dist)


    def steer(self, node, target_state):
        traj = self.interpolate(node.state, target_state)
        if traj is None:
            return None
        feas_duration = 0.
        for t in numpy.linspace(0, traj.duration, 3): # just check endpoints&mid for speed. default was 100.
            q, qd, qdd = traj.q(t), traj.qd(t), traj.qdd(t)
            with self.robot.env:
                if not self.robot.check_torque_limits(q, qd, qdd):
                    break
            feas_duration = t
        if feas_duration > traj.duration / 10.:
            traj.duration = feas_duration
            reached_state = traj.end_state
            return self.Node(reached_state, node, target_state, feas_duration)
        return None

    def get_traj_from_parent(self, node):
        traj = self.interpolate(node.parent.state, node.traj_goal)
        traj.duration = node.traj_duration
        return traj

    def get_trajectory_list(self, node):
        if node.parent is None:
            return []
        traj = self.get_traj_from_parent(node)
        return self.get_trajectory_list(node.parent) + [traj]

    def plot_roadmap(self, lw=5, fontsize=18, traj_steps=100, markersize=4):
        pylab.ion()
        pylab.clf()
        pylab.plot(
            [self.init_state[0]], [self.init_state[1]], 'g*',
            markersize=50)
        pylab.plot(
            [self.goal_state[0]], [self.goal_state[1]], 'r*',
            markersize=50)
        pylab.grid(True)
        pylab.xlabel("$\\theta$", fontsize=fontsize)
        pylab.ylabel("$\\dot{\\theta}$", fontsize=fontsize)
        for node in self.nodes:
            pylab.plot([node.state[0]], [node.state[1]], 'go',
                       markersize=markersize)
        if self.solution:
            for traj in self.solution:
                dt = traj.duration / traj_steps
                for t in numpy.arange(0, traj.duration, dt):
                    pylab.plot(
                        [traj.q(t), traj.q(t + dt)],
                        [traj.qd(t), traj.qd(t + dt)], 'r-', lw=lw)
        velocity_limits = self.robot.qd_max
        torque_limits = self.robot.torque_limits
        pylab.ylim(-velocity_limits[0], +velocity_limits[0])
        is_soc = self.interpolate == Trajectory.soc_interpolate
        label = 'SOC' if is_soc else 'Bezier'
        pylab.title("Interpolation: %s, torque limit: %.1f Nm"
                    % (label, torque_limits[0]))

    def show_solution(self):
        if not self.solution:
            print("This RRT has found no solution")
            return
        self.robot.env.SetViewer('qtcoin')
        viewer = self.robot.env.GetViewer()
        viewer.SetBkgndColor([.8, .85, .9])
        viewer.SetCamera([
            [0.,  0., -1., 0.8],
            [1.,  0.,  0., 0.],
            [0., -1.,  0., 0.],
            [0.,  0.,  0., 1.]])
        self.robot.play_trajectory_list(self.solution)
