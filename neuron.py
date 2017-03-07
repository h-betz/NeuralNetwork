import numpy as np
import random
from matplotlib import pyplot as plt

global Pref, Pmin, Pth, D, Prest
Pref = 0
Prest = 0
Pmin = -1
Pth = 5.5
D = 0.5

class neuron:
	def __init__(self):
		self.Pth = Pth
		self.t_ref = 4
		self.t_rest = -1
		self.P = Prest
		self.D = D
		self.Pmin = Pmin
		self.Prest = Prest

	def izh_simulation(self, a, b, c, d, time_ita, current, v_init):
		# a,b,c,d parameters for Izhikevich model
		# time_ita time iterations for euler method
		# current list of current for each time step
		# v_init initial voltage
		v = v_init
		u = v * b
		v_plt = np.zeros(time_ita)
		u_plt = np.zeros(time_ita)
		spike = np.zeros(time_ita)
		tstep = 0.1  # ms
		ita = 0
		while ita < time_ita:
			v_plt[ita] = v
			u_plt[ita] = u
			v += tstep * (0.04 * (v ** 2) + 5 * v + 140 - u + current[ita])
			u += tstep * a * (b * v - u)
			if v > 30.:
				spike[ita] = 1
				v = c
				u += d
			ita += 1
		time = np.arange(time_ita) * tstep
		return time, v_plt, spike