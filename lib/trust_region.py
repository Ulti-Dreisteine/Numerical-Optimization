# -*- coding: utf-8 -*-
"""
Created on 2020/4/15 14:52

@Project -> File: numerical-optimization -> trust_region.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 置信域优化（Trust Region, TR）
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import numpy as np
import sys, os

sys.path.append('../')

from mod.mathematics.partial_derives import NumPartialDerives


def local_quadratic_approx_func(func, x: np.ndarray, p: np.ndarray) -> np.ndarray:
	"""
	获得func的Taylor展开局部二次型近似函数
	:param func: function, 函数对象, 输出为一维np.array
	:param x: np.ndarray, 自变量, shape = (N, )
	:param p: np.ndarray, 局部上的扰动, shape = (N, )
	"""
	p = p.reshape(-1, 1)
	npd = NumPartialDerives(func, x)
	g = npd.solve_jacobian_matrix()  # Jacobian matrix, shape = (1, N)
	B = npd.solve_hessian_matrix()   # Hessian matrix, shape = (N, N)
	m = func(x).flatten() + np.dot(g, p).flatten() + 0.5 * np.dot(np.dot(p.T, B), p).flatten()
	return m


class TrustRegionOptim(object):
	"""一维函数置信域优化"""
	
	def __init__(self, func, x0: np.ndarray, max_iter: int = 100, tol_x: float = 1e-6):
		try:
			assert type(func(x0)) == np.ndarray
			assert len(func(x0).flatten()) == 1  # 必须是一维函数
		except:
			raise ValueError('function output is not np.ndarray or is not one-dimensional')
		
		try:
			assert type(x0) == np.ndarray
		except:
			raise ValueError('x0 is not np.ndarray')
		
		self.func = func
		self.x0 = np.array(x0, dtype = np.float64).flatten()
		self.max_iter = max_iter
		self.tol_x = tol_x
	
	@time_cost
	def optimize(self, h_init: float = 0.1, verbose: bool = False):
		h = h_init
		x: np.ndarray = self.x0
		for i in range(self.max_iter):
			y: np.ndarray = self.func(x)
			
			# 计算当前的Jacobian和Hessian矩阵.
			npd_ = NumPartialDerives(self.func, x)
			g = npd_.solve_jacobian_matrix()  # Jacobian, shape = (1, N)
			B = npd_.solve_hessian_matrix()  # Hessian, shape = (N, N)
			
			# 计算1阶和2阶近似的最优迭代向量解
			pU = - np.dot((np.dot(g, g.T) / np.dot(np.dot(g, B), g.T))[0, 0], g.T)  # 1阶近似迭代向量
			pB = -np.dot(np.linalg.inv(B), g.T)  # 2阶近似迭代向量
			
			# Dogleg法求解最优迭代向量ps
			norm_pB, norm_pU = np.linalg.norm(pB, 2), np.linalg.norm(pU, 2)
			
			ps = None
			if norm_pB <= h:
				ps = pB
			elif norm_pB > h:
				if norm_pU <= h:
					pB_U = pB - pU
					a = np.dot(pU.T, pB_U)[0, 0]
					b = np.dot(pB_U.T, pB_U)[0, 0]
					c = np.dot(pU.T, pU)[0, 0]
					d = np.dot(pB_U.T, pB_U)[0, 0]
					tau = (-a + np.power(pow(a, 2) - b * (c - pow(h, 2)), 0.5)) / d
					ps = pU + tau * pB_U
				elif norm_pU > h:
					ps = (h / norm_pU) * pU
			else:
				raise RuntimeError('error in calculating p*!')
			
			x_new = np.array(x).astype(np.float64).reshape(-1, 1) + ps
			x_new = x_new.flatten()
			
			# 信赖域半径更新
			y_new = func(x_new)  # 实际更新值
			m = local_quadratic_approx_func(func, x, ps)  # 根据局部近似得出最优值
			
			y_new, m = y_new[0], m[0]
			if y == m:
				r = np.inf
			else:
				r = (y - y_new) / (y - m)
			
			if r == np.inf:
				h = h
				x_new = x_new
			else:
				if r >= 0:
					if 0 <= r < 0.25:
						h = h / 4
					elif 0.25 < r <= 0.75:
						h = h
					elif r > 0.75:
						h = 2 * h
					x_new = x_new
				else:
					h = h
					x_new = x
			
			x = x_new
			
			if verbose:
				print('current x: {}, h: {}'.format(x, h))
				
		return x
		

if __name__ == '__main__':
	# %% 测试参数.
	def func(x: np.ndarray) -> np.ndarray:
		return np.array([np.power(x[0] - 2, 2) + np.power(x[1], 2)])
	
	x0 = np.array([6, 20])
	
	# %% 测试函数.
	self = TrustRegionOptim(func, x0)
	x_opt = self.optimize()



