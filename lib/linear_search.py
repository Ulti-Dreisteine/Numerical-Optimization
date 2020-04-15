# -*- coding: utf-8 -*-
"""
Created on 2020/4/14 18:17

@Project -> File: numerical-optimization -> linear_search.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 一维函数线性优化（Linear Search）
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import numpy as np
import warnings
import sys

sys.path.append('../')

from lib import NumPartialDerives


class LinearSearchOptim(object):
	"""一维函数线性搜索法寻优"""
	
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
	
	def _solve_alpha_grad(self, x: np.ndarray, x_search_direc: np.ndarray, alpha: float) -> float:
		"""固定x时求解alpha对应的梯度"""
		f_alpha = lambda alpha: self.func(x + x_search_direc * alpha[0])
		npd = NumPartialDerives(f_alpha, np.array([alpha]))
		alpha_grad = npd.solve_jacobian_matrix()
		return alpha_grad.flatten()[0]
	
	def _solve_alpha(self, x: np.ndarray, x_search_direc: np.ndarray, alpha_init: float = 1e-1, step_size = 1e-6,
	                 max_alpha_iter = 1000, tol_alpha = 1e-6) -> float:
		"""
		固定x在某一方向上搜索最优步长alpha
		
		Notes:
		------------
		* step_size应尽可能小以确保内层迭代收敛, 但是太小的话也可能导致计算效率降低
		"""
		alpha = alpha_init
		converged_ = False
		for i in range(max_alpha_iter):
			alpha_search_direc_ = -self._solve_alpha_grad(x, x_search_direc, alpha)
			delta_alpha_ = step_size * alpha_search_direc_
			alpha += delta_alpha_
			
			if np.abs(delta_alpha_) < tol_alpha:
				converged_ = True
				break
		
		if not converged_:
			warnings.warn(
				'Alpha does not converge during the inner {} iterations, tol_alpha = {}'.format(max_alpha_iter, tol_alpha))
		
		return alpha
	
	@time_cost
	def optimize(self, verbose: bool = False) -> (np.ndarray, list):
		"""
		进行优化
		:param verbose: bool, 是否打印并记录迭代过程
		"""
		x = self.x0
		records = []
		converged_ = False
			
		for i in range(self.max_iter):
			if verbose:
				print('iteration {}: x = {}'.format(i, x))
				records.append([i, x])
				
			# 求解负梯度搜索方向.
			npd = NumPartialDerives(self.func, x)
			x_search_direc_ = -npd.solve_jacobian_matrix().flatten()
			
			# 求解搜索步长alpha.
			alpha_ = self._solve_alpha(x, x_search_direc_)
			
			# 更新x.
			delta_x_ = x_search_direc_ * alpha_
			x += delta_x_
			
			if np.linalg.norm(delta_x_, 2) < self.tol_x:
				converged_ = True
				break
		
		if not converged_:
			warnings.warn('x does not converge in {} iterations, tol_x = {}'.format(self.max_iter, self.tol_x))
		
		return x, records


if __name__ == '__main__':
	# %% 测试参数.
	def func(x: np.ndarray) -> np.ndarray:
		return np.array([np.power(x[0] - 1, 2) + np.power(x[1], 2)])

	x0 = np.array([6, 20])

	# %% 测试函数.
	self = LinearSearchOptim(func, x0, max_iter = 100, tol_x = 1e-6)
	x_opt, records = self.optimize(verbose = True)
