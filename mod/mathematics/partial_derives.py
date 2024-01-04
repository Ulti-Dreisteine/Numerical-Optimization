# -*- coding: utf-8 -*-
"""
Created on 2020/3/12 11:16

@Project -> File: algorithm-tools -> partial_derives.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 计算函数偏导数, 支持符号运算SymPartialDerives和数值摄动求解NumPartialDerives两种方式
"""

import numpy as np
import sys

sys.path.append('../..')

from mod.mathematics.symbol_calculation import gen_x_syms, gen_sym_S, gen_sym_G, cal_partial_derive_syms, \
    extract_x_syms_in_func_sym, cal_sym_func_value


class SymPartialDerives(object):
    """计算隐函数各维度偏导数的符号表达式和数值"""
    
    def __init__(self, func, f_dim):
        """
        初始化
        
        params
        ------
        func: 函数必须为转换为隐函数形式输入, func(x) = 0
        f_dim: 函数里自变量的维数
        """
        
        self.F = func
        self.F_dim = f_dim
        
        # 生成变量和函数的偏导数符号.
        self.x_syms = gen_x_syms(self.F_dim)
        self.F_sym = gen_sym_S(self.F, self.x_syms)
        self.G_syms_list = gen_sym_G(self.F_sym, self.x_syms)
        self.G_sym = self.G_syms_list[0]  # TODO: 目前默认选择第一个解, 需要改为对所有解均进行计算
    
    def _cal_partial_derive_syms(self) -> list:
        """
        计算隐函数F对各x的偏导符号
        """
        
        PD_syms = cal_partial_derive_syms(self.F_sym, self.F_dim)
        return PD_syms
    
    @property
    def PD_syms(self):
        return self._cal_partial_derive_syms()
    
    def _extract_x_syms_in_PD(self) -> list:
        """
        提取PD_syms中各个符号表达式中所含自变量x符号
        """
        
        PD_have_x_syms = []
        for i in range(self.F_dim):
            PD_ = self.PD_syms[i]
            x_sym_strs_ = extract_x_syms_in_func_sym(PD_, self.F_dim)
            PD_have_x_syms.append(x_sym_strs_)
            
        return PD_have_x_syms
    
    def _extract_x_syms_in_G(self) -> list:
        G_has_x_syms = extract_x_syms_in_func_sym(self.G_sym, self.F_dim)
        return G_has_x_syms
    
    @property
    def PD_have_x_syms(self):
        return self._extract_x_syms_in_PD()
    
    @property
    def G_has_x_syms(self):
        return self._extract_x_syms_in_G()
    
    def cal_partial_derive_values(self, x: list) -> (list, list):
        """
        计算偏导数
        
        note
        ----
        x不需要输入最后x_n的值, 可以通过G函数计算, 所以 x = [x_0, x_1, ..., x_n-1]
        """
  
        x = x.copy()
        assert len(x) == self.F_dim - 1
        
        # 计算G的值.
        subs_dict_ = {}
        for x_sym_str in self.G_has_x_syms:
            subs_dict_[x_sym_str] = x[int(x_sym_str.split('_')[1])]
        x_end = cal_sym_func_value(self.G_sym, subs_dict_)
        x.append(x_end)
        
        pd_values = []
        for i in range(self.F_dim):
            subs_dict_ = {}
            for x_sym_str in self.PD_have_x_syms[i]:
                subs_dict_[x_sym_str] = x[int(x_sym_str.split('_')[1])]
            pd_ = float(self.PD_syms[i].subs(subs_dict_))
            pd_values.append(pd_)
        
        return x, pd_values


class NumPartialDerives(object):
    """
    采用摄动法计算函数func对输入x各维度上的偏导数
    """
    
    def __init__(self, func, x0: np.ndarray):
        assert isinstance(func(x0), np.ndarray)
        self.x0: np.ndarray = x0.flatten()          # 输入为一维向量
        self.y0: np.ndarray = func(x0).flatten()    # 输出为一维向量
        self.eps = 1e-6
        self.x_dim = len(self.x0)
        
    def _perturb(self, func, x: np.ndarray, pert_loc: int, kind: str = None) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        对x进行局部摄动
        
        params
        ------
        func: 待计算函数
        x: 扰动时x的位置
        pert_loc: 扰动的维数
        kind: s数值扰动类型, must be in {'relative', 'abs'}
        """
        
        x = np.array(x).flatten()
        delta_x = np.zeros_like(x).astype(np.float64)
        
        if (kind == 'relative') or (kind is None):
            delta_x[pert_loc] = self.eps * x[pert_loc]
        else:
            delta_x[pert_loc] = self.eps
        
        x_pert = x + delta_x
        y_pert = func(x_pert).flatten()  # 转为一维向量
        
        return x_pert, y_pert, delta_x
    
    def _cal_jacobian(self, func, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        求解函数func在任意x处的Jacobian矩阵
        
        params
        ------
        func: 函数
        x: 输入数组
        kwargs:
            kind: 类型,  参见self._perturb
        """
        
        assert len(x) == self.x_dim
        y = func(x).flatten()
        jacobian = None
        for dim in range(self.x_dim):
            _, y_pert, delta_x = self._perturb(func, x, dim, **kwargs)
            dy_dx = ((y_pert - y) / delta_x[dim]).reshape(-1, 1)
            
            if jacobian is None:
                jacobian = dy_dx
            else:
                jacobian = np.hstack((jacobian, dy_dx))
        return jacobian
    
    def solve_jacobian_matrix(self, **kwargs) -> np.ndarray:
        """
        求解x0处的Jacobian矩阵
        :param kwargs:
            kind: str, 参见self._perturb
            
        note
        ----
        *   Jacobian = [
                pf0 / px0, pf0 /px1, ..., pf0 / pxn,
                            ...
                pfm / px0, pfm /px1, ..., pfm / pxn,
            ]
        *	Jacobian.shape = (m + 1, n + 1)
        """
        
        jacobian = self._cal_jacobian(self.func, self.x0, **kwargs)
        
        return jacobian
    
    def solve_hessian_matrix(self, **kwargs) -> np.ndarray:
        """
        求解x0处的Hessian矩阵:
        :param kwargs:
            kind: str, 参见self._perturb
        
        note
        ------------
        * f必须为标量函数
        *   Hessian = [
                p^2f / (px0 * px0), p^2f / (px0 * px1), ..., p^2f / (px0 * pxn),
                p^2f / (px1 * px0), p^2f / (px1 * px1), ..., p^2f / (px1 * pxn),
                                            ...
                p^2f / (pxn * px0), p^2f / (px1 * px1), ..., p^2f / (pxn * pxn),
            ]
        *   Hessian.shape = (n + 1, n + 1)
        """
        
        if len(self.y0) != 1:
            raise RuntimeError('The output self.y0 is not one-dimensional, while Hessian requires an one-dim output func')
        else:
            cal_jacobian_ = lambda x: self._cal_jacobian(self.func, x, **kwargs)
            jacob_0 = cal_jacobian_(self.x0).flatten()
            hessian = None
            for dim in range(self.x_dim):
                _, jacob_pert, delta_x = self._perturb(cal_jacobian_, self.x0, dim, **kwargs)
                djacob_dx = ((jacob_pert - jacob_0) / delta_x[dim]).reshape(1, -1)
                
                if hessian is None:
                    hessian = djacob_dx
                else:
                    hessian = np.vstack((hessian, djacob_dx))
            return hessian
        
        
if __name__ == '__main__':
    
    # 测试SymPartialDerives.
    def f1(x: list):
        y = 0.5 * x[1] - x[0] ** 2
        return y

    f_dim = 2
    self = SymPartialDerives(f1, f_dim)
    x, pd_values = self.cal_partial_derive_values([1])
    
    # 测试NumPartialDerives.
    def f2(x: list):
        y = 0.5 * x[1] - x[0] ** 2
        return np.array([y])
    
    self = NumPartialDerives(f2, x0 = np.array([2, 2]))
    jacobian = self.solve_jacobian_matrix()
    hessian = self.solve_hessian_matrix()



