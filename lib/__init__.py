# -*- coding: utf-8 -*-
"""
Created on 2020/4/14 18:18

@Project -> File: numerical-optimization -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys, os

sys.path.append('../')

from mod.config.config_loader import config
from mod.mathematics.partial_derives import NumPartialDerives

proj_dir, proj_cmap = config.proj_dir, config.proj_cmap

__all__ = ['NumPartialDerives', 'proj_dir', 'proj_cmap']



