# -*- coding: utf-8 -*-
"""误码统计与指标计算。"""

import numpy as np


def count_errors(ref: np.ndarray, test: np.ndarray) -> int:
    """逐位比较，返回不一致的比特数。"""
    return int(np.count_nonzero(np.asarray(ref) != np.asarray(test)))


def bit_error_rate(ref: np.ndarray, test: np.ndarray) -> float:
    """返回误比特率 BER。"""
    ref, test = np.asarray(ref), np.asarray(test)
    if ref.size == 0:
        return 0.0
    return float(np.count_nonzero(ref != test) / ref.size)


def error_positions(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    """返回出错位置的下标数组。"""
    return np.flatnonzero(np.asarray(ref) != np.asarray(test))
