# -*- coding: utf-8 -*-
"""
数字基带码型
============
整合自原 baseband_codes_simulation.py：
    NRZ  双极性不归零
    Manchester 曼彻斯特（IEEE 802.3）
    AMI  传号交替反转
    HDB3 三阶高密度双极性
"""

import numpy as np


def nrz_encode(bits) -> np.ndarray:
    """双极性不归零：1 -> +1，0 -> -1。"""
    return np.array([1 if b == 1 else -1 for b in bits], dtype=float)


def manchester_encode(bits) -> np.ndarray:
    """曼彻斯特：1 -> [+1,-1]，0 -> [-1,+1]，返回长度为输入 2 倍。"""
    out = []
    for b in bits:
        out.extend([1, -1] if b == 1 else [-1, 1])
    return np.array(out, dtype=float)


def ami_encode(bits) -> np.ndarray:
    """AMI：0 -> 0 电平；1 交替出现 +1、-1。"""
    out, pol = [], 1
    for b in bits:
        if b == 1:
            out.append(pol)
            pol = -pol
        else:
            out.append(0)
    return np.array(out, dtype=float)


def hdb3_encode(bits):
    """
    HDB3：连 0 段 0000 取代。
      距上一个 V 后信息位 1 为奇数 -> 000V；偶数（含 0）-> B00V。
    返回 (编码序列, B 脉冲位置, V 脉冲位置)。
    """
    n = len(bits)
    v_pos, b_pos = [], []
    ones_since_v = 0
    i = 0
    while i < n:
        if i + 3 < n and list(bits[i:i + 4]) == [0, 0, 0, 0]:
            if ones_since_v % 2 == 0:
                b_pos.append(i)
            v_pos.append(i + 3)
            ones_since_v = 0
            i += 4
        else:
            if bits[i] == 1:
                ones_since_v += 1
            i += 1

    v_set, b_set = set(v_pos), set(b_pos)
    out = [0] * n
    last_pol = 0
    for k in range(n):
        if k in b_set:
            pol = -last_pol if last_pol != 0 else 1   # B 遵守交替
            out[k], last_pol = pol, pol
        elif k in v_set:
            pol = last_pol if last_pol != 0 else 1    # V 与前一脉冲同极性
            out[k], last_pol = pol, pol
        elif bits[k] == 1:
            pol = -last_pol if last_pol != 0 else 1
            out[k], last_pol = pol, pol
        else:
            out[k] = 0
    return np.array(out, dtype=float), b_pos, v_pos


# 码型注册表：名称 -> 编码函数（HDB3 单独处理，因为返回元组）
LINE_CODE_FUNCS = {
    "NRZ": nrz_encode,
    "Manchester": manchester_encode,
    "AMI": ami_encode,
}
