# -*- coding: utf-8 -*-
"""
脉冲成形滤波器
==============
整合自 early_late_gate_sim.py 的根升余弦（RRC）与 baseband_codes_simulation.py
的升余弦（RC），并补充矩形脉冲。三种脉冲统一约定：

    长度 L = span*sps + 1，时间轴以符号周期 T 归一化；
    单位能量归一化  sum(h^2) = 1。

因此收发各用一次 h（匹配滤波）后：
    * 总响应在整数符号点过零 -> 无码间串扰；
    * 总链路延迟恰好 span 个符号；
    * 采样点信号幅度 = 符号值，噪声能量保持不变。
"""

import numpy as np


def _rrc_values(beta: float, t: np.ndarray) -> np.ndarray:
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den
    return h


def _rc_values(beta: float, t: np.ndarray) -> np.ndarray:
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0
        elif np.isclose(abs(ti), 1.0 / (2.0 * beta)):
            h[i] = (beta / np.sqrt(2.0)) * (
                np.sin(np.pi / (2.0 * beta)) + np.cos(np.pi / (2.0 * beta)))
        else:
            h[i] = (np.sinc(ti) * np.cos(np.pi * beta * ti)
                    / (1.0 - (2.0 * beta * ti) ** 2))
    return h


def design_pulse(kind: str, beta: float, sps: int, span: int):
    """返回 (时间轴 t, 单位能量脉冲 h)。"""
    n_tap = span * sps + 1
    t = (np.arange(n_tap) - span * sps / 2.0) / sps

    if kind == "RRC":
        h = _rrc_values(beta, t)
    elif kind == "RC":
        h = _rc_values(beta, t)
    elif kind == "rect":
        h = np.zeros(n_tap, dtype=float)
        center = span * sps // 2
        # 关于 t=0 对称放置 sps 个点（区间 [-0.5, 0.5)），
        # 两个矩形卷积后峰值在 0、整数符号点过零，总延迟仍为 span
        start = center - sps // 2
        h[start: start + sps] = 1.0 / np.sqrt(sps)
        return t, h
    else:
        raise ValueError(f"未知脉冲类型：{kind}")

    h /= np.sqrt(np.sum(h ** 2))           # 单位能量归一化
    return t, h


def impulse_train(symbols: np.ndarray, sps: int) -> np.ndarray:
    """把符号序列扩展为冲激串（每 sps 个点放一个符号）。"""
    up = np.zeros(len(symbols) * sps, dtype=complex if np.iscomplexobj(symbols) else float)
    up[::sps] = symbols
    return up


def pulse_shape(symbols: np.ndarray, h: np.ndarray, sps: int) -> np.ndarray:
    """冲激激励 + 脉冲滤波，得到发送波形。"""
    up = impulse_train(symbols, sps)
    if np.iscomplexobj(up):
        return np.convolve(up, h)          # 实脉冲与复信号卷积
    return np.convolve(up, h)
