# -*- coding: utf-8 -*-
"""
位同步与采样判决
================
整合自 early_late_gate_sim.py：
    判决引导线性早迟门鉴相 + 二阶比例积分环路滤波器 + NCO 闭环跟踪。

实信号（BPSK）：a_hat = sign(yP)，e = a_hat·(yL - yE)
复信号（QPSK）：a_hat = 最近邻星座点，e = Re[ conj(a_hat)·(yL - yE) ]
BPSK 公式是复信号公式在实星座下的特例，两者统一。
"""

import numpy as np
from scipy.interpolate import CubicSpline

from .modulation import BPSK_CONSTELLATION, QPSK_CONSTELLATION, nearest_neighbor


def ideal_samples(mf: np.ndarray, n_sym: int, sps: int,
                  delay_sym: float) -> np.ndarray:
    """理想最佳采样：在总链路延迟处按符号周期取点。"""
    idx = (np.arange(n_sym) + delay_sym) * sps
    idx = np.round(idx).astype(int)
    return mf[idx]


def _decide_symbol(yP, mod_type: str):
    """返回本地判决符号 a_hat（与发送符号同类型）。"""
    if mod_type == "BPSK":
        return 1.0 if yP >= 0 else -1.0
    idx = nearest_neighbor(np.array([yP]), QPSK_CONSTELLATION)[0]
    return QPSK_CONSTELLATION[idx]


def early_late_track(mf: np.ndarray, t_axis: np.ndarray, n_sym: int,
                     mod_type: str, tau0: float = 0.35, delta: float = 0.5,
                     kp: float = 0.06, ki: float = 0.006,
                     nco_guard: float = 0.5):
    """
    闭环跟踪，返回：
        tau_hist  每符号定时偏差
        err_hist  鉴相误差
        yP_hist   准时门样本
        a_hist    判决符号
    """
    cs = CubicSpline(t_axis, mf)
    tau = float(tau0)
    integ = 0.0
    tau_hist, err_hist, yP_hist, a_hist = [], [], [], []

    for n in range(n_sym):
        tn = n + tau
        yE = cs(tn - delta)
        yL = cs(tn + delta)
        yP = cs(tn)

        a_hat = _decide_symbol(yP, mod_type)
        diff = yL - yE
        if mod_type == "BPSK":
            e = float(a_hat * diff)
        else:
            e = float(np.real(np.conj(a_hat) * diff))

        integ += ki * e
        w = kp * e + integ

        tau_hist.append(tau)
        err_hist.append(e)
        yP_hist.append(yP)
        a_hist.append(a_hat)

        tau = float(np.clip(tau + w, -nco_guard, nco_guard))

    return (np.array(tau_hist), np.array(err_hist),
            np.array(yP_hist), np.array(a_hist, dtype=complex if mod_type == "QPSK" else float))


def open_loop_samples(mf: np.ndarray, t_axis: np.ndarray, n_sym: int,
                      tau0: float) -> np.ndarray:
    """开环对照：始终带固定偏差采样。"""
    cs = CubicSpline(t_axis, mf)
    return np.array([cs(n + tau0) for n in range(n_sym)])


def s_curve_scan(mf: np.ndarray, t_axis: np.ndarray, tx_symbols: np.ndarray,
                 mod_type: str, delta: float = 0.5,
                 taus=None) -> tuple:
    """扫描固定定时偏差，测量鉴相 S 曲线（数据辅助）。"""
    if taus is None:
        taus = np.linspace(-0.5, 0.5, 41)
    cs = CubicSpline(t_axis, mf)
    n_sym = len(tx_symbols)
    scan_n = np.arange(min(50, n_sym // 5), max(n_sym - 50, n_sym // 5 + 1))
    curve = []
    for tv in taus:
        vals = []
        for n in scan_n:
            diff = cs(n + tv + delta) - cs(n + tv - delta)
            if mod_type == "BPSK":
                vals.append(tx_symbols[n] * diff)
            else:
                vals.append(np.real(np.conj(tx_symbols[n]) * diff))
        curve.append(np.mean(vals))
    return np.array(taus), np.array(curve)
