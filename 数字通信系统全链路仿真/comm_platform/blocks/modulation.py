# -*- coding: utf-8 -*-
"""
调制与解调
==========
整合自 qpsk_awgn_demo.py 与 channel_coding_comparison.py：
    BPSK：0 -> +1，1 -> -1（实符号，Es=1）
    QPSK：格雷映射 00/01/11/10（复符号，Es=1）

噪声口径（发送符号能量归一化为 1，编码码率 R）：
    BPSK 每符号 1 个编码比特：Eb = Es / R，     N0 = 1 / (R·EbN0)
    QPSK 每符号 2 个编码比特：Eb = Es / (2R)，   N0 = 1 / (2R·EbN0)
过采样波形上每个实数维度加 N(0, N0/2) 的噪声；
单位能量匹配滤波器不改变采样点噪声方差。
"""

import math

import numpy as np


# ----------------------------- 星座 ----------------------------- #
QPSK_CONSTELLATION = np.array([
    1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j,
]) / np.sqrt(2)
QPSK_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

BPSK_CONSTELLATION = np.array([1.0, -1.0])


def _pair_to_index(b0, b1):
    """两位格雷比特 -> QPSK 星座下标。"""
    for i, pair in enumerate(QPSK_BITS):
        if pair[0] == b0 and pair[1] == b1:
            return i
    raise ValueError("非法比特对")


def map_bits(bits: np.ndarray, mod_type: str) -> np.ndarray:
    """0/1 比特 -> 归一化发送符号。QPSK 时比特数需为偶数（引擎补零保证）。"""
    if mod_type == "BPSK":
        return 1.0 - 2.0 * bits.astype(float)

    if mod_type == "QPSK":
        n = len(bits) // 2
        reshaped = bits.reshape(n, 2)
        idx = np.array([_pair_to_index(p[0], p[1]) for p in reshaped])
        return QPSK_CONSTELLATION[idx]

    raise ValueError(f"未知调制方式：{mod_type}")


# ----------------------------- 噪声 ----------------------------- #
def noise_psd_n0(mod_type: str, rate: float, ebno_db: float) -> float:
    """按 Eb/N0 返回单边噪声功率谱密度 N0。"""
    ebno = 10.0 ** (ebno_db / 10.0)
    factor = 2.0 if mod_type == "QPSK" else 1.0
    return 1.0 / (factor * rate * ebno)


def add_awgn_waveform(wave: np.ndarray, mod_type: str, rate: float,
                      ebno_db: float, rng) -> np.ndarray:
    """给成形后的波形加 AWGN，返回接收波形。"""
    n0 = noise_psd_n0(mod_type, rate, ebno_db)
    sigma2 = n0 / 2.0
    if mod_type == "QPSK":
        noise = np.sqrt(sigma2) * (rng.standard_normal(wave.shape)
                                   + 1j * rng.standard_normal(wave.shape))
    else:
        noise = np.sqrt(sigma2) * rng.standard_normal(wave.shape)
    return wave + noise


# ----------------------------- 判决 ----------------------------- #
def nearest_neighbor(rx_symbols: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """最近邻判决，返回星座下标。"""
    dist = (np.abs(rx_symbols[:, None]) ** 2
            + np.abs(constellation[None, :]) ** 2
            - 2.0 * np.real(rx_symbols[:, None] * np.conj(constellation[None, :])))
    return np.argmin(dist, axis=1)


def demodulate_hard(rx_symbols: np.ndarray, mod_type: str) -> np.ndarray:
    """接收符号 -> 硬判决 0/1 比特。"""
    if mod_type == "BPSK":
        return (rx_symbols < 0.0).astype(np.uint8)

    if mod_type == "QPSK":
        idx = nearest_neighbor(rx_symbols, QPSK_CONSTELLATION)
        return QPSK_BITS[idx].reshape(-1)

    raise ValueError(f"未知调制方式：{mod_type}")


# ----------------------------- 理论 BER ----------------------------- #
def q_function(x):
    """高斯 Q 函数 Q(x)=0.5·erfc(x/sqrt(2))。"""
    x = np.asarray(x, dtype=float)
    return 0.5 * np.array([math.erfc(v / math.sqrt(2)) for v in x.flat]).reshape(x.shape)


def theory_ber(mod_type: str, ebno_db):
    """未编码理论 BER：BPSK 与格雷映射 QPSK 均为 Q(sqrt(2 Eb/N0))。"""
    ebno = 10.0 ** (np.asarray(ebno_db, dtype=float) / 10.0)
    return q_function(np.sqrt(2.0 * ebno))
