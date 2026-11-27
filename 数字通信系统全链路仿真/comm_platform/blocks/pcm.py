# -*- coding: utf-8 -*-
"""
PCM 脉冲编码调制
================
整合自 pcm_simulation.py：
    模拟信号 -> 抽样 -> 均匀量化 / A 律压扩 -> 自然/折叠二进制编码 -> 译码重建
理论验证：满幅正弦均匀量化 SNR(dB) = 6.02N + 1.76。
"""

import numpy as np


def generate_signal(t, f1=50.0, f2=120.0):
    """双音信号：f1 正弦 + 0.5 倍 f2 正弦，归一化到 [-1, 1]。"""
    x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    return x / np.max(np.abs(x))


def uniform_quantize(signal, n_bits, v_ref=1.0):
    """中平均匀量化，返回 (量化信号, 量化级序号, 量化间隔 Δ)。"""
    levels = 2 ** n_bits
    delta = 2.0 * v_ref / levels
    s = np.clip(signal, -v_ref, v_ref - 1e-12)
    index = np.floor((s + v_ref) / delta).astype(int)
    index = np.clip(index, 0, levels - 1)
    quantized = -v_ref + (index + 0.5) * delta
    return quantized, index, delta


def encode_natural(index, n_bits):
    """自然二进制码。"""
    return [format(i, f"0{n_bits}b") for i in index]


def encode_folded(index, n_bits):
    """折叠二进制码：最高位表极性，其余表幅度。"""
    half = 2 ** (n_bits - 1)
    codes = []
    for i in index:
        polarity = 1 if i >= half else 0
        magnitude = i - half if i >= half else half - 1 - i
        codes.append(str(polarity) + format(magnitude, f"0{n_bits - 1}b"))
    return codes


def decode_codes(codes, v_ref=1.0):
    """二进制码 -> 量化电平。"""
    n_bits = len(codes[0])
    delta = 2.0 * v_ref / (2 ** n_bits)
    index = np.array([int(c, 2) for c in codes])
    return -v_ref + (index + 0.5) * delta


def alaw_compress(x, A=87.56):
    """A 律压缩。"""
    sign = np.sign(x)
    ax = np.abs(x)
    ay = np.where(
        ax < 1.0 / A,
        A * ax / (1.0 + np.log(A)),
        (1.0 + np.log(np.maximum(A * ax, 1e-12))) / (1.0 + np.log(A)))
    return sign * ay


def alaw_expand(y, A=87.56):
    """A 律扩张。"""
    sign = np.sign(y)
    ay = np.abs(y)
    threshold = 1.0 / (1.0 + np.log(A))
    ax = np.where(
        ay < threshold,
        ay * (1.0 + np.log(A)) / A,
        np.exp(ay * (1.0 + np.log(A)) - 1.0) / A)
    return sign * ax


def snr_db(original, noise):
    """信噪比，单位 dB。"""
    p_signal = np.mean(np.asarray(original) ** 2)
    p_noise = np.mean(np.asarray(noise) ** 2)
    return 10.0 * np.log10(p_signal / p_noise)
