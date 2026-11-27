# -*- coding: utf-8 -*-
"""
信道编码模块
============
整合自原 channel_coding_comparison.py，统一接口：
    无编码      none       R=1
    重复码      rep31      R=1/3，多数判决
    汉明码      ham74      R=4/7，纠 1 位错误
    卷积码      conv213    R=1/2，生成多项式 [7,5]，维特比硬判决译码
    BCH(15,7)   bch157     R=7/15，纠 2 位错误，GF(2^4)

每个编码方案用 CodecSpec 描述：码率、信息分组长度、encode / decode。
引擎负责把任意长度的信息比特补零到分组长度的整数倍，解码后再去掉补零。
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ----------------------------- 重复码 (3,1) ----------------------------- #
def rep31_encode(bits: np.ndarray) -> np.ndarray:
    return np.repeat(bits, 3)


def rep31_decode(hard_bits: np.ndarray) -> np.ndarray:
    blocks = hard_bits.reshape(-1, 3)
    return (blocks.sum(axis=1) >= 2).astype(np.uint8)


# ----------------------------- 汉明码 (7,4) ----------------------------- #
HAM_G = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 1, 1],
], dtype=np.uint8)

HAM_H = np.array([
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1],
], dtype=np.uint8)

HAM_SYN_TO_BIT = {}
for j in range(7):
    syn_val = int(HAM_H[0, j]) * 4 + int(HAM_H[1, j]) * 2 + int(HAM_H[2, j])
    HAM_SYN_TO_BIT[syn_val] = j


def ham74_encode(bits: np.ndarray) -> np.ndarray:
    msg = bits.reshape(-1, 4)
    return ((msg @ HAM_G) % 2).reshape(-1)


def ham74_decode(hard_bits: np.ndarray) -> np.ndarray:
    blocks = hard_bits.reshape(-1, 7).copy()
    syndrome = (blocks @ HAM_H.T) % 2
    syn_val = syndrome[:, 0] * 4 + syndrome[:, 1] * 2 + syndrome[:, 2]
    for i, sv in enumerate(syn_val):
        if sv != 0:
            pos = HAM_SYN_TO_BIT.get(int(sv))
            if pos is not None:
                blocks[i, pos] ^= 1
    return blocks[:, :4].reshape(-1)


# ----------------------------- 卷积码 (2,1,3) ----------------------------- #
def conv213_encode_frame(msg: np.ndarray) -> np.ndarray:
    """一帧信息比特编码，末尾补 K-1=2 个零尾使寄存器归零。"""
    reg0, reg1 = 0, 0
    out = []
    for u in list(msg) + [0, 0]:
        v1 = u ^ reg0 ^ reg1          # 生成多项式 111 (7)
        v2 = u ^ reg1                 # 生成多项式 101 (5)
        out.extend([v1, v2])
        reg1, reg0 = reg0, u
    return np.array(out, dtype=np.uint8)


def viterbi_decode_frame(rx: np.ndarray, n_msg: int) -> np.ndarray:
    """维特比硬判决译码，状态 state = 2*reg0 + reg1，共 4 个状态。"""
    n_steps = n_msg + 2
    inf = 10 ** 9
    path_metric = np.full(4, inf)
    path_metric[0] = 0
    history = []

    for t in range(n_steps):
        y0, y1 = rx[2 * t], rx[2 * t + 1]
        new_metric = np.full(4, inf)
        predecessor = np.zeros(4, dtype=np.uint8)

        for s in range(4):
            if path_metric[s] == inf:
                continue
            reg0 = (s >> 1) & 1
            reg1 = s & 1
            for u in (0, 1):
                v1 = u ^ reg0 ^ reg1
                v2 = u ^ reg1
                ns = 2 * u + reg0
                branch_cost = (y0 ^ v1) + (y1 ^ v2)     # 汉明距离
                candidate = path_metric[s] + branch_cost
                if candidate < new_metric[ns]:          # ACS
                    new_metric[ns] = candidate
                    predecessor[ns] = s

        history.append(predecessor)
        path_metric = new_metric

    decoded = []
    s = 0
    for t in range(n_steps - 1, -1, -1):
        u = (s >> 1) & 1              # 结束状态高位即当时输入
        decoded.append(u)
        s = history[t][s]
    decoded.reverse()
    return np.array(decoded[:n_msg], dtype=np.uint8)


def make_conv213(conv_frame: int):
    """构造绑定帧长的卷积码 encode/decode。"""
    def encode(bits: np.ndarray) -> np.ndarray:
        frames = bits.reshape(-1, conv_frame)
        return np.concatenate([conv213_encode_frame(f) for f in frames])

    def decode(hard_bits: np.ndarray) -> np.ndarray:
        frame_len = 2 * (conv_frame + 2)
        frames = hard_bits.reshape(-1, frame_len)
        return np.concatenate([viterbi_decode_frame(f, conv_frame) for f in frames])

    return encode, decode


# ----------------------------- BCH (15,7) ----------------------------- #
PRIM_POLY = 0b10011


def build_gf16():
    exp_table = [0] * 15
    log_table = [0] * 16
    x = 1
    for i in range(15):
        exp_table[i] = x
        log_table[x] = i
        x <<= 1
        if x & 0b10000:
            x ^= PRIM_POLY
    return exp_table, log_table


GF_EXP, GF_LOG = build_gf16()


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return GF_EXP[(GF_LOG[a] + GF_LOG[b]) % 15]


def gf_inv(a):
    return GF_EXP[(-GF_LOG[a]) % 15]


def gf_pow3(a):
    return GF_EXP[(3 * GF_LOG[a]) % 15]


# g(x) = x^8 + x^7 + x^6 + x^4 + 1，数组下标即 x 的幂次
BCH_GEN = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)


def poly_remainder(shifted):
    reg = list(shifted)
    g_deg = 8
    for i in range(len(reg) - 1, g_deg - 1, -1):
        if reg[i]:
            for j in range(g_deg + 1):
                reg[i - g_deg + j] ^= BCH_GEN[j]
    return reg[:g_deg]


def bch157_encode(bits: np.ndarray) -> np.ndarray:
    msg = bits.reshape(-1, 7)
    codewords = np.zeros((msg.shape[0], 15), dtype=np.uint8)
    for i, m in enumerate(msg):
        shifted = np.zeros(15, dtype=np.uint8)
        shifted[8:15] = m
        shifted[:8] = poly_remainder(shifted)
        codewords[i] = shifted
    return codewords.reshape(-1)


def bch157_decode(hard_bits: np.ndarray) -> np.ndarray:
    blocks = hard_bits.reshape(-1, 15).copy()
    decoded = np.zeros((blocks.shape[0], 7), dtype=np.uint8)

    for k in range(blocks.shape[0]):
        cw = blocks[k]
        S1, S3 = 0, 0
        for i in range(15):
            if cw[i]:
                S1 ^= GF_EXP[i]
                S3 ^= GF_EXP[(3 * i) % 15]

        if S1 != 0:
            if gf_pow3(S1) == S3:
                cw[GF_LOG[S1]] ^= 1                  # 单错
            else:
                product = gf_mul(gf_pow3(S1) ^ S3, gf_inv(S1))
                roots = []
                for z in range(1, 16):               # 穷举求根
                    if gf_mul(z, z) ^ gf_mul(S1, z) ^ product == 0:
                        roots.append(z)
                if len(roots) == 2:                  # 双错
                    for X in roots:
                        cw[GF_LOG[X]] ^= 1
        decoded[k] = cw[8:15]
    return decoded.reshape(-1)


# ----------------------------- 统一注册表 ----------------------------- #
@dataclass
class CodecSpec:
    key: str
    name: str
    rate: float
    group: int                      # 信息分组长度（输入需补零到其倍数）
    encode: Callable
    decode: Callable
    coded_block: int                # 编码后每组的比特数


def _identity(x):
    return x


def get_codec(key: str, conv_frame: int = 500) -> CodecSpec:
    if key == "none":
        return CodecSpec("none", "无编码", 1.0, 1, _identity, _identity, 1)
    if key == "rep31":
        return CodecSpec("rep31", "重复码(3,1)", 1.0 / 3.0, 3,
                         rep31_encode, rep31_decode, 3)
    if key == "ham74":
        return CodecSpec("ham74", "汉明码(7,4)", 4.0 / 7.0, 4,
                         ham74_encode, ham74_decode, 7)
    if key == "conv213":
        enc, dec = make_conv213(conv_frame)
        return CodecSpec("conv213", "卷积码(2,1,3)", 1.0 / 2.0, conv_frame,
                         enc, dec, 2 * (conv_frame + 2))
    if key == "bch157":
        return CodecSpec("bch157", "BCH(15,7)", 7.0 / 15.0, 7,
                         bch157_encode, bch157_decode, 15)
    raise ValueError(f"未知信道编码方案：{key}")


CODEC_KEYS = ["none", "rep31", "ham74", "conv213", "bch157"]
