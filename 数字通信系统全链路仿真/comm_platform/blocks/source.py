# -*- coding: utf-8 -*-
"""
信源模块：二进制比特流生成
整合自原 channel_coding_comparison.py 的固定种子信源，并扩展手动 / 全 0 / 全 1 模式。
"""

import numpy as np

from ..config import SourceConfig


def generate_bits(cfg: SourceConfig) -> np.ndarray:
    """根据配置返回 0/1 比特流，类型 uint8。"""
    if cfg.mode == "random":
        rng = np.random.default_rng(cfg.seed)
        return rng.integers(0, 2, size=cfg.n_bits).astype(np.uint8)

    if cfg.mode == "manual":
        text = cfg.manual_bits.strip()
        return np.array([int(c) for c in text], dtype=np.uint8)

    if cfg.mode == "zeros":
        return np.zeros(cfg.n_bits, dtype=np.uint8)

    if cfg.mode == "ones":
        return np.ones(cfg.n_bits, dtype=np.uint8)

    raise ValueError(f"未知信源模式：{cfg.mode}")
