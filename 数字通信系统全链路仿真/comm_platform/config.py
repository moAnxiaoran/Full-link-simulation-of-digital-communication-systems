# -*- coding: utf-8 -*-
"""
仿真配置系统
============
全平台所有可调参数集中在本文件的 dataclass 中定义：
    SimConfig
      ├─ SourceConfig      信源比特生成
      ├─ CodingConfig      信道编码
      ├─ ModulationConfig  调制与脉冲成形
      ├─ ChannelConfig     AWGN 信道
      ├─ TimingConfig      采样判决 / 位同步
      ├─ ScanConfig        BER 扫描
      └─ OutputConfig      结果输出

配置可整体序列化为 JSON（导出配置 / 导入配置），
同一份 JSON 在任何机器上运行都得到完全一致的结果，实现“可复现”。
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field


# ----------------------------- 各模块参数 ----------------------------- #
@dataclass
class SourceConfig:
    mode: str = "random"            # random=随机 / manual=手动输入 / zeros / ones
    n_bits: int = 105000            # 随机模式下的信息比特数
    seed: int = 42                  # 信源随机种子
    manual_bits: str = "101010110011"  # manual 模式下的比特字符串（仅允许 0/1）


@dataclass
class CodingConfig:
    # none=无编码 / rep31=重复码(3,1) / ham74=汉明(7,4)
    # conv213=卷积(2,1,3) / bch157=BCH(15,7)
    scheme: str = "none"
    conv_frame: int = 500           # 卷积码每帧信息比特数


@dataclass
class ModulationConfig:
    mod_type: str = "BPSK"          # BPSK / QPSK
    pulse: str = "RRC"              # RRC=根升余弦 / RC=升余弦 / rect=矩形
    sps: int = 16                   # 每符号采样点数（过采样率）
    beta: float = 0.5               # 滚降系数 α/β，取值 (0, 1]
    span: int = 8                   # 滤波器截断长度（符号数，取偶数）


@dataclass
class ChannelConfig:
    add_noise: bool = True          # 是否加 AWGN（False=理想无噪信道）
    ebno_db: float = 8.0            # 单点运行的 Eb/N0，单位 dB
    noise_seed: int = 2026          # 噪声种子；-1 表示由信源种子自动派生


@dataclass
class TimingConfig:
    mode: str = "ideal"             # ideal=理想最佳采样 / early_late=早迟门跟踪
    tau0: float = 0.35              # 初始定时偏差（单位 T）
    delta: float = 0.5              # 早/迟门偏移（单位 T）
    kp: float = 0.06                # 环路滤波器比例系数
    ki: float = 0.006               # 环路滤波器积分系数


@dataclass
class ScanConfig:
    ebno_start: float = 0.0         # BER 扫描起点 dB
    ebno_stop: float = 10.0         # BER 扫描终点 dB（含）
    ebno_step: float = 2.0          # 扫描步长 dB
    # 参与对比的编码方案
    schemes: list = field(default_factory=lambda: [
        "none", "rep31", "ham74", "conv213", "bch157"])
    use_target_errors: bool = False  # True=按目标错误数自适应；False=固定比特数
    target_errors: int = 300        # 每个 SNR 点至少累计的错误比特数
    max_symbols: int = 1_000_000    # 自适应模式每点最大发送符号数


@dataclass
class OutputConfig:
    # 留空时由 main 自动指向“通信原理作业/platform_results”
    out_dir: str = ""


@dataclass
class SimConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    coding: CodingConfig = field(default_factory=CodingConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ----------------------------- 序列化 ----------------------------- #
def config_to_dict(cfg: SimConfig) -> dict:
    return asdict(cfg)


def config_from_dict(d: dict) -> SimConfig:
    """从普通字典构造配置，缺失字段自动取默认值，多余字段忽略。"""
    sub = {
        "source": SourceConfig, "coding": CodingConfig,
        "modulation": ModulationConfig, "channel": ChannelConfig,
        "timing": TimingConfig, "scan": ScanConfig,
        "output": OutputConfig,
    }
    kwargs = {}
    for key, cls in sub.items():
        data = d.get(key, {}) or {}
        valid = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        kwargs[key] = cls(**valid)
    return SimConfig(**kwargs)


def save_config(cfg: SimConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, ensure_ascii=False, indent=2)


def load_config(path: str) -> SimConfig:
    with open(path, "r", encoding="utf-8") as f:
        return config_from_dict(json.load(f))


def clone_config(cfg: SimConfig) -> SimConfig:
    return copy.deepcopy(cfg)


# ----------------------------- 参数校验 ----------------------------- #
_CHOICE = {
    "source.mode": {"random", "manual", "zeros", "ones"},
    "coding.scheme": {"none", "rep31", "ham74", "conv213", "bch157"},
    "modulation.mod_type": {"BPSK", "QPSK"},
    "modulation.pulse": {"RRC", "RC", "rect"},
    "timing.mode": {"ideal", "early_late"},
}


def validate_config(cfg: SimConfig) -> list:
    """返回错误信息列表；空列表表示参数合法。"""
    errs: list[str] = []

    def check(cond, msg):
        if not cond:
            errs.append(msg)

    for path, choices in _CHOICE.items():
        obj, attr = path.split(".")
        check(getattr(getattr(cfg, obj), attr) in choices,
              f"{path} 取值非法，可选：{sorted(choices)}")

    check(cfg.source.n_bits >= 1, "信源比特数 n_bits 必须 >= 1")
    check(cfg.source.seed >= 0, "信源种子必须为非负整数")
    if cfg.source.mode == "manual":
        s = cfg.source.manual_bits.strip()
        check(len(s) >= 1 and set(s) <= {"0", "1"},
              "手动比特只能包含字符 0 和 1，且不能为空")

    check(cfg.coding.conv_frame >= 1, "卷积码帧长必须 >= 1")

    check(cfg.modulation.sps >= 2, "每符号采样点数 sps 必须 >= 2")
    check(0.0 < cfg.modulation.beta <= 1.0, "滚降系数必须在 (0, 1] 内")
    check(cfg.modulation.span >= 2 and cfg.modulation.span % 2 == 0,
          "滤波器跨度 span 必须为 >= 2 的偶数")

    check(cfg.channel.noise_seed >= -1, "噪声种子必须 >= -1")

    check(0.0 < cfg.timing.delta <= 0.5, "早迟门偏移 delta 必须在 (0, 0.5] 内")
    check(abs(cfg.timing.tau0) <= 0.5, "初始定时偏差 |tau0| 必须 <= 0.5")
    check(cfg.timing.kp > 0 and cfg.timing.ki > 0, "环路增益 kp/ki 必须为正数")

    check(cfg.scan.ebno_stop >= cfg.scan.ebno_start,
          "BER 扫描终点不能小于起点")
    check(cfg.scan.ebno_step > 0, "BER 扫描步长必须为正数")
    check(len(cfg.scan.schemes) >= 1, "BER 扫描至少选择一个编码方案")
    check(cfg.scan.target_errors >= 1, "目标错误数必须 >= 1")
    check(cfg.scan.max_symbols >= 1, "最大符号数必须 >= 1")

    if cfg.output.out_dir:
        parent = os.path.dirname(os.path.abspath(cfg.output.out_dir))
        check(os.path.isdir(parent), f"输出目录的父路径不存在：{parent}")

    return errs


def default_results_dir() -> str:
    """平台默认结果目录：通信原理作业/platform_results。"""
    here = os.path.dirname(os.path.abspath(__file__))
    homework_dir = os.path.dirname(here)
    return os.path.join(homework_dir, "platform_results")
