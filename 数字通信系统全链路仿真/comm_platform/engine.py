# -*- coding: utf-8 -*-
"""
仿真引擎
========
全链路波形级编排（真实过采样波形 + 匹配滤波 + 同步），并提供符号级
BER 扫描（与波形级数学等效，速度更快）。

链路顺序（BLOCKS）：
    source   比特生成
    coding   信道编码
    modulation  调制 + 脉冲成形
    channel  AWGN 信道
    filter   匹配滤波
    timing   采样判决（理想 / 早迟门）
    decoding 信道解码
    metrics  误码统计

特性：
    * 参数修改后从受影响模块起下游全部置脏（dirty），流程图实时提示；
    * 单步运行自动补全上游依赖；
    * 所有随机数种子确定，配置可导出 JSON，结果可复现；
    * 结果可保存（配置快照、NPZ 数据、BER 的 CSV、运行日志）。
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import time

import numpy as np

from .config import SimConfig, default_results_dir, save_config
from .blocks.channel_codes import get_codec
from .blocks.metrics import bit_error_rate, count_errors
from .blocks.modulation import (add_awgn_waveform, demodulate_hard, map_bits,
                                noise_psd_n0, theory_ber)
from .blocks.source import generate_bits
from .blocks.timing import early_late_track, ideal_samples
from .blocks.waveforms import design_pulse, pulse_shape

BLOCKS = ["source", "coding", "modulation", "channel",
          "filter", "timing", "decoding", "metrics"]
_BLOCK_INDEX = {k: i for i, k in enumerate(BLOCKS)}


class Engine:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.data: dict = {}
        self.status: dict = {k: "idle" for k in BLOCKS}
        self.scan_result: dict | None = None
        self.logs: list[str] = []
        self.last_run_time: str | None = None

    # ----------------------------- 日志 ----------------------------- #
    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.logs.append(line)
        return line

    # ----------------------------- 脏标记 ----------------------------- #
    def invalidate(self, from_block: str):
        """从 from_block 起下游全部置脏，并清空对应数据。"""
        idx = _BLOCK_INDEX[from_block]
        for k in BLOCKS[idx:]:
            self.status[k] = "dirty"
        # 数据在重跑时覆盖；扫描结果与任何参数都相关，参数变化即失效
        self.scan_result = None

    def first_runnable(self) -> str | None:
        """返回第一个未完成（非 done）的模块；全部完成返回 None。"""
        for k in BLOCKS:
            if self.status[k] != "done":
                return k
        return None

    # ----------------------------- 全链路 / 单步 ----------------------------- #
    def run_all(self, progress_cb=None) -> dict:
        self.run_until("metrics", progress_cb)
        return self.data

    def run_until(self, target_block: str, progress_cb=None):
        target = _BLOCK_INDEX[target_block]
        for i, key in enumerate(BLOCKS):
            if i > target:
                break
            if self.status[key] == "done":
                continue
            getattr(self, f"_step_{key}")()
            self.status[key] = "done"
            if progress_cb:
                progress_cb(key, i + 1, target + 1)
        self.last_run_time = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"链路运行至 {target_block} 完成")

    # ----------------------------- 各步骤 ----------------------------- #
    def _step_source(self):
        cfg = self.cfg
        bits = generate_bits(cfg.source)
        self.data["info_bits"] = bits
        self.log(f"信源生成 {len(bits)} bit（模式 {cfg.source.mode}，种子 {cfg.source.seed}）")

    def _step_coding(self):
        cfg = self.cfg
        bits = self.data["info_bits"]
        codec = get_codec(cfg.coding.scheme, cfg.coding.conv_frame)
        pad = (codec.group - len(bits) % codec.group) % codec.group
        bits_padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        coded = codec.encode(bits_padded)

        coded_pad = 0
        if cfg.modulation.mod_type == "QPSK" and len(coded) % 2 == 1:
            coded = np.concatenate([coded, np.zeros(1, dtype=np.uint8)])
            coded_pad = 1

        self.data.update(codec=codec, info_pad=pad,
                         bits_padded=bits_padded, coded_bits=coded,
                         coded_pad=coded_pad)
        self.log(f"信道编码 {codec.name}，码率 R={codec.rate:.4f}，"
                 f"补零 {pad} bit，编码后 {len(coded)} bit")

    def _step_modulation(self):
        cfg = self.cfg
        coded = self.data["coded_bits"]
        mcfg = cfg.modulation
        t_pulse, h = design_pulse(mcfg.pulse, mcfg.beta, mcfg.sps, mcfg.span)
        symbols = map_bits(coded, mcfg.mod_type)
        tx = pulse_shape(symbols, h, mcfg.sps)
        self.data.update(t_pulse=t_pulse, pulse=h, tx_symbols=symbols, tx_wave=tx)
        self.log(f"{mcfg.mod_type} 调制 + {mcfg.pulse} 成形，"
                 f"发送符号 {len(symbols)} 个，波形 {len(tx)} 点")

    def _step_channel(self):
        cfg = self.cfg
        tx = self.data["tx_wave"]
        if cfg.channel.add_noise:
            seed = cfg.channel.noise_seed
            if seed < 0:
                seed = cfg.source.seed + 7
            rng = np.random.default_rng(seed)
            rx = add_awgn_waveform(tx, cfg.modulation.mod_type,
                                   self.data["codec"].rate,
                                   cfg.channel.ebno_db, rng)
            self.data["noise_rng_seed"] = seed
            self.log(f"AWGN 信道，Eb/N0={cfg.channel.ebno_db} dB，噪声种子 {seed}")
        else:
            rx = tx.copy()
            self.data["noise_rng_seed"] = None
            self.log("理想无噪信道")
        self.data["rx_wave"] = rx

    def _step_filter(self):
        cfg = self.cfg
        h = self.data["pulse"]
        rx = self.data["rx_wave"]
        mf = np.convolve(rx, h)
        sps = cfg.modulation.sps
        t_axis = (np.arange(len(mf)) - cfg.modulation.span * sps) / sps
        self.data.update(mf_wave=mf, t_axis=t_axis)
        self.log(f"匹配滤波完成，输出 {len(mf)} 点")

    def _step_timing(self):
        cfg = self.cfg
        n_sym = len(self.data["tx_symbols"])
        sps, span = cfg.modulation.sps, cfg.modulation.span
        mf, t_axis = self.data["mf_wave"], self.data["t_axis"]

        if cfg.timing.mode == "ideal":
            samples = ideal_samples(mf, n_sym, sps, span)
            self.data.update(samples=samples, tau_hist=None, err_hist=None,
                             a_hist=None)
            self.log("理想最佳采样")
        else:
            tcfg = cfg.timing
            tau_hist, err_hist, yP, a_hist = early_late_track(
                mf, t_axis, n_sym, cfg.modulation.mod_type,
                tcfg.tau0, tcfg.delta, tcfg.kp, tcfg.ki)
            self.data.update(samples=yP, tau_hist=tau_hist,
                             err_hist=err_hist, a_hist=a_hist)
            jitter = float(np.sqrt(np.mean(tau_hist[len(tau_hist) // 2:] ** 2)))
            self.data["jitter_rms"] = jitter
            self.log(f"早迟门跟踪，初始偏差 {tcfg.tau0} T，稳态抖动 RMS {jitter:.4f} T")

    def _step_decoding(self):
        cfg = self.cfg
        codec = self.data["codec"]
        coded = self.data["coded_bits"]
        samples = self.data["samples"]

        hard = demodulate_hard(samples, cfg.modulation.mod_type)
        hard = hard[:len(coded) - self.data["coded_pad"]]
        decoded_padded = codec.decode(hard)
        decoded = decoded_padded[:len(self.data["info_bits"])]

        self.data.update(hard_bits=hard, decoded_bits=decoded)
        self.log(f"信道解码完成，输出 {len(decoded)} bit")

    def _step_metrics(self):
        bits = self.data["info_bits"]
        decoded = self.data["decoded_bits"]
        coded = self.data["coded_bits"]
        hard = self.data["hard_bits"]

        n_err = count_errors(bits, decoded)
        info_ber = bit_error_rate(bits, decoded)
        pre_ber = bit_error_rate(coded[:len(hard)], hard)
        theory = float(np.atleast_1d(theory_ber(
            self.cfg.modulation.mod_type, [self.cfg.channel.ebno_db]))[0])

        self.data.update(n_errors=n_err, info_ber=info_ber,
                         pre_decode_ber=pre_ber, theory_ber_point=theory)
        self.log(f"误码统计：错误 {n_err} bit，信息 BER={info_ber:.3e}，"
                 f"未编码理论参考 {theory:.3e}")

    # ----------------------------- BER 扫描 ----------------------------- #
    def run_ber_scan(self, progress_cb=None, should_stop=None):
        """
        符号级快速扫描（数学等效于波形级采样点分布）。
        产出 self.scan_result：
            snr_grid, per scheme BER 列表, theory, n_sent
        """
        scfg = self.cfg.scan
        mcfg = self.cfg.modulation
        grid = np.round(np.arange(
            scfg.ebno_start, scfg.ebno_stop + scfg.ebno_step / 2.0,
            scfg.ebno_step), 4)

        bits = generate_bits(self.cfg.source)
        per_scheme = {}
        total_jobs = len(scfg.schemes) * len(grid)
        done_jobs = 0

        for s_idx, scheme in enumerate(scfg.schemes):
            codec = get_codec(scheme, self.cfg.coding.conv_frame)
            pad = (codec.group - len(bits) % codec.group) % codec.group
            bits_padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
            coded = codec.encode(bits_padded)
            if mcfg.mod_type == "QPSK" and len(coded) % 2 == 1:
                coded = np.concatenate([coded, np.zeros(1, dtype=np.uint8)])

            bers, sent_list = [], []
            for p_idx, ebno in enumerate(grid):
                rng = np.random.default_rng([9001, s_idx, p_idx])
                n0 = noise_psd_n0(mcfg.mod_type, codec.rate, float(ebno))
                sigma = np.sqrt(n0 / 2.0)

                if not scfg.use_target_errors:
                    symbols = map_bits(coded, mcfg.mod_type)
                    if mcfg.mod_type == "BPSK":
                        y = symbols + sigma * rng.standard_normal(symbols.shape)
                    else:
                        y = symbols + sigma * (
                            rng.standard_normal(symbols.shape)
                            + 1j * rng.standard_normal(symbols.shape))
                    hard = demodulate_hard(y, mcfg.mod_type)
                    dec = codec.decode(hard)[:len(bits)]
                    bers.append(bit_error_rate(bits, dec))
                    sent_list.append(len(symbols))
                else:
                    # 目标错误数自适应：按整份编码分批，累计错误或触及符号上限
                    total_err, total_bits, n_sent = 0, 0, 0
                    while n_sent < scfg.max_symbols:
                        symbols = map_bits(coded, mcfg.mod_type)
                        if mcfg.mod_type == "BPSK":
                            y = symbols + sigma * rng.standard_normal(symbols.shape)
                        else:
                            y = symbols + sigma * (
                                rng.standard_normal(symbols.shape)
                                + 1j * rng.standard_normal(symbols.shape))
                        hard = demodulate_hard(y, mcfg.mod_type)
                        dec = codec.decode(hard)[:len(bits)]
                        total_err += count_errors(bits, dec)
                        total_bits += len(bits)
                        n_sent += len(symbols)
                        if total_err >= scfg.target_errors:
                            break
                    bers.append(total_err / total_bits if total_bits else 0.0)
                    sent_list.append(n_sent)

                done_jobs += 1
                if progress_cb:
                    progress_cb(scheme, float(ebno), done_jobs, total_jobs)
                if should_stop and should_stop():
                    self.log("BER 扫描被用户中止")
                    return self.scan_result

            per_scheme[scheme] = {"ber": bers, "n_sent": sent_list,
                                  "rate": codec.rate, "name": codec.name}

        self.scan_result = {
            "snr_grid": grid.tolist(),
            "schemes": per_scheme,
            "theory": np.atleast_1d(theory_ber(mcfg.mod_type, grid)).tolist(),
            "mod_type": mcfg.mod_type,
        }
        self.log("BER 扫描完成")
        return self.scan_result

    # ----------------------------- 结果保存 ----------------------------- #
    def results_dir(self) -> str:
        out = self.cfg.output.out_dir or default_results_dir()
        os.makedirs(out, exist_ok=True)
        return out

    def save_results(self) -> dict:
        out = self.results_dir()
        stamp = time.strftime("%Y%m%d_%H%M%S")

        cfg_path = os.path.join(out, f"config_snapshot_{stamp}.json")
        save_config(self.cfg, cfg_path)

        log_path = os.path.join(out, f"run_log_{stamp}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.logs))

        saved = {"config": cfg_path, "log": log_path}

        if self.data:
            npz_path = os.path.join(out, f"chain_data_{stamp}.npz")
            pack = {k: v for k, v in self.data.items()
                    if isinstance(v, np.ndarray)}
            np.savez_compressed(npz_path, **pack)
            saved["npz"] = npz_path

        if self.scan_result:
            csv_path = os.path.join(out, f"ber_scan_{stamp}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                grid = self.scan_result["snr_grid"]
                w.writerow(["Eb/N0(dB)"]
                           + [d["name"] for d in self.scan_result["schemes"].values()]
                           + [f"理论{self.scan_result['mod_type']}"])
                for i, snr in enumerate(grid):
                    row = [snr]
                    for d in self.scan_result["schemes"].values():
                        row.append(f"{d['ber'][i]:.3e}")
                    row.append(f"{self.scan_result['theory'][i]:.3e}")
                    w.writerow(row)
            saved["ber_csv"] = csv_path

        self.log(f"结果已保存到 {out}")
        return saved
