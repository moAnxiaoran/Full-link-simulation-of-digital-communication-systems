# -*- coding: utf-8 -*-
"""
绘图模块
========
平台所有图表的绘制函数集中于此，统一接收 matplotlib 的 fig/ax 并就地重画，
由 GUI 嵌入显示。分为四组：
    1. 全链路各模块输入输出（信源/编码/调制/信道/滤波/同步/解码）
    2. BER 扫描曲线与判决边界
    3. 基带码型（波形/眼图/PSD）
    4. PCM（抽样量化重建/SNR 理论验证）
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import welch

from .blocks.linecodes import LINE_CODE_FUNCS, hdb3_encode
from .blocks.modulation import (QPSK_BITS, QPSK_CONSTELLATION,
                                nearest_neighbor, theory_ber)
from .blocks import pcm as pcm_block
from .blocks.timing import s_curve_scan


# ============================== 通用工具 ============================== #
def _real(x):
    return np.real(x) if np.iscomplexobj(x) else np.asarray(x, dtype=float)


def waveform_time(n_points: int, sps: int, delay_sym: float) -> np.ndarray:
    """conv 输出波形的时间轴（单位 T），中心对齐到 delay_sym。"""
    return (np.arange(n_points) - delay_sym * sps) / sps


def eye_diagram(ax, sig, sps, title="眼图", max_traces=250):
    """按 2T 分段重叠绘制眼图，去掉两端瞬态。"""
    sig = _real(sig)
    seg = 2 * sps
    start = 2 * sps
    end = max(len(sig) - 2 * sps, start + seg)
    n_traces = min((end - start) // seg, max_traces)
    x = np.linspace(-1, 1, seg, endpoint=False)
    for k in range(n_traces):
        trace = sig[start + k * seg: start + (k + 1) * seg]
        ax.plot(x, trace, color="#1f6feb", alpha=0.35, linewidth=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("t / T")
    ax.set_ylabel("幅度")
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ============================== 1. 信源 ============================== #
def plot_source(ax, bits, show=40):
    n = min(show, len(bits))
    ax.step(np.arange(n), bits[:n], where="post", color="#c0392b", lw=1.6)
    ax.plot(np.arange(n), bits[:n], "o", color="#c0392b", ms=4)
    ax.set_xlim(-0.3, n - 0.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("比特序号")
    ax.set_ylabel("比特值")
    ones = int(np.sum(bits))
    ax.set_title(f"信源比特流（前 {n} bit；共 {len(bits)} bit，"
                 f"1 占比 {ones/len(bits):.3f}）")
    ax.grid(alpha=0.3)


# ============================== 2. 信道编码 ============================== #
def plot_coding(ax, bits, coded, show=24):
    n = min(show, len(bits))
    m = min(show * 2 + 4, len(coded))
    ax2 = ax.twiny()
    ax.step(np.arange(n) + 0.5, bits[:n], where="post",
            color="#c0392b", lw=1.8, label="信息比特")
    ax2.step(np.arange(m) + 0.5, coded[:m], where="post",
             color="#21618c", lw=1.2, alpha=0.8, label="编码比特")
    ax.set_xlim(0, n)
    ax2.set_xlim(0, m)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("信息比特序号")
    ax2.set_xlabel("编码比特序号")
    ax.set_ylabel("比特值")
    ax.set_title(f"编码前后对比（信息 {len(bits)} bit -> 编码 {len(coded)} bit）")
    ax.grid(alpha=0.3)


# ============================== 3. 调制与成形 ============================== #
def plot_constellation(ax, symbols, mod_type, title="发送星座"):
    if mod_type == "BPSK":
        ax.scatter(_real(symbols), np.zeros(len(symbols)),
                   s=18, alpha=0.5, color="#1f77b4")
        ax.scatter([-1, 1], [0, 0], marker="*", s=260, color="k")
        ax.annotate("0", (1, 0), xytext=(6, 8), textcoords="offset points")
        ax.annotate("1", (-1, 0), xytext=(6, 8), textcoords="offset points")
        ax.set_ylim(-1, 1)
    else:
        ax.scatter(_real(symbols), np.imag(symbols), s=14,
                   alpha=0.5, color="#1f77b4")
        for s, lab in zip(QPSK_CONSTELLATION, QPSK_BITS):
            ax.plot(s.real, s.imag, "k*", ms=16)
            ax.annotate(f"{lab[0]}{lab[1]}", (s.real, s.imag),
                        xytext=(7, 7), textcoords="offset points")
        ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("同相 I")
    ax.set_ylabel("正交 Q")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_tx_waveform(ax, tx, sps, span, show=12):
    t = waveform_time(len(tx), sps, span / 2)
    n = min(show * sps + sps, len(tx))
    ax.plot(t[:n], _real(tx[:n]), color="#1f77b4", lw=1.3)
    if np.iscomplexobj(tx):
        ax.plot(t[:n], np.imag(tx[:n]), color="#ff7f0e", lw=1.1, alpha=0.8)
    for k in range(show + 1):
        ax.axvline(k, color="gray", ls=":", lw=0.5, alpha=0.6)
    ax.set_xlabel("时间 t / T")
    ax.set_ylabel("幅度")
    ax.set_title(f"成形后发送波形（前 {show} 个符号）")
    ax.grid(alpha=0.3)


def plot_pulse(ax, t_pulse, h, beta, kind):
    ax.plot(t_pulse, h, color="#6c3483", lw=1.6)
    ax.set_xlabel("时间 t / T")
    ax.set_ylabel("h(t)")
    ax.set_title(f"{kind} 脉冲（滚降 {beta}，能量 {np.sum(h**2):.3f}）")
    ax.grid(alpha=0.3)


# ============================== 4. 信道 ============================== #
def plot_channel(ax, tx, rx, sps, span, show=12):
    t = waveform_time(len(tx), sps, span / 2)
    n = min(show * sps + sps, len(tx))
    ax.plot(t[:n], _real(tx[:n]), color="#2ca02c", lw=1.2, label="发送波形")
    ax.plot(t[:n], _real(rx[:n]), color="#d62728", lw=1.0,
            alpha=0.8, label="接收波形")
    ax.set_xlabel("时间 t / T")
    ax.set_ylabel("幅度")
    ax.set_title("AWGN 信道：发送 vs 接收")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)


def plot_noise_hist(ax, tx, rx):
    noise = _real(rx) - _real(tx)
    ax.hist(noise, bins=60, density=True, color="#7f8c8d", alpha=0.8)
    sigma = np.std(noise)
    x = np.linspace(-4 * sigma, 4 * sigma, 300)
    ax.plot(x, np.exp(-x ** 2 / (2 * sigma ** 2))
            / (np.sqrt(2 * np.pi) * sigma), "r-", lw=1.6,
            label=f"高斯拟合 σ={sigma:.3f}")
    ax.set_xlabel("噪声采样值")
    ax.set_ylabel("概率密度")
    ax.set_title("噪声幅度分布（应服从高斯）")
    ax.legend()
    ax.grid(alpha=0.3)


# ============================== 5. 匹配滤波 ============================== #
def plot_mf(ax, mf, t_axis, sps, show=14):
    n = min(show * sps, len(mf))
    ax.plot(t_axis[:n], _real(mf[:n]), color="#1f77b4", lw=1.3)
    ax.axhline(0, color="gray", lw=0.7)
    for k in range(show):
        ax.axvline(k, color="gray", ls=":", lw=0.5, alpha=0.6)
    ax.set_xlabel("时间 t / T")
    ax.set_ylabel("幅度")
    ax.set_title("匹配滤波输出波形")
    ax.grid(alpha=0.3)


def plot_psd(ax, wave, sps, title="功率谱密度"):
    f, P = welch(_real(wave), fs=sps, nperseg=min(4096, len(wave)),
                 noverlap=None, window="hann")
    P_dB = 10 * np.log10(P + 1e-15)
    P_dB -= np.max(P_dB)
    ax.plot(f, P_dB, color="#21618c", lw=1.4)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-70, 5)
    ax.set_xlabel("归一化频率 fT")
    ax.set_ylabel("PSD / dB")
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ============================== 6. 采样判决 / 同步 ============================== #
def plot_sample_points(ax, mf, t_axis, samples, cfg, show=12):
    m = (t_axis >= -0.5) & (t_axis <= show + 0.5)
    ax.plot(t_axis[m], _real(mf[m]), color="#1f77b4", lw=1.2,
            label="匹配滤波输出")
    n = min(show, len(samples))
    if cfg.timing.mode == "ideal":
        ax.scatter(np.arange(n), _real(samples[:n]), marker="o",
                   s=55, facecolors="none", edgecolors="#2ca02c",
                   lw=1.5, label="理想采样点")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel("时间 t / T")
    ax.set_ylabel("幅度")
    ax.set_title("采样点位置")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)


def plot_timing_overview(fig, data, cfg):
    """早迟门四连图：采样点 / 偏差收敛 / S 曲线 / 判决样本。"""
    fig.clear()
    ax1, ax2 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)
    ax3, ax4 = fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
    sps, span = cfg.modulation.sps, cfg.modulation.span
    mf, t_axis = data["mf_wave"], data["t_axis"]
    show = 12

    m = (t_axis >= -0.5) & (t_axis <= show + 0.5)
    ax1.plot(t_axis[m], _real(mf[m]), color="#1f77b4", lw=1.1)
    tau = data["tau_hist"]
    n = min(show, len(tau))
    cs = CubicSpline(t_axis, mf)
    unsync = _real(cs(np.arange(n) + cfg.timing.tau0))
    ax1.scatter(np.arange(n) + cfg.timing.tau0, unsync,
                marker="v", s=55, color="#d62728", label="未同步采样点")
    ax1.scatter(np.arange(n) + tau[:n], _real(data["samples"][:n]),
                marker="o", s=45, facecolors="none", edgecolors="#2ca02c",
                lw=1.4, label="跟踪采样点")
    ax1.set_title("波形与采样点（未同步 vs 跟踪锁定）")
    ax1.set_xlabel("t / T")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(tau, color="#1f77b4", lw=1.2)
    ax2.axhline(0, color="red", ls="--", lw=1)
    jitter = data.get("jitter_rms", 0.0)
    ax2.set_title(f"定时偏差收敛（稳态抖动 RMS={jitter:.4f} T）")
    ax2.set_xlabel("符号序号")
    ax2.set_ylabel("τ / T")
    ax2.grid(alpha=0.3)

    taus, curve = s_curve_scan(mf, t_axis, data["tx_symbols"],
                               cfg.modulation.mod_type, cfg.timing.delta)
    ax3.plot(taus, curve, color="#ff7f0e", lw=1.6)
    ax3.axhline(0, color="gray", lw=0.7)
    ax3.axvline(0, color="red", ls=":", lw=1)
    ax3.set_title("鉴相 S 曲线")
    ax3.set_xlabel("τ / T")
    ax3.grid(alpha=0.3)

    steady = slice(len(data["samples"]) // 2, len(data["samples"]))
    vals = _real(data["samples"][steady])
    ax4.scatter(vals, np.random.default_rng(0).normal(0, 0.04, len(vals)),
                s=12, alpha=0.4, color="#2ca02c")
    ax4.axvline(0, color="gray", ls="--", lw=0.8)
    ax4.set_title("锁定后判决样本分布")
    ax4.set_xlabel("样本值")
    ax4.set_yticks([])
    ax4.grid(alpha=0.3)

    fig.tight_layout()


def plot_decision_samples(ax, samples, mod_type, title):
    vals = _real(samples)
    ax.scatter(vals, np.random.default_rng(1).normal(0, 0.04, len(vals)),
               s=14, alpha=0.4, color="#1f77b4")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("判决样本值")
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ============================== 7. 解码 ============================== #
def plot_decoding(ax, bits, decoded, show=60):
    n = min(show, len(bits))
    err = bits[:n] != decoded[:n]
    ax.step(np.arange(n), bits[:n], where="post", color="#c0392b",
            lw=1.4, label="原始比特")
    ax.step(np.arange(n), decoded[:n], where="post", color="#21618c",
            lw=1.2, alpha=0.7, label="解码比特")
    for i in np.flatnonzero(err):
        ax.axvline(i, color="red", alpha=0.4, lw=2)
    ax.set_xlim(-0.3, n - 0.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("比特序号")
    ax.set_title(f"解码对比（红竖线为误码；前 {n} bit）")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)


# ============================== 8. BER 扫描 ============================== #
def plot_ber_scan(ax, scan_result):
    grid = np.array(scan_result["snr_grid"])
    markers = ["s", "^", "o", "D", "v"]
    colors = ["#555555", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for i, (key, d) in enumerate(scan_result["schemes"].items()):
        curve = np.array(d["ber"])
        masked = np.ma.masked_where(curve == 0, curve)
        ax.semilogy(grid, masked, marker=markers[i % 5], color=colors[i % 5],
                    lw=1.5, ms=6, label=d["name"])
    ax.semilogy(grid, scan_result["theory"], "--", color="black",
                lw=1.1, label=f"未编码理论值")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("误比特率 BER")
    ax.set_title(f"{scan_result['mod_type']} 下各编码方案 BER 对比")
    ax.grid(which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=9)
    ax.set_xticks(grid)


def plot_decision_boundary(ax, mod_type, lim=1.8, res=220):
    """QPSK Voronoi 判决区域；BPSK 画一维判决点。"""
    if mod_type == "BPSK":
        ax.axvline(0, color="k", lw=1.6)
        ax.axvspan(-lim, 0, color="#ffe0d6", alpha=0.5)
        ax.axvspan(0, lim, color="#d6e8ff", alpha=0.5)
        ax.plot([-1, 1], [0, 0], "k*", ms=16)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    else:
        xx, yy = np.meshgrid(np.linspace(-lim, lim, res),
                             np.linspace(-lim, lim, res))
        grid_rx = xx + 1j * yy
        decision = nearest_neighbor(grid_rx.ravel(),
                                    QPSK_CONSTELLATION).reshape(xx.shape)
        colors = ["#d6e8ff", "#ffe0d6", "#e6d6ff", "#d6f5e3"]
        ax.contourf(xx, yy, decision,
                    levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
                    colors=colors, alpha=0.75)
        ax.contour(xx, yy, decision, levels=[0.5, 1.5, 2.5],
                   colors="k", linewidths=1.2)
        for s, lab in zip(QPSK_CONSTELLATION, QPSK_BITS):
            ax.plot(s.real, s.imag, "k*", ms=15)
            ax.annotate(f"{lab[0]}{lab[1]}", (s.real, s.imag),
                        xytext=(7, 7), textcoords="offset points")
        ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("同相 I")
    ax.set_ylabel("正交 Q")
    ax.set_title("最近邻判决边界")
    ax.grid(alpha=0.25)


# ============================== 9. 基带码型 ============================== #
def _linecode_symbols(name, bits):
    if name == "HDB3":
        return hdb3_encode(bits)[0]
    return LINE_CODE_FUNCS[name](bits)


def plot_linecode_overview(fig, bits, name, sps=32, alpha=0.5):
    fig.clear()
    ax1, ax2, ax3 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), \
        fig.add_subplot(2, 2, 3)
    sym_sps = sps // 2 if name == "Manchester" else sps
    sym = _linecode_symbols(name, bits)
    wave = np.repeat(sym, sym_sps)
    t = np.arange(len(wave)) / sps

    ax1.step(t, wave, where="post", color="#c0392b", lw=1.3)
    ax1.set_xlim(0, min(len(bits), 24))
    ax1.set_title(f"{name} 时域波形")
    ax1.set_xlabel("码元序号")
    ax1.grid(alpha=0.3)

    # 眼图用升余弦成形后的长序列
    rng = np.random.default_rng(7)
    long_bits = rng.integers(0, 2, 300)
    long_sym = _linecode_symbols(name, long_bits)
    from .blocks.waveforms import design_pulse
    _, h = design_pulse("RC", alpha, sym_sps, 8)
    shaped = np.convolve(_impulse(long_sym, sym_sps), h)
    eye_diagram(ax2, shaped, sym_sps, f"{name} 眼图")

    f, P = welch(wave, fs=sps, nperseg=min(4096, len(wave)), window="hann")
    P_dB = 10 * np.log10(P + 1e-15)
    P_dB -= np.max(P_dB)
    ax3.plot(f, P_dB, color="#21618c", lw=1.4)
    ax3.set_xlim(0, 2.2)
    ax3.set_ylim(-70, 5)
    ax3.set_title(f"{name} 功率谱")
    ax3.set_xlabel("归一化频率 fT")
    ax3.grid(alpha=0.3)
    fig.tight_layout()


def _impulse(sym, sps):
    up = np.zeros(len(sym) * sps)
    up[::sps] = sym
    return up


# ============================== 10. PCM ============================== #
def plot_pcm_chain(fig, fs_cont=10000, fs_pcm=1000, duration=0.05, n_bits=4):
    fig.clear()
    ax1, ax2, ax3 = (fig.add_subplot(3, 1, 1), fig.add_subplot(3, 1, 2),
                     fig.add_subplot(3, 1, 3))
    t_cont = np.arange(0, duration, 1 / fs_cont)
    t_pcm = np.arange(0, duration, 1 / fs_pcm)
    x_cont = pcm_block.generate_signal(t_cont)
    x_sample = pcm_block.generate_signal(t_pcm)
    x_q, idx, _ = pcm_block.uniform_quantize(x_sample, n_bits)
    x_rec = pcm_block.decode_codes(pcm_block.encode_natural(idx, n_bits))

    ax1.plot(t_cont * 1000, x_cont, color="#1f77b4", lw=1.4, label="模拟信号")
    ax1.stem(t_pcm * 1000, x_sample, linefmt="r", markerfmt="ro",
             basefmt=" ", label="抽样")
    ax1.set_title(f"① 抽样（{fs_pcm} Hz）")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(t_cont * 1000, x_cont, color="#bbb", lw=1.2)
    ax2.step(t_pcm * 1000, x_q, where="mid", color="#d62728",
             lw=1.5, label=f"{n_bits} 位量化")
    ax2.set_title("② 量化（阶梯近似）")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.plot(t_cont * 1000, x_cont, color="#bbb", lw=1.2)
    ax3.plot(t_pcm * 1000, x_rec, "o-", color="#ff7f0e", ms=3,
             lw=1.1, label="PCM 重建")
    ax3.set_title("③ 译码重建")
    ax3.set_xlabel("时间 / ms")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig.tight_layout()


def plot_pcm_snr(fig, fs_cont=10000):
    fig.clear()
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    t = np.arange(0, 1.0, 1 / fs_cont)
    sine = np.sin(2 * np.pi * 100 * t)
    bits_range = np.arange(1, 13)
    measured, theory = [], []
    for n in bits_range:
        q, _, _ = pcm_block.uniform_quantize(sine, n)
        measured.append(pcm_block.snr_db(sine, sine - q))
        theory.append(6.02 * n + 1.76)
    ax1.plot(bits_range, theory, "b-", lw=1.6, label="理论 6.02N+1.76")
    ax1.plot(bits_range, measured, "ro", ms=5, label="实测")
    ax1.set_xlabel("量化位数 N")
    ax1.set_ylabel("SNR / dB")
    ax1.set_title("量化 SNR 理论验证")
    ax1.legend()
    ax1.grid(alpha=0.3)

    xx = np.linspace(-1, 1, 500)
    ax2.plot(xx, xx, "k--", lw=1, label="y=x")
    ax2.plot(xx, pcm_block.alaw_compress(xx), "r-", lw=1.8, label="A 律压缩")
    ax2.set_xlabel("输入")
    ax2.set_ylabel("输出")
    ax2.set_title("A 律压扩特性")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
