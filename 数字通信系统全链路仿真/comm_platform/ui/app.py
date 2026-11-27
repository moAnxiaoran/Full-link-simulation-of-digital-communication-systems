# -*- coding: utf-8 -*-
"""
仿真平台主窗口
==============
组装左侧链路框图、顶部操作栏、右侧“参数 + 结果”选项卡。

设计原则：
    * 所有参数在界面直接修改，修改后下游模块自动置脏，流程图实时提示；
    * 运行在后台线程，界面不卡顿，进度与日志实时可见；
    * 配置可导入 / 导出 JSON，配合固定随机种子实现严格可复现；
    * 结果可一键保存（配置快照、NPZ、BER 的 CSV、日志）。
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
from matplotlib import pyplot as plt

from ..config import SimConfig, load_config, save_config, validate_config
from ..engine import BLOCKS, Engine
from .. import plotting as P
from .flow import FlowCanvas
from .params import ParamForm
from .results import DataTable, MetricCards, MplFrame

# matplotlib 中文与负号
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_CODE_NAME = {
    "none": "无编码", "rep31": "重复码(3,1)", "ham74": "汉明码(7,4)",
    "conv213": "卷积码(2,1,3)", "bch157": "BCH(15,7)",
}
_NAME_CODE = {v: k for k, v in _CODE_NAME.items()}


class PlatformApp(tk.Tk):
    def __init__(self, cfg: SimConfig | None = None):
        super().__init__()
        self.title("数字通信系统全链路仿真平台")
        self.geometry("1500x920")
        self.minsize(1200, 760)

        self.engine = Engine(cfg or SimConfig())
        self.busy = False
        self._stop_scan = False
        self.forms: dict[str, ParamForm] = {}
        self.mpls: dict[str, MplFrame] = {}
        self._block_tab: dict[str, int] = {}

        self._build_topbar()
        self._build_body()
        self._build_statusbar()
        self._sync_form_states()
        # 基带码型 / PCM 选项卡首次打开即出图，无需先改参数
        self._draw_linecode()
        self._draw_pcm()
        self.after(120, lambda: self.state("zoomed"))

    # ================================================================== #
    # 界面骨架
    # ================================================================== #
    def _build_topbar(self):
        bar = tk.Frame(self, bg="#1565c0", height=46)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(bar, text="数字通信全链路仿真平台", bg="#1565c0", fg="white",
                 font=("Microsoft YaHei", 13, "bold")).pack(side=tk.LEFT, padx=14)

        btns = [
            ("全链路运行", self._click_run_all),
            ("单步运行", self._click_step),
            ("BER 扫描", lambda: self.nb.select(self._tab["scan"])),
            ("保存结果", self._click_save_results),
            ("导入配置", self._click_import),
            ("导出配置", self._click_export),
            ("恢复默认", self._click_reset),
            ("运行日志", self._show_logs),
        ]
        self.top_buttons = []
        for text, cmd in btns:
            b = tk.Button(bar, text=text, command=cmd, relief="flat",
                          bg="#ffffff", fg="#1565c0",
                          font=("Microsoft YaHei", 9), padx=10, pady=3,
                          activebackground="#e3f2fd")
            b.pack(side=tk.LEFT, padx=4, pady=8)
            self.top_buttons.append(b)

    def _build_body(self):
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.flow = FlowCanvas(paned, on_click=self._flow_click)
        paned.add(self.flow, weight=0)

        self.nb = ttk.Notebook(paned)
        paned.add(self.nb, weight=1)
        self._tab = {}

        self._build_overview_tab()
        self._build_module_tabs()
        self._build_scan_tab()
        self._build_linecode_tab()
        self._build_pcm_tab()

    def _build_statusbar(self):
        sb = tk.Frame(self, bg="#eceff1", height=26)
        sb.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="就绪。修改参数后点击“全链路运行”，或点击流程图选择模块。")
        tk.Label(sb, textvariable=self.status_var, bg="#eceff1",
                 anchor="w", font=("Microsoft YaHei", 9)).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.progress = ttk.Progressbar(sb, mode="determinate", length=240)
        self.progress.pack(side=tk.RIGHT, padx=10)

    # ------------------------- 选项卡 ------------------------- #
    def _new_tab(self, key, title):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=title)
        self._tab[key] = self.nb.index(frame)
        return frame

    def _new_form(self, parent, section):
        form = ParamForm(parent, on_change=self.on_param_change)
        form.pack(side=tk.TOP, fill=tk.X)
        ttk.Separator(parent).pack(fill=tk.X)
        self.forms[section] = form
        return form

    def _new_mpl(self, parent, key, nrows=1, ncols=1, figsize=(7.5, 4.4)):
        mpl = MplFrame(parent, nrows, ncols, figsize)
        mpl.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.mpls[key] = mpl
        return mpl

    def _build_overview_tab(self):
        frame = self._new_tab("overview", "总览")
        self.cards = MetricCards(frame)
        self.cards.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        for key, title in [
            ("n_bits", "信息比特数"), ("scheme", "信道编码"),
            ("rate", "码率 R"), ("mod", "调制方式"),
            ("ebno", "Eb/N0 (dB)"), ("pre_ber", "编码域 BER"),
            ("info_ber", "信息 BER"), ("n_err", "误比特数"),
            ("theory", "理论 BER"), ("jitter", "定时抖动(T)"),
        ]:
            self.cards.add(key, title)

        ttk.Label(frame, text="全链路数据维度", font=("Microsoft YaHei", 10, "bold")
                  ).pack(anchor="w", padx=14)
        self.overview_table = DataTable(
            frame, ["block", "in", "out", "size"],
            ["模块", "输入", "输出", "长度"], heights=10)
        self.overview_table.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

    def _build_module_tabs(self):
        titles = {
            "source": "① 比特生成", "coding": "② 信道编码",
            "modulation": "③ 调制成形", "channel": "④ 信道",
            "filter": "⑤ 匹配滤波", "timing": "⑥ 采样判决",
            "decoding": "⑦ 解码", "metrics": "⑧ 误码统计",
        }
        for block in BLOCKS:
            f = self._new_tab(block, titles[block])
            self._block_tab[block] = self._tab[block]

        # ----- ① 信源 ----- #
        f = self._new_form(self._tab_frame("source"), "source")
        f.title("信源参数")
        f.add_combo("source.mode", "生成模式",
                    ["random", "manual", "zeros", "ones"], "random")
        f.add_entry("source.n_bits", "比特数", 105000, int)
        f.add_entry("source.seed", "随机种子", 42, int)
        f.add_entry("source.manual_bits", "手动比特串",
                    "101010110011", str)
        self._new_mpl(self._tab_frame("source"), "source")

        # ----- ② 信道编码 ----- #
        f = self._new_form(self._tab_frame("coding"), "coding")
        f.title("信道编码参数")
        f.add_combo("coding.scheme", "编码方案",
                    list(_NAME_CODE), _CODE_NAME["none"])
        f.add_entry("coding.conv_frame", "卷积码帧长", 500, int)
        self._new_mpl(self._tab_frame("coding"), "coding")

        # ----- ③ 调制成形 ----- #
        f = self._new_form(self._tab_frame("modulation"), "modulation")
        f.title("调制与成形参数")
        f.add_combo("modulation.mod_type", "调制方式",
                    ["BPSK", "QPSK"], "BPSK")
        f.add_combo("modulation.pulse", "脉冲类型",
                    ["RRC", "RC", "rect"], "RRC")
        f.add_entry("modulation.sps", "每符号采样点", 16, int)
        f.add_entry("modulation.beta", "滚降系数", 0.5, float)
        f.add_entry("modulation.span", "滤波器跨度", 8, int)
        self._new_mpl(self._tab_frame("modulation"), "modulation",
                      1, 3, (11, 3.8))

        # ----- ④ 信道 ----- #
        f = self._new_form(self._tab_frame("channel"), "channel")
        f.title("信道参数")
        f.add_check("channel.add_noise", "加入 AWGN 噪声", True)
        f.add_entry("channel.ebno_db", "Eb/N0 (dB)", 8.0, float)
        f.add_entry("channel.noise_seed", "噪声种子", 2026, int)
        self._new_mpl(self._tab_frame("channel"), "channel",
                      1, 2, (10, 4.2))

        # ----- ⑤ 匹配滤波（无独立参数）----- #
        tip = ttk.Label(self._tab_frame("filter"),
                        text="滤波器参数继承“③ 调制成形”，修改成形参数后本模块自动置脏。",
                        background="#fff8e1")
        tip.pack(fill=tk.X)
        self._new_mpl(self._tab_frame("filter"), "filter",
                      1, 3, (11, 3.8))

        # ----- ⑥ 采样判决 ----- #
        f = self._new_form(self._tab_frame("timing"), "timing")
        f.title("采样与位同步参数")
        f.add_combo("timing.mode", "采样模式",
                    ["ideal", "early_late"], "ideal")
        f.add_entry("timing.tau0", "初始偏差 tau0 (T)", 0.35, float)
        f.add_entry("timing.delta", "早迟门偏移 delta (T)", 0.5, float)
        f.add_entry("timing.kp", "比例系数 kp", 0.06, float)
        f.add_entry("timing.ki", "积分系数 ki", 0.006, float)
        self._new_mpl(self._tab_frame("timing"), "timing",
                      2, 2, (9.5, 6.6))

        # ----- ⑦ 解码 ----- #
        ttk.Label(self._tab_frame("decoding"),
                  text="解码器与“② 信道编码”自动配套。",
                  background="#e8f5e9").pack(fill=tk.X)
        self._new_mpl(self._tab_frame("decoding"), "decoding")

        # ----- ⑧ 误码统计 ----- #
        self.metric_cards = MetricCards(self._tab_frame("metrics"), ncols=3)
        self.metric_cards.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        for key, title in [
            ("pre_ber", "编码域 BER（解码前）"),
            ("info_ber", "信息 BER（解码后）"),
            ("n_err", "误比特数"),
            ("theory", "未编码理论 BER"),
            ("jitter", "稳态定时抖动 RMS (T)"),
            ("n_sent", "发送符号数"),
        ]:
            self.metric_cards.add(key, title)
        ttk.Label(self._tab_frame("metrics"),
                  text="说明：理论 BER 为未编码 BPSK/QPSK 参考值；编码方案通过纠错使信息 BER 显著低于编码域 BER。",
                  background="#f5f5f5", wraplength=900, justify="left").pack(
            fill=tk.X, padx=10, pady=6)

    def _build_scan_tab(self):
        frame = self._new_tab("scan", "BER 扫描")
        f = self._new_form(frame, "scan")
        f.title("扫描参数")
        f.add_entry("scan.ebno_start", "起点 Eb/N0 (dB)", 0.0, float)
        f.add_entry("scan.ebno_stop", "终点 Eb/N0 (dB)", 10.0, float)
        f.add_entry("scan.ebno_step", "步长 (dB)", 2.0, float)
        f.add_check("scan.use_target_errors", "按目标错误数自适应", False)
        f.add_entry("scan.target_errors", "目标错误数", 300, int)
        f.add_entry("scan.max_symbols", "最大符号数", 1_000_000, int)

        tk.Label(f, text="参与对比的编码方案：", bg="white").grid(
            row=f._row, column=0, sticky="w", padx=8)
        for k in ["none", "rep31", "ham74", "conv213", "bch157"]:
            f.add_check(f"scan.scheme.{k}", _CODE_NAME[k], True)

        row = tk.Frame(frame, bg="white")
        row.pack(fill=tk.X, pady=4)
        self.scan_run_btn = tk.Button(row, text="运行 BER 扫描",
                                      command=self._click_run_scan,
                                      bg="#2e7d32", fg="white", padx=12)
        self.scan_run_btn.pack(side=tk.LEFT, padx=10)
        tk.Button(row, text="停止", command=lambda: setattr(self, "_stop_scan", True),
                  bg="#c62828", fg="white", padx=12).pack(side=tk.LEFT)
        self.scan_progress = ttk.Progressbar(row, mode="determinate", length=300)
        self.scan_progress.pack(side=tk.LEFT, padx=14, fill=tk.X, expand=True)

        sub = ttk.PanedWindow(frame, orient=tk.VERTICAL)
        sub.pack(fill=tk.BOTH, expand=True)
        mpl = MplFrame(sub, figsize=(8, 4.2))
        sub.add(mpl, weight=2)
        self.mpls["scan"] = mpl
        table_holder = ttk.Frame(sub)
        sub.add(table_holder, weight=1)
        self.scan_table = DataTable(table_holder, ["snr"], ["Eb/N0(dB)"], heights=6)
        self.scan_table.pack(fill=tk.BOTH, expand=True)

    def _build_linecode_tab(self):
        frame = self._new_tab("linecode", "基带码型分析")
        f = ParamForm(frame, on_change=self._linecode_param)
        f.pack(side=tk.TOP, fill=tk.X)
        ttk.Separator(frame).pack(fill=tk.X)
        f.add_combo("lc.name", "码型",
                    ["NRZ", "Manchester", "AMI", "HDB3"], "NRZ")
        f.add_entry("lc.bits", "演示比特串",
                    "100001100001010000111000", str)
        f.add_entry("lc.sps", "每码元采样点", 32, int)
        self.forms["linecode"] = f
        self._new_mpl(frame, "linecode", figsize=(9, 6.2))

    def _build_pcm_tab(self):
        frame = self._new_tab("pcm", "PCM 脉冲编码调制")
        f = ParamForm(frame, on_change=self._pcm_param)
        f.pack(side=tk.TOP, fill=tk.X)
        ttk.Separator(frame).pack(fill=tk.X)
        f.add_entry("pcm.fs_cont", "连续信号采样率", 10000, int)
        f.add_entry("pcm.fs_pcm", "PCM 抽样频率", 1000, int)
        f.add_entry("pcm.duration", "观测时长(s)", 0.05, float)
        f.add_entry("pcm.n_bits", "量化位数", 4, int)
        self.forms["pcm"] = f

        sub = ttk.Notebook(frame)
        sub.pack(fill=tk.BOTH, expand=True)
        p1, p2 = ttk.Frame(sub), ttk.Frame(sub)
        sub.add(p1, text="抽样-量化-重建")
        sub.add(p2, text="SNR 理论与 A 律")
        self._new_mpl(p1, "pcm_chain", figsize=(9, 6.4))
        self._new_mpl(p2, "pcm_snr", 1, 2, (10, 4.2))

    def _tab_frame(self, key) -> ttk.Frame:
        return self.nametowidget(self.nb.tabs()[self._tab[key]])

    # ================================================================== #
    # 参数变化处理
    # ================================================================== #
    def on_param_change(self, key, value):
        section, field = key.split(".", 1)
        if value is None:
            self._set_status(f"参数 {key} 输入非法，已忽略，请重新输入。")
            return

        if section == "scan":
            self._apply_scan_param(field, value)
        else:
            if key == "coding.scheme":
                value = _NAME_CODE[value]
            setattr(getattr(self.engine.cfg, section), field, value)
            block = self._invalidate_block(section, field)
            if block:
                self.engine.invalidate(block)

        self._sync_form_states()
        self.flow.refresh(self.engine.status)
        self._set_status(f"参数已修改：{key} = {value}，下游模块需重新运行。")

    def _invalidate_block(self, section, field):
        if section == "source":
            return "source"
        if section == "coding":
            return "coding"
        if section == "modulation":
            return "coding" if field == "mod_type" else "modulation"
        if section == "channel":
            return "channel"
        if section == "timing":
            return "timing"
        return None

    def _apply_scan_param(self, field, value):
        cfg = self.engine.cfg.scan
        if field.startswith("scheme."):
            k = field.split(".", 1)[1]
            schemes = list(cfg.schemes)
            if value and k not in schemes:
                schemes.append(k)
            elif not value and k in schemes:
                schemes.remove(k)
            cfg.schemes = schemes
        else:
            setattr(cfg, field, value)
        self.engine.scan_result = None

    def _sync_form_states(self):
        cfg = self.engine.cfg
        sf = self.forms.get("source")
        if sf:
            mode = cfg.source.mode
            sf.set_enabled("source.n_bits", mode in ("random", "zeros", "ones"))
            sf.set_enabled("source.seed", mode == "random")
            sf.set_enabled("source.manual_bits", mode == "manual")
        tf = self.forms.get("timing")
        if tf:
            el = cfg.timing.mode == "early_late"
            for k in ("timing.tau0", "timing.delta", "timing.kp", "timing.ki"):
                tf.set_enabled(k, el)

    # ---------------- 独立演示 Tab 的参数回调 ---------------- #
    def _linecode_param(self, key, value):
        if value is None:
            self._set_status("基带码型参数非法，已忽略。")
            return
        self._draw_linecode()

    def _pcm_param(self, key, value):
        if value is None:
            self._set_status("PCM 参数非法，已忽略。")
            return
        self._draw_pcm()

    # ================================================================== #
    # 运行控制
    # ================================================================== #
    def _click_run_all(self):
        errs = validate_config(self.engine.cfg)
        if errs:
            messagebox.showerror("参数错误", "\n".join(errs))
            return
        self._start_chain(until=None)

    def _click_step(self):
        target = self.engine.first_runnable()
        if target is None:
            self._set_status("全部模块均已完成，无需单步运行。")
            return
        self._start_chain(until=target, focus=target)

    def _start_chain(self, until, focus=None):
        if self.busy:
            return
        self.busy = True
        self._set_buttons(False)
        self.progress.configure(mode="determinate", maximum=100, value=0)
        if focus:
            self.nb.select(self._block_tab[focus])

        def progress(block, done, total):
            self.after(0, lambda: self._chain_progress(block, done, total))

        def work():
            try:
                if until:
                    self.engine.run_until(until, progress)
                else:
                    self.engine.run_all(progress)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._run_failed(exc))
            else:
                self.after(0, self._run_finished)

        threading.Thread(target=work, daemon=True).start()

    def _chain_progress(self, block, done, total):
        self.progress.configure(value=done / total * 100)
        self._set_status(f"正在运行：{block}（{done}/{total}）")
        self.flow.refresh(self.engine.status)

    def _run_finished(self):
        self.busy = False
        self._set_buttons(True)
        self.flow.refresh(self.engine.status)
        self._draw_all()
        self._set_status("运行完成，结果已更新。")

    def _run_failed(self, exc):
        self.busy = False
        self._set_buttons(True)
        self.flow.refresh(self.engine.status)
        messagebox.showerror("运行出错", f"{type(exc).__name__}: {exc}")
        self._set_status("运行出错，请查看弹窗。")

    def _set_buttons(self, enabled):
        state = "normal" if enabled else "disabled"
        for b in self.top_buttons:
            b.configure(state=state)

    # ---------------- BER 扫描 ---------------- #
    def _click_run_scan(self):
        if self.busy:
            return
        errs = validate_config(self.engine.cfg)
        if errs:
            messagebox.showerror("参数错误", "\n".join(errs))
            return
        if not self.engine.cfg.scan.schemes:
            messagebox.showwarning("提示", "请至少勾选一个编码方案。")
            return

        self.busy = True
        self._stop_scan = False
        self.scan_run_btn.configure(state="disabled")
        self.scan_progress.configure(value=0)

        def progress(scheme, ebno, done, total):
            self.after(0, lambda: self._scan_progress(scheme, ebno, done, total))

        def work():
            try:
                self.engine.run_ber_scan(
                    progress, should_stop=lambda: self._stop_scan)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._run_failed(exc))
            else:
                self.after(0, self._scan_finished)

        threading.Thread(target=work, daemon=True).start()

    def _scan_progress(self, scheme, ebno, done, total):
        self.scan_progress.configure(maximum=total, value=done)
        self._set_status(f"BER 扫描：{_CODE_NAME.get(scheme, scheme)} "
                         f"@ {ebno:g} dB（{done}/{total}）")

    def _scan_finished(self):
        self.busy = False
        self.scan_run_btn.configure(state="normal")
        self._draw_scan()
        self._set_status("BER 扫描完成。")

    # ================================================================== #
    # 绘图刷新
    # ================================================================== #
    def _draw_all(self):
        d = self.engine.data
        mpl = self.mpls

        if d:
            mpl["source"].clear()
            P.plot_source(mpl["source"].axes, d["info_bits"])
            mpl["source"].redraw()

            mpl["coding"].clear()
            P.plot_coding(mpl["coding"].axes, d["info_bits"], d["coded_bits"])
            mpl["coding"].redraw()

            axes = mpl["modulation"].axes
            for ax in axes:
                ax.clear()
            mc = self.engine.cfg.modulation
            P.plot_tx_waveform(axes[0], d["tx_wave"], mc.sps, mc.span)
            P.plot_constellation(axes[1], d["tx_symbols"], mc.mod_type)
            P.plot_pulse(axes[2], d["t_pulse"], d["pulse"], mc.beta, mc.pulse)
            mpl["modulation"].redraw()

            axes = mpl["channel"].axes
            for ax in axes:
                ax.clear()
            P.plot_channel(axes[0], d["tx_wave"], d["rx_wave"], mc.sps, mc.span)
            if self.engine.cfg.channel.add_noise:
                P.plot_noise_hist(axes[1], d["tx_wave"], d["rx_wave"])
            else:
                axes[1].set_title("无噪声，未生成噪声分布")
            mpl["channel"].redraw()

            axes = mpl["filter"].axes
            for ax in axes:
                ax.clear()
            P.plot_mf(axes[0], d["mf_wave"], d["t_axis"], mc.sps)
            P.eye_diagram(axes[1], d["mf_wave"], mc.sps, "匹配滤波后眼图")
            P.plot_psd(axes[2], d["mf_wave"], mc.sps)
            mpl["filter"].redraw()

            self._draw_timing()
            mpl["decoding"].clear()
            P.plot_decoding(mpl["decoding"].axes,
                            d["info_bits"], d["decoded_bits"])
            mpl["decoding"].redraw()

            self._update_cards()
            self._update_overview_table()

    def _draw_timing(self):
        d = self.engine.data
        cfg = self.engine.cfg
        mpl = self.mpls["timing"]
        if cfg.timing.mode == "early_late":
            P.plot_timing_overview(mpl.fig, d, cfg)
            mpl.canvas.draw_idle()
        else:
            mpl.clear()
            ax1, ax2 = mpl.axes[0], mpl.axes[1]
            P.plot_sample_points(ax1, d["mf_wave"], d["t_axis"],
                                 d["samples"], cfg)
            P.plot_decision_samples(ax2, d["samples"],
                                    cfg.modulation.mod_type, "判决样本分布")
            # 2x2 布局下理想模式只使用前两个子图，隐藏其余两个
            for ax in mpl.axes[2:]:
                ax.set_visible(False)
            mpl.redraw()

    def _update_cards(self):
        d = self.engine.data
        cfg = self.engine.cfg
        vals = {
            "n_bits": len(d["info_bits"]),
            "scheme": d["codec"].name,
            "rate": f"{d['codec'].rate:.4f}",
            "mod": cfg.modulation.mod_type,
            "ebno": f"{cfg.channel.ebno_db:g}" if cfg.channel.add_noise else "无噪",
            "pre_ber": f"{d['pre_decode_ber']:.2e}",
            "info_ber": f"{d['info_ber']:.2e}",
            "n_err": str(d["n_errors"]),
            "theory": f"{d['theory_ber_point']:.2e}",
            "jitter": f"{d.get('jitter_rms', 0.0):.4f}",
        }
        for k, v in vals.items():
            self.cards.set(k, v)

        mc = self.metric_cards
        mc.set("pre_ber", f"{d['pre_decode_ber']:.2e}")
        mc.set("info_ber", f"{d['info_ber']:.2e}")
        mc.set("n_err", str(d["n_errors"]))
        mc.set("theory", f"{d['theory_ber_point']:.2e}")
        mc.set("jitter", f"{d.get('jitter_rms', 0.0):.4f}")
        mc.set("n_sent", str(len(d["tx_symbols"])))

    def _update_overview_table(self):
        d = self.engine.data
        rows = [
            ["① 信源", "种子/模式", "信息比特", len(d["info_bits"])],
            ["② 编码", "信息比特", "编码比特", len(d["coded_bits"])],
            ["③ 调制", "编码比特", "发送符号", len(d["tx_symbols"])],
            ["③ 成形", "发送符号", "发送波形点", len(d["tx_wave"])],
            ["④ 信道", "发送波形", "接收波形点", len(d["rx_wave"])],
            ["⑤ 匹配滤波", "接收波形", "滤波波形点", len(d["mf_wave"])],
            ["⑥ 采样判决", "滤波波形", "判决样本", len(d["samples"])],
            ["⑦ 解码", "硬判决比特", "解码比特", len(d["decoded_bits"])],
            ["⑧ 统计", "原始/解码", "误比特数", d["n_errors"]],
        ]
        self.overview_table.fill(rows)

    def _draw_scan(self):
        res = self.engine.scan_result
        if not res:
            return
        mpl = self.mpls["scan"]
        mpl.clear()
        P.plot_ber_scan(mpl.axes, res)
        mpl.redraw()

        grid = res["snr_grid"]
        cols = ["snr"] + list(res["schemes"].keys())
        heads = ["Eb/N0(dB)"] + [d["name"] for d in res["schemes"].values()]
        self.scan_table.tree.configure(columns=cols)
        self.scan_table.tree.delete(*self.scan_table.tree.get_children())
        for c, h in zip(cols, heads):
            self.scan_table.tree.heading(c, text=h)
            self.scan_table.tree.column(c, anchor="center", width=120)
        for i, snr in enumerate(grid):
            row = [f"{snr:g}"]
            for d in res["schemes"].values():
                row.append(f"{d['ber'][i]:.2e}")
            self.scan_table.tree.insert("", tk.END, values=row)

    def _draw_linecode(self):
        f = self.forms["linecode"]
        name = f.get("lc.name")
        bits_str = f.get("lc.bits") or ""
        sps = f.get("lc.sps")
        if not bits_str or set(bits_str) > {"0", "1"} or not sps:
            return
        bits = [int(c) for c in bits_str]
        mpl = self.mpls["linecode"]
        P.plot_linecode_overview(mpl.fig, bits, name, sps)
        mpl.canvas.draw_idle()

    def _draw_pcm(self):
        f = self.forms["pcm"]
        fs_cont = f.get("pcm.fs_cont")
        fs_pcm = f.get("pcm.fs_pcm")
        duration = f.get("pcm.duration")
        n_bits = f.get("pcm.n_bits")
        if not all(x is not None for x in (fs_cont, fs_pcm, duration, n_bits)):
            return
        if fs_pcm <= 0 or duration <= 0 or n_bits < 1 or fs_cont < fs_pcm:
            self._set_status("PCM 参数不合法（需 fs_cont>=fs_pcm，时长>0，位数>=1）。")
            return
        m1, m2 = self.mpls["pcm_chain"], self.mpls["pcm_snr"]
        P.plot_pcm_chain(m1.fig, fs_cont, fs_pcm, duration, n_bits)
        P.plot_pcm_snr(m2.fig, fs_cont)
        m1.canvas.draw_idle()
        m2.canvas.draw_idle()

    # ================================================================== #
    # 其他交互
    # ================================================================== #
    def _flow_click(self, block):
        self.nb.select(self._block_tab[block])

    def _click_save_results(self):
        if not self.engine.data and not self.engine.scan_result:
            messagebox.showinfo("提示", "还没有可保存的结果，请先运行仿真。")
            return
        try:
            saved = self.engine.save_results()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失败", str(exc))
            return
        messagebox.showinfo("保存成功",
                            "已保存到：\n" + "\n".join(saved.values()))

    def _click_export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON 配置", "*.json")],
            initialfile="sim_config.json")
        if path:
            save_config(self.engine.cfg, path)
            self._set_status(f"配置已导出：{path}")

    def _click_import(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON 配置", "*.json")])
        if not path:
            return
        try:
            cfg = load_config(path)
            errs = validate_config(cfg)
            if errs:
                messagebox.showerror("配置非法", "\n".join(errs))
                return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导入失败", str(exc))
            return
        self.engine = Engine(cfg)
        self._load_cfg_into_forms()
        self.flow.refresh(self.engine.status)
        self._set_status(f"配置已导入：{path}，请运行仿真。")

    def _load_cfg_into_forms(self):
        cfg = self.engine.cfg
        mapping = {
            "source": [("source.mode", cfg.source.mode),
                       ("source.n_bits", cfg.source.n_bits),
                       ("source.seed", cfg.source.seed),
                       ("source.manual_bits", cfg.source.manual_bits)],
            "coding": [("coding.scheme", _CODE_NAME[cfg.coding.scheme]),
                       ("coding.conv_frame", cfg.coding.conv_frame)],
            "modulation": [("modulation.mod_type", cfg.modulation.mod_type),
                           ("modulation.pulse", cfg.modulation.pulse),
                           ("modulation.sps", cfg.modulation.sps),
                           ("modulation.beta", cfg.modulation.beta),
                           ("modulation.span", cfg.modulation.span)],
            "channel": [("channel.add_noise", cfg.channel.add_noise),
                        ("channel.ebno_db", cfg.channel.ebno_db),
                        ("channel.noise_seed", cfg.channel.noise_seed)],
            "timing": [("timing.mode", cfg.timing.mode),
                       ("timing.tau0", cfg.timing.tau0),
                       ("timing.delta", cfg.timing.delta),
                       ("timing.kp", cfg.timing.kp),
                       ("timing.ki", cfg.timing.ki)],
            "scan": [("scan.ebno_start", cfg.scan.ebno_start),
                     ("scan.ebno_stop", cfg.scan.ebno_stop),
                     ("scan.ebno_step", cfg.scan.ebno_step),
                     ("scan.use_target_errors", cfg.scan.use_target_errors),
                     ("scan.target_errors", cfg.scan.target_errors),
                     ("scan.max_symbols", cfg.scan.max_symbols)],
        }
        for section, pairs in mapping.items():
            form = self.forms.get(section)
            if not form:
                continue
            for k, v in pairs:
                form.set(k, str(v))
        # scan 方案勾选
        sform = self.forms["scan"]
        for k in ["none", "rep31", "ham74", "conv213", "bch157"]:
            sform.set(f"scan.scheme.{k}", k in cfg.scan.schemes)
        self._sync_form_states()

    def _click_reset(self):
        if not messagebox.askyesno("确认", "确定恢复全部默认参数？当前修改将丢失。"):
            return
        self.engine = Engine(SimConfig())
        self._load_cfg_into_forms()
        self.flow.refresh(self.engine.status)
        self._set_status("已恢复默认参数。")

    def _show_logs(self):
        win = tk.Toplevel(self)
        win.title("运行日志")
        win.geometry("640x420")
        txt = tk.Text(win, wrap="word", font=("Consolas", 9))
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", "\n".join(self.engine.logs) or "暂无日志。")
        txt.configure(state="disabled")

    def _set_status(self, msg):
        self.status_var.set(msg)


def launch():
    app = PlatformApp()
    app.mainloop()
