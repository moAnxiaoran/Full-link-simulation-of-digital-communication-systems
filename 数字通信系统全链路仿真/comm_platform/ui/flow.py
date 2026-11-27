# -*- coding: utf-8 -*-
"""
左侧全链路流程框图
==================
8 个模块按信号流向排列，模块颜色实时反映运行状态：
    灰色=未运行，黄色=参数已变待重跑，绿色=已完成；
点击任意模块通知主窗口切换到对应选项卡。
"""

import tkinter as tk


FLOW_BLOCKS = [
    ("source", "① 比特生成"),
    ("coding", "② 信道编码"),
    ("modulation", "③ 调制成形"),
    ("channel", "④ 信道"),
    ("filter", "⑤ 匹配滤波"),
    ("timing", "⑥ 采样判决"),
    ("decoding", "⑦ 解码"),
    ("metrics", "⑧ 误码统计"),
]

_STATUS_COLOR = {
    "idle": ("#eceff1", "#90a4ae", "未运行"),
    "dirty": ("#fff3cd", "#e0a800", "待重跑"),
    "done": ("#d4edda", "#28a745", "已完成"),
}


class FlowCanvas(tk.Canvas):
    def __init__(self, parent, on_click=None, **kw):
        super().__init__(parent, width=220, bg="white",
                         highlightthickness=0, **kw)
        self.on_click = on_click
        self._rects = {}
        self._selected = None
        self._build()
        self.bind("<Configure>", lambda e: self._build())

    def _build(self):
        self.delete("all")
        self._rects.clear()
        w = max(self.winfo_width(), 220)
        box_w, box_h, gap = w - 40, 52, 22
        x0, y = 20, 18

        for i, (key, label) in enumerate(FLOW_BLOCKS):
            r = self.create_rectangle(x0, y, x0 + box_w, y + box_h,
                                      fill="#eceff1", outline="#90a4ae",
                                      width=1.5, tags=("box", key))
            t = self.create_text(x0 + box_w / 2, y + box_h / 2,
                                 text=label, font=("Microsoft YaHei", 11),
                                 tags=("box", key))
            self._rects[key] = (r, t)
            self.tag_bind(key, "<Button-1>", lambda e, k=key: self._click(k))

            if i < len(FLOW_BLOCKS) - 1:
                ay = y + box_h
                self.create_line(x0 + box_w / 2, ay,
                                 x0 + box_w / 2, ay + gap,
                                 arrow=tk.LAST, fill="#607d8b", width=1.6)
            y += box_h + gap

        if self._selected:
            self.highlight(self._selected)

    def _click(self, key):
        self.highlight(key)
        if self.on_click:
            self.on_click(key)

    def highlight(self, key):
        self._selected = key
        for k, (r, t) in self._rects.items():
            self.itemconfigure(r, width=3 if k == key else 1.5)

    def refresh(self, status: dict):
        for key, (r, t) in self._rects.items():
            st = status.get(key, "idle")
            fill, outline, _ = _STATUS_COLOR[st]
            self.itemconfigure(r, fill=fill, outline=outline)
            if key == self._selected:
                self.itemconfigure(r, width=3)
