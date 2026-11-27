# -*- coding: utf-8 -*-
"""
结果展示组件
============
MplFrame：把 matplotlib Figure 嵌入 tk 窗口，自带导航工具栏（缩放/平移/保存）。
DataTable：基于 ttk.Treeview 的只读数据表。
MetricCards：键值指标卡片网格。
"""

import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)


class MplFrame(tk.Frame):
    def __init__(self, parent, nrows=1, ncols=1, figsize=(7.5, 4.6), **kw):
        super().__init__(parent, bg="white", **kw)
        self.fig = Figure(figsize=figsize, dpi=100, tight_layout=True)
        if nrows == 1 and ncols == 1:
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = [self.fig.add_subplot(nrows, ncols, i + 1)
                         for i in range(nrows * ncols)]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        # 新版工具栏默认自行 pack，先取回再按“图上、栏下”重新布局
        self.toolbar.pack_forget()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def redraw(self):
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def clear(self):
        if isinstance(self.axes, list):
            for ax in self.axes:
                ax.clear()
        else:
            self.axes.clear()


class DataTable(tk.Frame):
    def __init__(self, parent, columns, headings, heights=12, **kw):
        super().__init__(parent, **kw)
        self.tree = ttk.Treeview(self, columns=columns, show="headings",
                                 height=heights)
        for c, h in zip(columns, headings):
            self.tree.heading(c, text=h)
            self.tree.column(c, anchor="center", width=110)
        vsb = ttk.Scrollbar(self, orient="vertical",
                            command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def fill(self, rows):
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert("", tk.END, values=row)


class MetricCards(tk.Frame):
    def __init__(self, parent, ncols=4, **kw):
        super().__init__(parent, bg="white", **kw)
        self.ncols = ncols
        self._labels = {}
        self._row = 0

    def add(self, key, title, value="--", color="#1565c0"):
        col = len(self._labels) % self.ncols
        if col == 0 and len(self._labels) > 0:
            self._row += 1
        box = tk.Frame(self, bg="#f5f7fa", bd=1, relief="solid")
        box.grid(row=self._row, column=col, sticky="ew", padx=6, pady=6)
        self.grid_columnconfigure(col, weight=1)
        tk.Label(box, text=title, bg="#f5f7fa", fg="#666",
                 font=("Microsoft YaHei", 9)).pack(padx=14, pady=(6, 0))
        v = tk.Label(box, text=value, bg="#f5f7fa", fg=color,
                     font=("Consolas", 13, "bold"))
        v.pack(padx=14, pady=(0, 6))
        self._labels[key] = v

    def set(self, key, value):
        if key in self._labels:
            self._labels[key].configure(text=value)
