# -*- coding: utf-8 -*-
"""
参数控件工厂
============
统一生成下拉框 / 数值输入框 / 勾选框，并把每个控件绑定到一个字符串 key。
控件值变化时通过 on_change(key, value) 通知主窗口；主窗口负责写回配置、
失效下游模块并刷新流程图。
"""

import tkinter as tk
from tkinter import ttk


class ParamForm(tk.Frame):
    def __init__(self, parent, on_change=None, **kw):
        super().__init__(parent, bg="white", **kw)
        self._on_change = on_change
        self._vars: dict[str, tk.Variable] = {}
        self._widgets: dict[str, ttk.Widget] = {}
        self._types: dict[str, type] = {}
        self._row = 0
        self._silent = False
        self.grid_columnconfigure(1, weight=1)

    # ------------------------- 控件构造 ------------------------- #
    def title(self, text):
        lbl = tk.Label(self, text=text, bg="white",
                       fg="#1565c0", font=("Microsoft YaHei", 10, "bold"))
        lbl.grid(row=self._row, column=0, columnspan=2,
                 sticky="w", pady=(8, 2))
        self._row += 1

    def add_combo(self, key, label, choices, value):
        self._label(key, label)
        var = tk.StringVar(value=str(value))
        cb = ttk.Combobox(self, textvariable=var, state="readonly",
                          values=list(choices), width=18)
        cb.grid(row=self._row, column=1, sticky="ew", padx=(4, 8), pady=2)
        var.trace_add("write", lambda *_: self._emit(key))
        self._store(key, var, cb, str)
        self._row += 1
        return cb

    def add_entry(self, key, label, value, vtype):
        self._label(key, label)
        var = tk.StringVar(value=str(value))
        ent = ttk.Entry(self, textvariable=var, width=20)
        ent.grid(row=self._row, column=1, sticky="ew", padx=(4, 8), pady=2)
        var.trace_add("write", lambda *_: self._emit(key))
        self._store(key, var, ent, vtype)
        self._row += 1
        return ent

    def add_check(self, key, label, value):
        var = tk.BooleanVar(value=bool(value))
        chk = ttk.Checkbutton(self, text=label, variable=var)
        chk.grid(row=self._row, column=0, columnspan=2,
                 sticky="w", padx=(4, 8), pady=2)
        var.trace_add("write", lambda *_: self._emit(key))
        self._store(key, var, chk, bool)
        self._row += 1
        return chk

    def _label(self, key, text):
        lbl = tk.Label(self, text=text, bg="white",
                       font=("Microsoft YaHei", 9))
        lbl.grid(row=self._row, column=0, sticky="w", padx=(8, 4), pady=2)

    def _store(self, key, var, widget, vtype):
        self._vars[key] = var
        self._widgets[key] = widget
        self._types[key] = vtype

    # ------------------------- 读写 ------------------------- #
    def _emit(self, key):
        if self._silent:
            return
        value = self.get(key)
        if self._on_change and value is not None:
            self._on_change(key, value)

    def get(self, key):
        vtype = self._types[key]
        raw = self._vars[key].get()
        if vtype is bool:
            return raw
        try:
            return vtype(raw)
        except (ValueError, TypeError):
            return None

    def set(self, key, value, silent=True):
        old = self._silent
        self._silent = silent
        self._vars[key].set(value)
        self._silent = old

    def set_enabled(self, key, enabled):
        state = "normal" if enabled else "disabled"
        self._widgets[key].configure(state=state)
