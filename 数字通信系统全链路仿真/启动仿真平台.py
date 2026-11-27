# -*- coding: utf-8 -*-
"""
数字通信系统全链路仿真平台 —— 启动入口
======================================
双击“启动仿真平台.bat”或在本目录执行：
    python 启动仿真平台.py
"""

import os
import sys

# 保证从任意工作目录启动都能正确导入包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comm_platform.ui.app import launch  # noqa: E402


if __name__ == "__main__":
    launch()
