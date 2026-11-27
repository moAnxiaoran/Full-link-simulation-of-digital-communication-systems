# -*- coding: utf-8 -*-
"""
Digital Communication Full-Link Simulation Platform - launcher
Double-click run_platform.bat, or run:  python run_platform.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comm_platform.ui.app import launch  # noqa: E402

if __name__ == "__main__":
    launch()
