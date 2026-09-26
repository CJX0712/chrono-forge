"""pytest 根路径注入，保证 `import chrono_forge` 可用。

Author: 晨星
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
