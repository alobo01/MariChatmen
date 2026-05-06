"""Project-local interpreter tweaks.

Pytest on this workstation sees globally installed ROS pytest entry points from
outside the virtualenv. Disable auto-loading only for pytest invocations so this
repo's tests remain isolated from host plugins.
"""

from __future__ import annotations

import os
import sys

if any("pytest" in arg for arg in sys.argv):
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
