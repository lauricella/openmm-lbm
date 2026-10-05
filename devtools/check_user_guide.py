# --------------------------------------------------------------------------
# openmm-lbm
# Copyright (c) 2026 the Authors (see README.md).
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Run the Python examples of the user guide and check their outputs.

Every ```python block of docs/user_guide/*.md is run in its own process, in a temporary directory, with
the installed openmm and openmmlbm modules. When a block is followed by an "Output:" block, the standard
output must match it exactly. The blocks of api_reference.md are fragments: they run after a prelude
that defines system, force and context. A block that starts with simulation.loadCheckpoint continues
the block before it.

Usage: python devtools/check_user_guide.py
"""

import os
import re
import subprocess
import sys
import tempfile

GUIDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'user_guide')

PRELUDE = '''import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmlbm import LBMForce
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
system.addParticle(100.0)
force = LBMForce()
force.setGridSize(8, 8, 8)
system.addForce(force)
context = mm.Context(system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName('Reference'))
context.setPositions([mm.Vec3(1, 1, 1)])
'''


def main():
    failures = 0
    for name in sorted(os.listdir(GUIDE)):
        if not name.endswith('.md'):
            continue
        text = open(os.path.join(GUIDE, name)).read()
        previous = ''
        for number, match in enumerate(re.finditer(r'```python\n(.*?)```', text, re.S), 1):
            code = match.group(1)
            output = re.match(r'\s*Output:\s*\n\s*```\n(.*?)```', text[match.end():], re.S)
            expected = output.group(1) if output else None
            if name == 'api_reference.md':
                script = PRELUDE + code
            elif code.startswith('simulation.loadCheckpoint'):
                script = previous + code
            else:
                script = code
            previous = code
            with tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, '-c', script], cwd=directory, capture_output=True, text=True)
            ok = result.returncode == 0 and (expected is None or result.stdout == expected)
            print('%-20s block %2d  %s%s' % (name, number, 'ok' if ok else 'FAILED',
                                             ', output checked' if expected is not None else ''))
            if not ok:
                failures += 1
                print(result.stdout + result.stderr)
                if expected is not None:
                    print('expected output:\n' + expected)
    print('%d failed' % failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
