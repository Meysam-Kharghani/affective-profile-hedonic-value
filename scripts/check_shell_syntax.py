#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
files=list(ROOT.rglob("*.sh")); bad=[]
for p in files:
    cp=subprocess.run(["bash","-n",str(p)],capture_output=True,text=True)
    if cp.returncode: bad.append((p.relative_to(ROOT).as_posix(),cp.stderr.strip()))
print(f"Checked bash syntax for {len(files)} shell scripts.")
if bad:
    for p,e in bad: print("FAIL",p,e)
    sys.exit(1)
print("PASS: shell syntax check.")
