#!/usr/bin/env python3
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[1]
for p in R.rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
print('PASS: Python syntax')
