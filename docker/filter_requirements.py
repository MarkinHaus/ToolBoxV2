#!/usr/bin/env python3
"""
Filter requirements.txt based on active features.

Generates:
  - requirements-final.txt  — Python packages to install
  - active-features.txt     — space-separated list of active feature names
  - system-deps.txt         — Alpine apk packages needed at build time
"""
import os

# -- Read base requirements --------------------------------------------------
with open('requirements.txt', 'r') as f:
    lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

core_reqs = []
for line in lines:
    if '#' in line:
        parts = line.split('#', 1)
        if parts[0].strip():
            core_reqs.append(parts[0].strip())
    else:
        core_reqs.append(line)

# -- Feature definitions -----------------------------------------------------
# Each feature maps to:
#   extras  — additional pip packages
#   apk     — Alpine system packages needed to build/run them

FEATURES = {
    'cli': {
        'extras': [],
        'apk': [],
    },
    'web': {
        'extras': [
            'starlette',
            'uvicorn[standard]',
            'aiohttp-cors',
            'httpx',
            'waitress',
        ],
        'apk': [],
    },
    'desktop': {
        'extras': [],
        'apk': [
            'qt6-qtbase',  # PyQt6 runtime
        ],
    },
    'exotic': {
        'extras': [
            'scipy>=1.14.0',
            'matplotlib>=3.9.0',
            'pandas>=2.2.0',
        ],
        'apk': [
            'openblas-dev',   # scipy/numpy BLAS backend
            'freetype-dev',   # matplotlib
            'libpng-dev',     # matplotlib
        ],
    },
    'isaa': {
        'extras': [
            'litellm>=0.49.0',
            'langchain-core>=0.1.0',
            'groq>=0.11.0',
        ],
        'apk': [],
    },
}

ENV_MAP = {
    'cli': 'FEATURE_CLI',
    'web': 'FEATURE_WEB',
    'desktop': 'FEATURE_DESKTOP',
    'exotic': 'FEATURE_EXOTIC',
    'isaa': 'FEATURE_ISAA',
}

# -- Resolve active features -------------------------------------------------
active = []
all_extras = []
all_apk = set()

for feat, env_var in ENV_MAP.items():
    if os.environ.get(env_var, '0') == '1':
        active.append(feat)
        cfg = FEATURES[feat]
        all_extras.extend(cfg['extras'])
        all_apk.update(cfg['apk'])

# -- Write outputs -----------------------------------------------------------
with open('requirements-final.txt', 'w') as f:
    f.write('\n'.join(core_reqs))
    if all_extras:
        f.write('\n')
        f.write('\n'.join(all_extras))
    f.write('\n')

with open('active-features.txt', 'w') as f:
    f.write(' '.join(active) if active else 'none')

with open('system-deps.txt', 'w') as f:
    f.write(' '.join(sorted(all_apk)) if all_apk else '')

print(f'Active features: {active}')
print(f'Extra pip packages: {len(all_extras)}')
print(f'Extra system packages: {sorted(all_apk)}')
