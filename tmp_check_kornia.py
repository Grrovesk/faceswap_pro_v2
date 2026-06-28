import os
from pathlib import Path
root = Path(r'f:\faceswap_pro_v2\venv\Lib\site-packages')
print('root', root.exists())
print('kornia dir', (root / 'kornia').exists())
print('kornia init', (root / 'kornia' / '__init__.py').exists())
for rel in ['kornia/enhance/jpeg.py', 'kornia/filters/__init__.py']:
    p = root / rel
    print(rel, p.exists(), p.stat().st_size if p.exists() else 'missing')
    if p.exists():
        data = p.read_bytes()
        print('  nul_count=', data.count(b'\x00'))
        print('  prefix=', data[:64])
