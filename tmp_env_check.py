import os
import sys
from pathlib import Path
print('sys.version', sys.version)
for pkg in ['torch', 'torchvision', 'torchaudio', 'diffusers', 'transformers', 'kornia']:
    try:
        m = __import__(pkg)
        print(pkg, getattr(m, '__version__', '<no version>'), getattr(m, '__file__', '<no file>'))
    except Exception as e:
        print(pkg, 'IMPORT_FAILED', type(e).__name__, e)
root = Path(r'f:\faceswap_pro_v2\venv\Lib\site-packages')
for rel in ['torchvision', 'torchaudio', 'kornia']:
    p = root / rel
    print('dir', rel, p.exists())
print('torchaudio lib exists', (root / 'torchaudio' / 'lib' / 'libtorchaudio.pyd').exists())
