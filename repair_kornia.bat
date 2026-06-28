@echo off
cd /d f:\faceswap_pro_v2
venv\Scripts\python.exe -m pip uninstall -y kornia > repair_kornia.log 2>&1
venv\Scripts\python.exe -m pip install --no-cache-dir kornia >> repair_kornia.log 2>&1
venv\Scripts\python.exe -c "import kornia, os; fn=os.path.join(os.path.dirname(kornia.__file__), 'enhance', 'jpeg.py'); data=open(fn,'rb').read(); print(fn); print(os.path.exists(fn)); print(len(data)); print('has_nul', bytes([0]) in data)" >> repair_kornia.log 2>&1
