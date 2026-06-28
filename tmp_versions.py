import importlib
pkgs = ['torch', 'torchvision', 'torchaudio', 'diffusers', 'transformers']
for pkg in pkgs:
    try:
        m = importlib.import_module(pkg)
        print(pkg, getattr(m, '__version__', 'no-version'), getattr(m, '__file__', 'no-file'))
    except Exception as e:
        print(pkg, 'IMPORT_FAILED', type(e).__name__, e)
