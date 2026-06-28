"""
Comprehensive diagnostic and fix for onnxruntime initialization failure.

Checks:
1. Current package versions
2. Import errors with full traceback
3. Automatically attempts fixes
"""
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, capture=False):
    """Run command, optionally capture output."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode, "", ""
    except Exception as e:
        return -1, "", str(e)

def main():
    print("=" * 80)
    print("COMPREHENSIVE ONNXRUNTIME DIAGNOSTIC & FIX")
    print("=" * 80)
    
    # Phase 1: Diagnose current state
    print("\n[PHASE 1] Checking current package versions...")
    rc, out, err = run_cmd("pip show onnxruntime onnxruntime-gpu onnx", capture=True)
    print(out if out else "(no packages found)")
    
    # Phase 2: Try to import and identify exact error
    print("\n[PHASE 2] Attempting imports to identify exact error...")
    try:
        import onnxruntime
        print(f"✓ onnxruntime imported: {onnxruntime.__version__}")
        print(f"✓ Has InferenceSession: {hasattr(onnxruntime, 'InferenceSession')}")
    except Exception as e:
        print(f"✗ onnxruntime import failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        import onnx
        print(f"✓ onnx imported: {onnx.__version__}")
    except Exception as e:
        print(f"✗ onnx import failed: {e}")
    
    try:
        from insightface.app import FaceAnalysis
        print(f"✓ insightface imported")
    except Exception as e:
        print(f"✗ insightface import failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 3: Attempt automatic fix
    print("\n[PHASE 3] Attempting automatic fix...")
    print("\nRemoving conflicting packages...")
    run_cmd("pip uninstall -y onnxruntime onnxruntime-gpu onnx", capture=False)
    
    print("\nInstalling onnxruntime-gpu==1.16.3 (includes compatible onnx)...")
    rc, out, err = run_cmd("pip install --no-cache-dir onnxruntime-gpu==1.16.3", capture=True)
    if rc != 0:
        print(f"Installation output:\n{out}")
        if err:
            print(f"Errors:\n{err}")
    else:
        print("✓ Installation complete")
    
    # Phase 4: Verify fix
    print("\n[PHASE 4] Verifying fix...")
    try:
        # Force reimport
        import importlib
        if 'onnxruntime' in sys.modules:
            del sys.modules['onnxruntime']
        if 'onnx' in sys.modules:
            del sys.modules['onnx']
        if 'insightface' in sys.modules:
            del sys.modules['insightface']
        
        import onnxruntime
        print(f"✓ onnxruntime: {onnxruntime.__version__}")
        
        import onnx
        print(f"✓ onnx: {onnx.__version__}")
        
        # Test InsightFace can load
        print("\nTesting InsightFace initialization...")
        from insightface.app import FaceAnalysis
        
        PROJECT_ROOT = Path(__file__).resolve().parent
        FACE_ANALYSIS_ROOT = PROJECT_ROOT / "models" / "face_analysis"
        
        # This will attempt to load or download the model
        fa = FaceAnalysis(
            name="buffalo_l",
            root=str(FACE_ANALYSIS_ROOT),
            providers=[
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
        )
        fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        print("✓ InsightFace FaceAnalysis initialized successfully!")
        
        print("\n" + "=" * 80)
        print("✓✓✓ SUCCESS! onnxruntime is now fixed and working.")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Restart your Gradio app (Ctrl+C then python launch.py)")
        print("  2. Try video swap or webcam swap again")
        print("  3. Face detection should now work")
        return 0
        
    except Exception as e:
        print(f"\n✗ Fix incomplete. Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("MANUAL FIX REQUIRED")
        print("=" * 80)
        print("\nTry these steps manually:")
        print("  1. pip uninstall -y onnxruntime onnxruntime-gpu onnx")
        print("  2. pip install --no-cache-dir onnxruntime-gpu==1.16.3")
        print("  3. Restart Python/Gradio")
        print("\nIf still failing, try CPU-only:")
        print("  1. pip uninstall -y onnxruntime onnxruntime-gpu")
        print("  2. pip install onnxruntime==1.16.3")
        return 1

if __name__ == "__main__":
    sys.exit(main())
