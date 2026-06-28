"""
Direct onnxruntime fix with step-by-step instructions.
Run this if face detection is failing.
"""
import subprocess
import sys
import time

def run_cmd(cmd, timeout=300):
    """Run a command and return success/failure."""
    print(f"\n>>> {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("=" * 80)
    print("ONNXRUNTIME FIX - STEP BY STEP")
    print("=" * 80)
    
    # Step 1: Check current state
    print("\n[STEP 1] Checking current package state...")
    run_cmd("pip show onnxruntime onnxruntime-gpu onnx", timeout=30)
    
    # Step 2: Uninstall everything
    print("\n[STEP 2] Uninstalling conflicting packages...")
    print("(This may take 1-2 minutes)")
    if not run_cmd("pip uninstall -y onnxruntime onnxruntime-gpu onnx 2>nul", timeout=120):
        print("WARNING: Uninstall had issues, continuing anyway...")
    
    time.sleep(2)
    
    # Step 3: Clear pip cache
    print("\n[STEP 3] Clearing pip cache...")
    run_cmd("pip cache purge", timeout=60)
    
    time.sleep(2)
    
    # Step 4: Reinstall onnxruntime-gpu
    print("\n[STEP 4] Installing onnxruntime-gpu 1.16.3...")
    print("(This will download ~200 MB and take 2-5 minutes)")
    if not run_cmd("pip install --no-cache-dir onnxruntime-gpu==1.16.3", timeout=600):
        print("ERROR: onnxruntime-gpu installation failed!")
        print("\nTrying CPU-only fallback...")
        if not run_cmd("pip install --no-cache-dir onnxruntime==1.16.3", timeout=600):
            print("ERROR: CPU-only installation also failed!")
            return 1
        print("⚠ Installed CPU-only version (slower, but will work)")
    
    time.sleep(2)
    
    # Step 5: Verify
    print("\n[STEP 5] Verifying installation...")
    try:
        import importlib
        # Clear modules
        for mod in list(sys.modules.keys()):
            if 'onnx' in mod.lower() or 'insightface' in mod.lower():
                del sys.modules[mod]
        
        import onnxruntime as ort
        print(f"✓ onnxruntime: {ort.__version__}")
        print(f"✓ Available providers: {ort.get_available_providers()}")
        
        from insightface.app import FaceAnalysis
        print(f"✓ InsightFace: imported successfully")
        
        print("\n" + "=" * 80)
        print("✓✓✓ FIX COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Restart the Gradio app (Ctrl+C, then: python launch.py)")
        print("  2. Try video/webcam swap again")
        print("  3. Face detection should now work")
        return 0
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("MANUAL FIX REQUIRED")
        print("=" * 80)
        print("\nIn PowerShell, run these commands one by one:")
        print("\n1. pip uninstall -y onnxruntime onnxruntime-gpu onnx")
        print("2. pip cache purge")
        print("3. pip install --no-cache-dir onnxruntime-gpu==1.16.3")
        print("4. python launch.py")
        print("\nIf step 3 still fails, use CPU-only:")
        print("   pip install --no-cache-dir onnxruntime==1.16.3")
        return 1

if __name__ == "__main__":
    sys.exit(main())
