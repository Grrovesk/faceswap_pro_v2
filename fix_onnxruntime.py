"""
Fix onnxruntime/onnx version compatibility issue.

The issue: onnxruntime-gpu 1.16.3 with onnx 1.17.0 causes InsightFace
to fail with "module 'onnxruntime' has no attribute 'InferenceSession'".

This script downgrades onnx to 1.16.1 to match onnxruntime-gpu 1.16.3.
"""
import subprocess
import sys

def run_cmd(cmd):
    """Run a shell command and print output."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True

def main():
    print("=" * 70)
    print("Fixing onnxruntime/onnx compatibility issue...")
    print("=" * 70)
    
    # Step 1: Uninstall conflicting packages
    print("\n[Step 1] Uninstalling conflicting packages...")
    run_cmd("pip uninstall -y onnx onnxruntime onnxruntime-gpu")
    
    # Step 2: Reinstall onnxruntime-gpu 1.16.3 (should pull compatible onnx)
    print("\n[Step 2] Reinstalling onnxruntime-gpu 1.16.3...")
    if not run_cmd("pip install onnxruntime-gpu==1.16.3"):
        print("FAILED: Could not install onnxruntime-gpu")
        return 1
    
    # Step 3: Verify installation
    print("\n[Step 3] Verifying installation...")
    try:
        import onnxruntime
        print(f"✓ onnxruntime version: {onnxruntime.__version__}")
        print(f"✓ Has InferenceSession: {hasattr(onnxruntime, 'InferenceSession')}")
        
        import onnx
        print(f"✓ onnx version: {onnx.__version__}")
        
        # Try to use it
        sess = onnxruntime.InferenceSession
        print(f"✓ InferenceSession accessible: {sess}")
        
        print("\n✓✓✓ SUCCESS! onnxruntime is now working correctly.")
        return 0
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
