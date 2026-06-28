"""
Quick diagnostic: check if onnxruntime is actually broken.
"""
import sys

print("=" * 70)
print("ONNXRUNTIME DIAGNOSTIC")
print("=" * 70)

# Step 1: Check installed versions
print("\n[1] Checking installed packages...")
import subprocess
result = subprocess.run("pip show onnxruntime onnxruntime-gpu onnx", shell=True, capture_output=True, text=True)
print(result.stdout if result.stdout else "No onnxruntime packages found")

# Step 2: Try to import onnxruntime
print("\n[2] Attempting to import onnxruntime...")
try:
    import onnxruntime as ort
    print(f"✓ Import successful: {ort.__version__}")
    print(f"  Has InferenceSession: {hasattr(ort, 'InferenceSession')}")
    print(f"  Available providers: {ort.get_available_providers()}")
except Exception as e:
    print(f"✗ Import FAILED: {e}")
    sys.exit(1)

# Step 3: Try to use InferenceSession
print("\n[3] Testing InferenceSession access...")
try:
    sess_cls = ort.InferenceSession
    print(f"✓ InferenceSession accessible: {sess_cls}")
except Exception as e:
    print(f"✗ InferenceSession NOT accessible: {e}")
    sys.exit(1)

# Step 4: Try to use InsightFace
print("\n[4] Testing InsightFace FaceAnalysis...")
try:
    from insightface.app import FaceAnalysis
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).resolve().parent
    FACE_ANALYSIS_ROOT = PROJECT_ROOT / "models" / "face_analysis"
    
    print(f"  Loading from: {FACE_ANALYSIS_ROOT}")
    print("  This may take 30 seconds on first run...")
    
    fa = FaceAnalysis(
        name="buffalo_l",
        root=str(FACE_ANALYSIS_ROOT),
        providers=["CPUExecutionProvider"],  # Force CPU to avoid GPU issues
    )
    fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)  # ctx_id=-1 = CPU
    print(f"✓ InsightFace initialized successfully!")
    
    # Test detection on a dummy image
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = fa.get(dummy)
    print(f"✓ Detection works (found {len(faces)} faces in blank image)")
    
except Exception as e:
    print(f"✗ InsightFace FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓✓✓ ONNXRUNTIME IS WORKING!")
print("=" * 70)
