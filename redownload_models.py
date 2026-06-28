"""
Redownload face detection models from InsightFace.

This script:
1. Clears cached detection models (buffalo_l, antelopev2)
2. Redownloads fresh models from HuggingFace
3. Verifies the models load correctly
"""
import shutil
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent
    FACE_ANALYSIS_ROOT = PROJECT_ROOT / "models" / "face_analysis"
    
    print("=" * 70)
    print("Redownloading Face Detection Models")
    print("=" * 70)
    
    # Step 1: Clear existing models
    print("\n[Step 1] Clearing cached models...")
    models_to_clear = ["buffalo_l", "antelopev2", "buffalo"]
    for model_name in models_to_clear:
        model_dir = FACE_ANALYSIS_ROOT / model_name
        if model_dir.exists():
            logger.info(f"Removing {model_dir}")
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            logger.info(f"Not found (ok): {model_dir}")
    
    # Step 2: Download buffalo_l (primary model)
    print("\n[Step 2] Downloading buffalo_l detection model...")
    print("This may take 5-10 minutes (~500 MB)...")
    try:
        from insightface.app import FaceAnalysis
        
        fa = FaceAnalysis(
            name="buffalo_l",
            root=str(FACE_ANALYSIS_ROOT),
            providers=[
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
        )
        fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        logger.info("✓ buffalo_l model downloaded and verified")
    except Exception as e:
        logger.error(f"✗ Failed to download buffalo_l: {e}")
        return 1
    
    # Step 3: Download antelopev2 (fallback model, optional)
    print("\n[Step 3] Downloading antelopev2 fallback model (optional)...")
    print("This may take 3-5 minutes (~200 MB)...")
    try:
        fa_v2 = FaceAnalysis(
            name="antelopev2",
            root=str(FACE_ANALYSIS_ROOT),
            providers=[
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
        )
        fa_v2.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        logger.info("✓ antelopev2 model downloaded and verified")
    except Exception as e:
        logger.warning(f"⚠ Failed to download antelopev2 (optional): {e}")
        # Don't fail, this is just fallback
    
    # Step 4: List downloaded models
    print("\n[Step 4] Verifying downloaded models...")
    if FACE_ANALYSIS_ROOT.exists():
        logger.info(f"Models directory: {FACE_ANALYSIS_ROOT}")
        for item in FACE_ANALYSIS_ROOT.iterdir():
            if item.is_dir():
                size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024 * 1024)
                logger.info(f"  ✓ {item.name} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 70)
    print("✓✓✓ Face detection models redownloaded successfully!")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Restart the Gradio app")
    print("  2. Try webcam or video swap again")
    print("  3. InsightFace should now detect faces correctly")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
