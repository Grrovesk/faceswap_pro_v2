"""
Examine the source image that's failing to detect a face.

This script:
1. Checks if the source image file exists and is valid
2. Displays image info (dimensions, size, format)
3. Attempts to detect faces using multiple methods
4. Shows why detection is failing
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def examine_image(image_path):
    """Load and examine an image file."""
    image_path = Path(image_path)
    
    print("=" * 80)
    print("SOURCE IMAGE DIAGNOSTIC")
    print("=" * 80)
    
    # Check file exists
    print(f"\n[1] Checking file...")
    if not image_path.exists():
        print(f"✗ File does not exist: {image_path}")
        return False
    
    file_size = image_path.stat().st_size
    print(f"✓ File exists: {image_path}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    
    if file_size == 0:
        print(f"✗ File is EMPTY (0 bytes)")
        return False
    
    # Load with OpenCV
    print(f"\n[2] Loading with OpenCV...")
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"✗ OpenCV failed to load image (file may be corrupted)")
            return False
        
        h, w, c = img.shape
        print(f"✓ Image loaded successfully")
        print(f"  Dimensions: {w} x {h} ({c} channels)")
        print(f"  Data type: {img.dtype}")
        print(f"  Min/Max pixel values: {img.min()} / {img.max()}")
        
        # Check if image is mostly blank
        if img.max() < 10:
            print(f"✗ WARNING: Image is mostly black (all pixels < 10)")
            return False
        
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False
    
    # Try face detection with InsightFace
    print(f"\n[3] Attempting face detection with InsightFace...")
    try:
        from insightface.app import FaceAnalysis
        
        PROJECT_ROOT = Path(__file__).resolve().parent
        FACE_ANALYSIS_ROOT = PROJECT_ROOT / "models" / "face_analysis"
        
        print(f"  Loading model from: {FACE_ANALYSIS_ROOT}")
        
        fa = FaceAnalysis(
            name="buffalo_l",
            root=str(FACE_ANALYSIS_ROOT),
            providers=["CPUExecutionProvider"],
        )
        fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
        
        faces = fa.get(img)
        print(f"✓ Detection completed")
        print(f"  Faces found: {len(faces)}")
        
        if faces:
            for i, face in enumerate(faces):
                bbox = face.bbox
                conf = getattr(face, 'det_score', None)
                print(f"    Face {i+1}: bbox={bbox}, confidence={conf:.3f if conf else 'N/A'}")
            return True
        else:
            print(f"✗ NO FACES DETECTED")
            print(f"\n  Possible reasons:")
            print(f"    1. Image contains no face")
            print(f"    2. Face is too small (<50px)")
            print(f"    3. Face is partially obscured")
            print(f"    4. Image is rotated (InsightFace expects upright)")
            return False
            
    except Exception as e:
        print(f"✗ InsightFace error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python examine_source_image.py <path_to_image>")
        print("\nExample:")
        print("  python examine_source_image.py \"F:\\faceswap_pro_v2\\tmp\\gradio\\...\\2026-06-23_13h11_43.png\"")
        
        # Try to find the most recent temp image
        print("\n\nSearching for recent temp images...")
        temp_dir = Path("tmp/gradio")
        if temp_dir.exists():
            images = list(temp_dir.glob("*/*.png")) + list(temp_dir.glob("*/*.jpg"))
            if images:
                latest = max(images, key=lambda p: p.stat().st_mtime)
                print(f"\nMost recent: {latest}")
                print(f"\nRunning diagnostic on: {latest}")
                return 0 if examine_image(latest) else 1
        
        return 1
    
    image_path = sys.argv[1]
    return 0 if examine_image(image_path) else 1

if __name__ == "__main__":
    sys.exit(main())
