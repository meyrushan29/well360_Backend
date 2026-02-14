"""
Auto-Crop Tool for Lip Images
This script automatically crops images to focus on the center (lips) and removes excessive background
"""

import cv2
import numpy as np
from pathlib import Path
import json
import shutil

def auto_crop_lip_image(image_path, output_path, crop_percentage=0.6):
    """
    Automatically crop image to focus on center (lips)
    
    Args:
        image_path: Path to input image
        output_path: Path to save cropped image
        crop_percentage: How much of the image to keep (0.6 = 60% of original, centered)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load {image_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Calculate crop dimensions (centered)
    new_w = int(w * crop_percentage)
    new_h = int(h * crop_percentage)
    
    # Calculate crop coordinates (center crop)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    end_x = start_x + new_w
    end_y = start_y + new_h
    
    # Crop the image
    cropped = img[start_y:end_y, start_x:end_x]
    
    # Save the cropped image
    cv2.imwrite(str(output_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True

def process_problematic_images(dataset_path, quality_report_path):
    """Process all images with background focus issues"""
    
    dataset_path = Path(dataset_path)
    
    # Load quality report
    with open(quality_report_path, 'r') as f:
        report = json.load(f)
    
    # Create backup directory
    backup_dir = dataset_path / 'original_backup'
    backup_dir.mkdir(exist_ok=True)
    
    # Create fixed directory
    fixed_dir = dataset_path / 'fixed_images'
    fixed_dir.mkdir(exist_ok=True)
    
    background_issues = []
    for issue in report['issues_found']:
        if any('background' in i.lower() for i in issue['issues']):
            background_issues.append(issue)
    
    print(f"\n{'='*60}")
    print(f"AUTO-CROP TOOL FOR LIP IMAGES")
    print(f"{'='*60}")
    print(f"Found {len(background_issues)} images with background focus issues")
    print(f"\nThis tool will:")
    print(f"  1. Backup original images to: {backup_dir}")
    print(f"  2. Create cropped versions in: {fixed_dir}")
    print(f"  3. You can review and replace originals manually")
    print(f"\n{'='*60}\n")
    
    proceed = input("Proceed with auto-cropping? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Cancelled.")
        return
    
    fixed_count = 0
    for issue in background_issues:
        file_path = dataset_path / issue['file']
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            continue
        
        # Determine crop percentage based on severity
        center_focus = issue['metrics'].get('center_focus_ratio', 0)
        if center_focus < 0.1:
            crop_pct = 0.5  # Aggressive crop for very bad images
        elif center_focus < 0.2:
            crop_pct = 0.6
        else:
            crop_pct = 0.7
        
        # Create backup
        backup_path = backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        
        # Create fixed version
        fixed_path = fixed_dir / file_path.name
        success = auto_crop_lip_image(file_path, fixed_path, crop_pct)
        
        if success:
            fixed_count += 1
            print(f"✓ Fixed: {file_path.name} (focus: {center_focus:.1%} → cropped {crop_pct:.0%})")
        else:
            print(f"✗ Failed: {file_path.name}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully cropped: {fixed_count} images")
    print(f"Originals backed up to: {backup_dir}")
    print(f"Fixed images saved to: {fixed_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review the fixed images in {fixed_dir}")
    print(f"  2. If satisfied, copy them back to replace originals")
    print(f"  3. Run quality checker again to verify improvements")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    DATASET_PATH = r"d:\PP2\Research_Project_225\IT22564818_Meyrushan_N\Model\Human_Body_Hydration_Managment_PP1\data"
    QUALITY_REPORT = r"d:\PP2\Research_Project_225\IT22564818_Meyrushan_N\Model\Human_Body_Hydration_Managment_PP1\data\quality_report.json"
    
    process_problematic_images(DATASET_PATH, QUALITY_REPORT)
