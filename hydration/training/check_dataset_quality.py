"""
Dataset Quality Checker for Lip Hydration Images
Run this script to analyze your dataset for quality issues
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

class DatasetQualityChecker:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_images": 0,
            "issues_found": [],
            "statistics": {},
            "recommendations": []
        }
        
    def analyze_image(self, image_path):
        """Analyze a single image for quality issues"""
        img = cv2.imread(str(image_path))
        if img is None:
            return {"error": "Could not load image"}
        
        h, w = img.shape[:2]
        issues = []
        metrics = {}
        
        # 1. Check image dimensions and aspect ratio
        aspect_ratio = w / h
        metrics['width'] = w
        metrics['height'] = h
        metrics['aspect_ratio'] = round(aspect_ratio, 2)
        
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        # 2. Check for excessive background (using edge detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide image into regions
        center_h_start, center_h_end = h // 4, 3 * h // 4
        center_w_start, center_w_end = w // 4, 3 * w // 4
        
        center_region = edges[center_h_start:center_h_end, center_w_start:center_w_end]
        total_edges = np.sum(edges > 0)
        center_edges = np.sum(center_region > 0)
        
        if total_edges > 0:
            center_focus_ratio = center_edges / total_edges
            metrics['center_focus_ratio'] = round(center_focus_ratio, 3)
            
            # If less than 30% of edges are in center, likely too much background
            if center_focus_ratio < 0.3:
                issues.append(f"Too much background - only {center_focus_ratio*100:.1f}% focus in center")
        
        # 3. Check brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        metrics['brightness'] = round(brightness, 2)
        metrics['contrast'] = round(contrast, 2)
        
        if brightness < 50:
            issues.append(f"Image too dark (brightness: {brightness:.1f})")
        elif brightness > 200:
            issues.append(f"Image too bright (brightness: {brightness:.1f})")
            
        if contrast < 20:
            issues.append(f"Low contrast (contrast: {contrast:.1f})")
        
        # 4. Check for blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = round(laplacian_var, 2)
        
        if laplacian_var < 100:
            issues.append(f"Image appears blurry (sharpness: {laplacian_var:.1f})")
        
        # 5. Detect skin-like colors (lips should have significant skin tones)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        skin_percentage = (np.sum(skin_mask > 0) / (h * w)) * 100
        metrics['skin_percentage'] = round(skin_percentage, 2)
        
        if skin_percentage < 10:
            issues.append(f"Very low skin tone detected ({skin_percentage:.1f}%) - may not be a lip photo")
        
        # 6. Check file size
        file_size_kb = os.path.getsize(image_path) / 1024
        metrics['file_size_kb'] = round(file_size_kb, 2)
        
        if file_size_kb < 10:
            issues.append(f"Very small file size ({file_size_kb:.1f}KB) - may be low quality")
        
        return {
            "issues": issues,
            "metrics": metrics,
            "severity": "high" if len(issues) >= 3 else "medium" if len(issues) >= 1 else "low"
        }
    
    def check_dataset(self):
        """Check all images in the dataset"""
        categories = ['Dehydrate', 'Normal']
        
        for category in categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                print(f"Warning: {category} folder not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Analyzing {category} images...")
            print(f"{'='*60}")
            
            image_files = list(category_path.glob('*.jpg')) + \
                         list(category_path.glob('*.jpeg')) + \
                         list(category_path.glob('*.png'))
            
            category_issues = []
            
            for img_path in sorted(image_files):
                self.results['total_images'] += 1
                result = self.analyze_image(img_path)
                
                if 'error' in result:
                    print(f"ERROR {img_path.name}: {result['error']}")
                    continue
                
                if result['issues']:
                    issue_data = {
                        "file": str(img_path.relative_to(self.dataset_path)),
                        "category": category,
                        "issues": result['issues'],
                        "metrics": result['metrics'],
                        "severity": result['severity']
                    }
                    category_issues.append(issue_data)
                    self.results['issues_found'].append(issue_data)
                    
                    # Print issues
                    severity_icon = "HIGH" if result['severity'] == "high" else "MED"
                    print(f"\n[{severity_icon}] {img_path.name}")
                    for issue in result['issues']:
                        print(f"   - {issue}")
                    print(f"   Metrics: {result['metrics']}")
            
            print(f"\n{category} Summary: {len(category_issues)} images with issues out of {len(image_files)}")
        
        self._generate_recommendations()
        self._print_summary()
        
    def _generate_recommendations(self):
        """Generate recommendations based on found issues"""
        issue_types = {}
        
        for item in self.results['issues_found']:
            for issue in item['issues']:
                issue_type = issue.split('-')[0].strip()
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        self.results['statistics']['issue_types'] = issue_types
        
        # Generate recommendations
        recommendations = []
        
        if any('background' in issue.lower() for item in self.results['issues_found'] for issue in item['issues']):
            recommendations.append({
                "priority": "HIGH",
                "issue": "Background Focus Problem",
                "action": "Re-crop images to focus on lips only, remove excessive background",
                "affected_count": sum(1 for item in self.results['issues_found'] 
                                     if any('background' in i.lower() for i in item['issues']))
            })
        
        if any('dark' in issue.lower() or 'bright' in issue.lower() 
               for item in self.results['issues_found'] for issue in item['issues']):
            recommendations.append({
                "priority": "MEDIUM",
                "issue": "Brightness Issues",
                "action": "Adjust brightness/exposure or remove poorly lit images",
                "affected_count": sum(1 for item in self.results['issues_found'] 
                                     if any('dark' in i.lower() or 'bright' in i.lower() 
                                           for i in item['issues']))
            })
        
        if any('blur' in issue.lower() for item in self.results['issues_found'] for issue in item['issues']):
            recommendations.append({
                "priority": "HIGH",
                "issue": "Blurry Images",
                "action": "Remove blurry images or retake with better focus",
                "affected_count": sum(1 for item in self.results['issues_found'] 
                                     if any('blur' in i.lower() for i in item['issues']))
            })
        
        if any('skin' in issue.lower() for item in self.results['issues_found'] for issue in item['issues']):
            recommendations.append({
                "priority": "CRITICAL",
                "issue": "Non-lip Images",
                "action": "Remove images that don't contain lips or have wrong subject",
                "affected_count": sum(1 for item in self.results['issues_found'] 
                                     if any('skin' in i.lower() for i in item['issues']))
            })
        
        self.results['recommendations'] = recommendations
    
    def _print_summary(self):
        """Print summary of analysis"""
        print(f"\n{'='*60}")
        print("DATASET QUALITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Total images analyzed: {self.results['total_images']}")
        print(f"Images with issues: {len(self.results['issues_found'])}")
        print(f"Clean images: {self.results['total_images'] - len(self.results['issues_found'])}")
        
        if self.results['recommendations']:
            print(f"\n{'='*60}")
            print("RECOMMENDATIONS (Priority Order)")
            print(f"{'='*60}")
            for rec in sorted(self.results['recommendations'], 
                            key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}[x['priority']]):
                print(f"\n[{rec['priority']}] {rec['issue']}")
                print(f"   Affected: {rec['affected_count']} images")
                print(f"   Action: {rec['action']}")
        
        # Save detailed report
        report_path = self.dataset_path / 'quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Create list of problematic files
        if self.results['issues_found']:
            problem_files_path = self.dataset_path / 'problematic_files.txt'
            with open(problem_files_path, 'w') as f:
                f.write("PROBLEMATIC FILES - SORTED BY SEVERITY\n")
                f.write("="*60 + "\n\n")
                
                for severity in ['high', 'medium', 'low']:
                    files = [item for item in self.results['issues_found'] if item['severity'] == severity]
                    if files:
                        f.write(f"\n{severity.upper()} PRIORITY ({len(files)} files):\n")
                        f.write("-"*60 + "\n")
                        for item in files:
                            f.write(f"\n{item['file']}\n")
                            for issue in item['issues']:
                                f.write(f"  - {issue}\n")
            
            print(f"Problematic files list saved to: {problem_files_path}")

if __name__ == "__main__":
    # Set your dataset path - UPDATE THIS PATH
    DATASET_PATH = r"d:\PP2\Research_Project_225\IT22564818_Meyrushan_N\Model\Human_Body_Hydration_Managment_PP1\data"
    
    print("Starting Dataset Quality Check...")
    print(f"Dataset path: {DATASET_PATH}\n")
    
    checker = DatasetQualityChecker(DATASET_PATH)
    checker.check_dataset()
    
    print("\nAnalysis complete!")
