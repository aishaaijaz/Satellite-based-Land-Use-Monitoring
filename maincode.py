import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

# ========================================
# DATASET LOADER
# ========================================

class OSCDDatasetLoader:
    def __init__(self, images_root, labels_root=None):
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root) if labels_root else None
        self.regions = sorted([d.name for d in self.images_root.iterdir() if d.is_dir()])
        
        self.regions_with_gt = []
        self.regions_without_gt = []
        
        for region in self.regions:
            if self.has_ground_truth(region):
                self.regions_with_gt.append(region)
            else:
                self.regions_without_gt.append(region)
        
        print(f"Found {len(self.regions)} total regions")
        print(f"   With GT: {len(self.regions_with_gt)}")
        print(f"   Without GT: {len(self.regions_without_gt)}")
    
    def has_ground_truth(self, region_name):
        if not self.labels_root:
            return False
        gt_path = self.labels_root / region_name / "cm"
        if gt_path.exists():
            return len(list(gt_path.glob("*.png"))) > 0
        return False
    
    def load_region(self, region_name):
        region_path = self.images_root / region_name
        pair_folder = region_path / "pair"
        pair_files = sorted(pair_folder.glob("*.png"))
        
        if len(pair_files) < 2:
            raise FileNotFoundError(f"No images for {region_name}")
        
        before = cv2.imread(str(pair_files[0]))
        after = cv2.imread(str(pair_files[1]))
        before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
        
        gt_mask = None
        if self.has_ground_truth(region_name):
            gt_path = self.labels_root / region_name / "cm"
            gt_files = sorted(gt_path.glob("*.png"))
            if gt_files:
                gt_mask = cv2.imread(str(gt_files[0]), cv2.IMREAD_GRAYSCALE)
        
        return before, after, gt_mask


IMAGES_ROOT = r"E:\Year 3\IMG\proj\images\Onera Satellite Change Detection dataset - Images"
LABELS_ROOT = r"E:\Year 3\IMG\proj\train_labels\Onera Satellite Change Detection dataset - Train Labels"

dataset = OSCDDatasetLoader(IMAGES_ROOT, LABELS_ROOT)

# ========================================
# CHANGE DETECTOR
# ========================================

class ProperlyTunedChangeDetector:
    """Change detection with balanced sensitivity modes"""
    
    def __init__(self, sensitivity='balanced'):
        self.sensitivity = sensitivity
        
        if sensitivity == 'conservative':
            self.threshold_offset = 20
            self.min_area = 150
            self.morph_kernel = 7
            self.morph_iterations = 3
            self.voting_threshold = 3
        elif sensitivity == 'balanced':
            self.threshold_offset = 5
            self.min_area = 70
            self.morph_kernel = 5
            self.morph_iterations = 2
            self.voting_threshold = 2
        elif sensitivity == 'sensitive':
            self.threshold_offset = -3
            self.min_area = 45
            self.morph_kernel = 4
            self.morph_iterations = 1
            self.voting_threshold = 2
        else:  # moderate
            self.threshold_offset = 12
            self.min_area = 100
            self.morph_kernel = 6
            self.morph_iterations = 2
            self.voting_threshold = 2
    
    def detect(self, before, after):
        mask1 = self._intensity_change(before, after)
        mask2 = self._color_change(before, after)
        mask3 = self._texture_change(before, after)
        combined = ((mask1.astype(int) + mask2.astype(int) + mask3.astype(int)) 
                   >= self.voting_threshold).astype(np.uint8) * 255
        final = self._postprocess(combined)
        return final
    
    def _intensity_change(self, before, after):
        gray1 = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        thresh_val, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_thresh = max(15, min(100, thresh_val + self.threshold_offset))
        _, mask = cv2.threshold(diff, adjusted_thresh, 255, cv2.THRESH_BINARY)
        return mask
    
    def _color_change(self, before, after):
        b1 = before.astype(np.float32)
        b2 = after.astype(np.float32)
        diff = np.linalg.norm(b1 - b2, axis=2)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh_val, _ = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_thresh = max(15, min(100, thresh_val + self.threshold_offset))
        _, mask = cv2.threshold(diff_norm, adjusted_thresh, 255, cv2.THRESH_BINARY)
        return mask
    
    def _texture_change(self, before, after):
        gray1 = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        grad_x1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
        grad_y1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
        grad_x2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
        grad_y2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
        grad_diff = np.abs(grad_mag1 - grad_mag2)
        grad_diff_norm = cv2.normalize(grad_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thresh_val, _ = cv2.threshold(grad_diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_thresh = max(15, min(100, thresh_val + self.threshold_offset))
        _, mask = cv2.threshold(grad_diff_norm, adjusted_thresh, 255, cv2.THRESH_BINARY)
        return mask
    
    def _postprocess(self, mask):
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                cleaned[labels == i] = 255
        return cleaned

# ========================================
# METRICS
# ========================================

def calculate_metrics(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    pred_bin = (pred_mask > 127).astype(np.uint8).flatten()
    gt_bin = (gt_mask > 127).astype(np.uint8).flatten()
    return {
        'iou': jaccard_score(gt_bin, pred_bin, zero_division=0),
        'precision': precision_score(gt_bin, pred_bin, zero_division=0),
        'recall': recall_score(gt_bin, pred_bin, zero_division=0),
        'f1': f1_score(gt_bin, pred_bin, zero_division=0)
    }

# ========================================
# BASIC VISUALIZATION
# ========================================

def visualize_results(before, after, pred_mask, gt_mask, region_name, metrics=None, method_name=""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].imshow(before)
    axes[0,0].set_title("BEFORE", fontsize=14, weight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(after)
    axes[0,1].set_title("AFTER", fontsize=14, weight='bold')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(pred_mask, cmap='gray')
    change_pct = (np.sum(pred_mask > 0) / pred_mask.size) * 100
    axes[0,2].set_title(f"Predicted ({change_pct:.2f}%)", fontsize=14, weight='bold')
    axes[0,2].axis('off')
    
    if gt_mask is not None:
        axes[1,0].imshow(gt_mask, cmap='gray')
        gt_pct = (np.sum(gt_mask > 0) / gt_mask.size) * 100
        axes[1,0].set_title(f"Ground Truth ({gt_pct:.2f}%)", fontsize=14, weight='bold')
    else:
        axes[1,0].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=14)
        axes[1,0].set_title("Ground Truth", fontsize=14, weight='bold')
    axes[1,0].axis('off')
    
    overlay = after.copy()
    pred_resized = cv2.resize(pred_mask, (after.shape[1], after.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    overlay[pred_resized > 127] = [255, 0, 0]
    blended = cv2.addWeighted(after, 0.7, overlay, 0.3, 0)
    axes[1,1].imshow(blended)
    axes[1,1].set_title("Overlay", fontsize=14, weight='bold')
    axes[1,1].axis('off')
    
    if metrics:
        text = f"Method: {method_name}\n\n"
        text += "\n".join([f"{k.upper():10s}: {v:.4f}" for k, v in metrics.items()])
        axes[1,2].text(0.1, 0.2, text, fontsize=12, family='monospace')
    else:
        axes[1,2].text(0.5, 0.5, 'No Metrics', ha='center', va='center')
    axes[1,2].set_title("Performance", fontsize=14, weight='bold')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{region_name}_{method_name}_basic.png', dpi=150, bbox_inches='tight')
    plt.show()

# ========================================
# ENHANCED VISUALIZATIONS
# ========================================

def visualize_detection_pipeline(before, after, gt_mask, region_name, sensitivity='balanced'):
    """Show complete detection pipeline step-by-step"""
    detector = ProperlyTunedChangeDetector(sensitivity=sensitivity)
    
    mask1 = detector._intensity_change(before, after)
    mask2 = detector._color_change(before, after)
    mask3 = detector._texture_change(before, after)
    combined_raw = ((mask1.astype(int) + mask2.astype(int) + mask3.astype(int)) 
                   >= detector.voting_threshold).astype(np.uint8) * 255
    final = detector._postprocess(combined_raw)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Input
    axes[0,0].imshow(before)
    axes[0,0].set_title("BEFORE", fontsize=12, weight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(after)
    axes[0,1].set_title("AFTER", fontsize=12, weight='bold')
    axes[0,1].axis('off')
    
    if gt_mask is not None:
        axes[0,2].imshow(gt_mask, cmap='gray')
        gt_pct = (np.sum(gt_mask > 0) / gt_mask.size) * 100
        axes[0,2].set_title(f"Ground Truth ({gt_pct:.2f}%)", fontsize=12, weight='bold')
    else:
        axes[0,2].text(0.5, 0.5, 'No GT', ha='center', va='center')
        axes[0,2].set_title("Ground Truth", fontsize=12, weight='bold')
    axes[0,2].axis('off')
    
    diff_gray = cv2.absdiff(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY),
                           cv2.cvtColor(after, cv2.COLOR_RGB2GRAY))
    axes[0,3].imshow(diff_gray, cmap='hot')
    axes[0,3].set_title("Raw Difference", fontsize=12, weight='bold')
    axes[0,3].axis('off')
    
    # Row 2: Methods
    axes[1,0].imshow(mask1, cmap='gray')
    axes[1,0].set_title(f"Intensity ({np.sum(mask1>0)/mask1.size*100:.1f}%)", fontsize=12, weight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(mask2, cmap='gray')
    axes[1,1].set_title(f"Color ({np.sum(mask2>0)/mask2.size*100:.1f}%)", fontsize=12, weight='bold')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(mask3, cmap='gray')
    axes[1,2].set_title(f"Texture ({np.sum(mask3>0)/mask3.size*100:.1f}%)", fontsize=12, weight='bold')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(combined_raw, cmap='gray')
    axes[1,3].set_title(f"Voting ({np.sum(combined_raw>0)/combined_raw.size*100:.1f}%)", fontsize=12, weight='bold')
    axes[1,3].axis('off')
    
    # Row 3: Final
    axes[2,0].imshow(final, cmap='gray')
    axes[2,0].set_title(f"Final ({np.sum(final>0)/final.size*100:.1f}%)", fontsize=12, weight='bold')
    axes[2,0].axis('off')
    
    overlay = after.copy()
    final_resized = cv2.resize(final, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay[final_resized > 127] = [255, 0, 0]
    axes[2,1].imshow(cv2.addWeighted(after, 0.7, overlay, 0.3, 0))
    axes[2,1].set_title("Overlay (Red=Change)", fontsize=12, weight='bold')
    axes[2,1].axis('off')
    
    if gt_mask is not None:
        comparison = after.copy()
        gt_resized = cv2.resize(gt_mask, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_NEAREST)
        comparison[final_resized > 127] = [255, 0, 0]
        comparison[gt_resized > 127] = [0, 255, 0]
        comparison[(final_resized > 127) & (gt_resized > 127)] = [255, 255, 0]
        axes[2,2].imshow(comparison)
        axes[2,2].set_title("Yellow=TP, Red=FP, Green=FN", fontsize=10, weight='bold')
        axes[2,2].axis('off')
        
        metrics = calculate_metrics(final, gt_mask)
        text = "\n".join([f"{k.upper()}: {v:.3f}" for k, v in metrics.items()])
        axes[2,3].text(0.1, 0.3, text, fontsize=11, family='monospace')
        axes[2,3].set_title("Metrics", fontsize=12, weight='bold')
        axes[2,3].axis('off')
    else:
        axes[2,2].axis('off')
        axes[2,3].axis('off')
    
    plt.suptitle(f"Pipeline: {region_name.upper()} ({sensitivity})", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{region_name}_pipeline_{sensitivity}.png', dpi=200, bbox_inches='tight')
    plt.show()


def visualize_multi_region_grid(regions, sensitivity='balanced', max_regions=9):
    """Show multiple regions in grid"""
    num_regions = min(len(regions), max_regions)
    cols = 3
    rows = (num_regions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    detector = ProperlyTunedChangeDetector(sensitivity=sensitivity)
    
    for idx, region in enumerate(regions[:max_regions]):
        try:
            before, after, gt_mask = dataset.load_region(region)
            pred_mask = detector.detect(before, after)
            
            overlay = after.copy()
            pred_resized = cv2.resize(pred_mask, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay[pred_resized > 127] = [255, 0, 0]
            blended = cv2.addWeighted(after, 0.7, overlay, 0.3, 0)
            
            axes[idx].imshow(blended)
            
            if gt_mask is not None:
                metrics = calculate_metrics(pred_mask, gt_mask)
                title = f"{region}\nF1: {metrics['f1']:.3f} | IOU: {metrics['iou']:.3f}"
            else:
                change_pct = (np.sum(pred_mask > 0) / pred_mask.size) * 100
                title = f"{region}\nChanged: {change_pct:.2f}% (No GT)"
            
            axes[idx].set_title(title, fontsize=11, weight='bold')
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error: {region}', ha='center', va='center')
            axes[idx].axis('off')
    
    for idx in range(num_regions, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Multi-Region Grid ({sensitivity})", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'multi_region_grid_{sensitivity}.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_best_worst_average(results, sensitivity='balanced'):
    """Show best, worst, median performers"""
    valid = [r for r in results if r['metrics'] is not None]
    if len(valid) < 3:
        print("Not enough regions with GT")
        return
    
    sorted_results = sorted(valid, key=lambda x: x['metrics']['f1'], reverse=True)
    selected = [
        ('BEST', sorted_results[0]),
        ('MEDIAN', sorted_results[len(sorted_results)//2]),
        ('WORST', sorted_results[-1])
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    detector = ProperlyTunedChangeDetector(sensitivity=sensitivity)
    
    for row_idx, (label, result) in enumerate(selected):
        region = result['region']
        metrics = result['metrics']
        
        before, after, gt_mask = dataset.load_region(region)
        pred_mask = detector.detect(before, after)
        
        axes[row_idx, 0].imshow(before)
        axes[row_idx, 0].set_title(f"{label}: {region}\nBEFORE", fontsize=11, weight='bold')
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(after)
        axes[row_idx, 1].set_title("AFTER", fontsize=11, weight='bold')
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(pred_mask, cmap='gray')
        axes[row_idx, 2].set_title(f"Predicted ({np.sum(pred_mask>0)/pred_mask.size*100:.2f}%)", fontsize=11, weight='bold')
        axes[row_idx, 2].axis('off')
        
        axes[row_idx, 3].imshow(gt_mask, cmap='gray')
        text = f"GT ({np.sum(gt_mask>0)/gt_mask.size*100:.2f}%)\n"
        text += f"F1: {metrics['f1']:.3f}\nIOU: {metrics['iou']:.3f}\n"
        text += f"Prec: {metrics['precision']:.3f}\nRec: {metrics['recall']:.3f}"
        axes[row_idx, 3].text(1.1, 0.5, text, transform=axes[row_idx, 3].transAxes,
                             fontsize=10, va='center', family='monospace')
        axes[row_idx, 3].set_title("Ground Truth", fontsize=11, weight='bold')
        axes[row_idx, 3].axis('off')
    
    plt.suptitle(f"Best vs Median vs Worst ({sensitivity})", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f'best_worst_comparison_{sensitivity}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_performance_stats(results):
    """Generate statistical plots"""
    valid = [r for r in results if r['metrics'] is not None]
    if len(valid) == 0:
        return
    
    regions = [r['region'] for r in valid]
    ious = [r['metrics']['iou'] for r in valid]
    precisions = [r['metrics']['precision'] for r in valid]
    recalls = [r['metrics']['recall'] for r in valid]
    f1s = [r['metrics']['f1'] for r in valid]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 by region
    colors = ['green' if f1 > 0.25 else 'orange' if f1 > 0.15 else 'red' for f1 in f1s]
    axes[0,0].barh(regions, f1s, color=colors)
    axes[0,0].set_xlabel('F1 Score', fontsize=12)
    axes[0,0].set_title('F1 Score by Region', fontsize=14, weight='bold')
    axes[0,0].axvline(np.mean(f1s), color='blue', linestyle='--', label=f'Mean: {np.mean(f1s):.3f}')
    axes[0,0].legend()
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # Precision vs Recall
    scatter = axes[0,1].scatter(recalls, precisions, s=100, alpha=0.6, c=f1s, cmap='viridis')
    for i, region in enumerate(regions):
        axes[0,1].annotate(region, (recalls[i], precisions[i]), fontsize=8, alpha=0.7)
    axes[0,1].set_xlabel('Recall', fontsize=12)
    axes[0,1].set_ylabel('Precision', fontsize=12)
    axes[0,1].set_title('Precision-Recall Trade-off', fontsize=14, weight='bold')
    axes[0,1].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,1], label='F1 Score')
    
    # All metrics
    x = np.arange(len(regions))
    width = 0.2
    axes[1,0].bar(x - 1.5*width, ious, width, label='IOU', alpha=0.8)
    axes[1,0].bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    axes[1,0].bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    axes[1,0].bar(x + 1.5*width, f1s, width, label='F1', alpha=0.8)
    axes[1,0].set_ylabel('Score', fontsize=12)
    axes[1,0].set_title('All Metrics by Region', fontsize=14, weight='bold')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(regions, rotation=45, ha='right')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Box plot
    axes[1,1].boxplot([ious, precisions, recalls, f1s], labels=['IOU', 'Precision', 'Recall', 'F1'])
    axes[1,1].set_ylabel('Score', fontsize=12)
    axes[1,1].set_title('Metric Distribution', fontsize=14, weight='bold')
    axes[1,1].grid(axis='y', alpha=0.3)
    for i, data in enumerate([ious, precisions, recalls, f1s]):
        axes[1,1].text(i+1, np.mean(data), f'{np.mean(data):.3f}', 
                      ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_all_visualizations(sensitivity='balanced'):
    """Generate comprehensive visualization suite"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    detector = ProperlyTunedChangeDetector(sensitivity=sensitivity)
    results = []
    
    for region in dataset.regions_with_gt:
        try:
            before, after, gt_mask = dataset.load_region(region)
            pred_mask = detector.detect(before, after)
            metrics = calculate_metrics(pred_mask, gt_mask)
            results.append({'region': region, 'metrics': metrics})
        except:
            pass
    
    print("\n1. Multi-region grid...")
    visualize_multi_region_grid(dataset.regions_with_gt, sensitivity, max_regions=9)
    
    print("2. Best/worst comparison...")
    visualize_best_worst_average(results, sensitivity)
    
    print("3. Statistical plots...")
    plot_performance_stats(results)
    
    print("4. Pipeline visualizations (top 3)...")
    sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
    for result in sorted_results[:3]:
        print(f"   - {result['region']} (F1: {result['metrics']['f1']:.3f})")
        before, after, gt_mask = dataset.load_region(result['region'])
        visualize_detection_pipeline(before, after, gt_mask, result['region'], sensitivity)
    
    print("5. Regions without GT...")
    visualize_multi_region_grid(dataset.regions_without_gt, sensitivity, max_regions=6)
    
    print("\n✅ All visualizations complete!")

# ========================================
# TUNING & TESTING
# ========================================

def find_best_sensitivity():
    print(f"\n{'='*60}")
    print("SENSITIVITY TUNING")
    print('='*60)
    
    if len(dataset.regions_with_gt) == 0:
        return 'balanced'
    
    tuning_regions = dataset.regions_with_gt[:3]
    sensitivities = ['conservative', 'moderate', 'balanced', 'sensitive']
    all_results = {s: [] for s in sensitivities}
    
    for region in tuning_regions:
        print(f"\nTesting {region}...")
        before, after, gt_mask = dataset.load_region(region)
        for sens in sensitivities:
            detector = ProperlyTunedChangeDetector(sensitivity=sens)
            pred_mask = detector.detect(before, after)
            metrics = calculate_metrics(pred_mask, gt_mask)
            all_results[sens].append(metrics)
    
    print(f"\n{'Sensitivity':<15} {'IOU':<8} {'Precision':<12} {'Recall':<10} {'F1':<8}")
    print("─" * 60)
    
    avg_scores = {}
    for sens in sensitivities:
        avg_iou = np.mean([m['iou'] for m in all_results[sens]])
        avg_prec = np.mean([m['precision'] for m in all_results[sens]])
        avg_rec = np.mean([m['recall'] for m in all_results[sens]])
        avg_f1 = np.mean([m['f1'] for m in all_results[sens]])
        avg_scores[sens] = avg_f1
        print(f"{sens:<15} {avg_iou:.4f}  {avg_prec:.4f}      {avg_rec:.4f}    {avg_f1:.4f}")
    
    best = max(avg_scores, key=avg_scores.get)
    print(f"\nBest: {best} (F1={avg_scores[best]:.4f})")
    return best

def test_all_regions(sensitivity='balanced'):
    detector = ProperlyTunedChangeDetector(sensitivity=sensitivity)
    
    print(f"\n{'='*60}")
    print("EVALUATION ON ALL REGIONS")
    print('='*60)
    
    results = []
    for region in dataset.regions_with_gt:
        try:
            before, after, gt_mask = dataset.load_region(region)
            pred_mask = detector.detect(before, after)
            metrics = calculate_metrics(pred_mask, gt_mask)
            results.append({'region': region, 'metrics': metrics})
            print(f"{region:15s} IOU:{metrics['iou']:.4f} F1:{metrics['f1']:.4f}")
        except Exception as e:
            print(f"{region:15s} Error: {e}")
    
    if results:
        avg_iou = np.mean([r['metrics']['iou'] for r in results])
        avg_prec = np.mean([r['metrics']['precision'] for r in results])
        avg_rec = np.mean([r['metrics']['recall'] for r in results])
        avg_f1 = np.mean([r['metrics']['f1'] for r in results])
        
        print(f"\n{'─'*60}")
        print(f"AVERAGE: IOU:{avg_iou:.4f} Prec:{avg_prec:.4f} Rec:{avg_rec:.4f} F1:{avg_f1:.4f}")
        print(f"{'─'*60}")
    
    return results

# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    print("🚀 Satellite-Based Land Use Monitoring")
    print("   Classical Image Processing with Comprehensive Visualizations\n")
    
    # Step 1: Tune
    print("STEP 1: Finding optimal sensitivity...")
    best_sens = find_best_sensitivity()
    
    # Step 2: Test all
    print(f"\n\nSTEP 2: Testing all regions with '{best_sens}'...")
    results = test_all_regions(sensitivity=best_sens)
    
    # Step 3: Generate ALL visualizations
    print(f"\n\nSTEP 3: Generating comprehensive visualizations...")
    generate_all_visualizations(sensitivity=best_sens)
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("\nGenerated files:")
    print("  - multi_region_grid_*.png")
    print("  - best_worst_comparison_*.png")
    print("  - performance_statistics.png")
    print("  - [region]_pipeline_*.png (top 3 regions)")
    print("="*60)