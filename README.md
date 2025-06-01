# Counting-Sea-Lions
[NOAA Fisheries Steller Sea Lion Population Count](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count) by [YOLO](https://docs.ultralytics.com/models/yolo11/).
![image](https://github.com/user-attachments/assets/9390ab5e-65bb-44bf-b825-66de5c7db4c9)

# Background
Steller sea lions in the western Aleutian Islands have declined 94 percent in the last 30 years. The endangered western population, found in the North Pacific, are the focus of conservation efforts which require annual population counts. Specially trained scientists at NOAA Fisheries Alaska Fisheries Science Center conduct these surveys using airplanes and unoccupied aircraft systems to collect aerial images. Having accurate population estimates enables us to better understand factors that may be contributing to lack of recovery of Stellers in this area.

Currently, it takes biologists up to four months to count sea lions from the thousands of images NOAA Fisheries collects each year. Once individual counts are conducted, the tallies must be reconciled to confirm their reliability. The results of these counts are time-sensitive.

Thus, we develop algorithms which accurately count the number of sea lions in aerial photographs. Automating the annual population count will free up critical resources allowing NOAA Fisheries to focus on ensuring we hear the sea lion’s roar for many years to come. Plus, advancements in computer vision applied to aerial population counts may also greatly benefit other endangered species.

You can learn more about research being done to better understand what's going on with the endangered Steller sea lion populations by joining scientists on a research vessel to the western Aleutian Islands in this [video](https://www.youtube.com/watch?v=oiL8tDCqzy4).

# Method

## Dataset Preprocessing

### 1. Adaptive Bounding Box Annotation

Our preprocessing pipeline implements an adaptive bounding box generation system that automatically adjusts annotation sizes based on sea lion density and proximity.

#### 1.1 Initial Box Generation
- **Base box size**: 55×55 pixels centered on each sea lion coordinate
- **Minimum allowable size**: 22×22 pixels (40% of base size)
- **Class mapping**: 
  - 0: Adult males
  - 1: Subadult males  
  - 2: Adult females
  - 3: Juveniles
  - 4: Pups

#### 1.2 IoU-based Box Adjustment
When bounding boxes overlap, we apply adaptive size reduction based on IoU thresholds:

| IoU Range | Primary Reduction | Secondary Reduction |
|-----------|------------------|-------------------|
| ≥ 0.50    | 0.70            | 0.85             |
| ≥ 0.45    | 0.72            | 0.86             |
| ≥ 0.40    | 0.74            | 0.87             |
| ≥ 0.35    | 0.76            | 0.88             |
| ≥ 0.30    | 0.78            | 0.89             |
| ≥ 0.25    | 0.81            | 0.905            |
| ≥ 0.20    | 0.84            | 0.92             |
| ≥ 0.15    | 0.87            | 0.935            |
| ≥ 0.10    | 0.90            | 0.95             |

The adjustment strategy prioritizes the dimension with larger displacement between box centers.

#### 1.3 Image Masking
- Apply dotted image masks to remove irrelevant background areas
- Black pixels from dotted images are transferred to original images
- Helps focus training on relevant sea lion habitats

### 2. Patch Generation Strategy

#### 2.1 Dual Sampling Approach

Method 1: **Targeted Sampling**
- Select patches centered on bounding boxes
- Sample rate: 20% of available bounding boxes
- Min patches per image: 1
- Max patches per image: 3
- Ensures high-quality positive samples

Method 2: **Grid Sampling**  
- Systematic grid-based patch extraction
- Stride: 95% of patch size (1216 pixels for 1280×1280 patches)
- Auto-calculated grid dimensions based on image size
- Provides comprehensive coverage

#### 2.2 Patch Processing Pipeline
1. **Initial extraction**: 1280×1280 pixels
2. **Resize to training size**: 640×640 pixels  
3. **Quality filtering**:
   - Black pixel threshold: < 50%
   - Zero-bbox dropout rate: 70%
4. **Box coordinate adjustment**: Scale and clip to patch boundaries

### 3. YOLO Format Conversion

#### 3.1 Coordinate Normalization
- Convert (x1, y1, x2, y2) to (x_center, y_center, width, height)
- Normalize all coordinates to [0, 1] range
- Ensure boxes remain within image boundaries

#### 3.2 Quality Control
- Filter boxes with area < 20% of original after clipping
- Remove degenerate boxes (width ≤ 0 or height ≤ 0)
- Maintain class-to-filename mapping in counts.csv

## Model Architecture

### YOLO11x Configuration
- **Base model**: YOLOv11x (largest variant)
- **Input size**: 640×640 pixels
- **Classes**: 5 (sea lion age/gender categories)
- **Confidence threshold**: 0.5
- **NMS IoU threshold**: 0.7

### Training Configuration

#### 4.1 Training Parameters
```python
TOTAL_EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 2
NUM_WORKERS = 4  # (0 on Windows)
SEED = 42
NUM_COUNTING_CLASSES = 5
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.7
```

#### 4.2 Optimization Settings
- **Learning rate scheduler**: Cosine annealing (`cos_lr=True`)
- **Mixed precision**: AMP enabled (`amp=True`)
- **Dropout**: 0.05
- **Multi-scale training**: Enabled
- **Random seed**: 42 (for reproducibility)

#### 4.3 Data Augmentation Pipeline

Our training pipeline employs a comprehensive data augmentation strategy to improve model robustness and generalization across diverse imaging conditions.

**Color Space Augmentation:**
```python
HSV_H = 0.015      # Hue shift: ±1.5%
HSV_S = 0.70       # Saturation: ±70%
HSV_V = 0.40       # Value/Brightness: ±40%
```

**Geometric Transformations:**
```python
ROTATION_DEGREE = 3.0    # Random rotation: ±3°
TRANSLATE = 0.02         # Translation: ±2% of image size
RANDOM_SCALE = 0.3       # Scaling: ±30%
SHEAR = 0.0             # Shear transformation: disabled
PERSPECTIVE = 0.0005     # Perspective distortion: 0.05%
```

**Flip Augmentations:**
```python
FLIP_UD = 0.2           # Vertical flip: 20% probability
FLIP_LR = 0.3           # Horizontal flip: 30% probability
```

**Advanced Augmentation Techniques:**
```python
MOSAIC = 0.0            # Mosaic augmentation: disabled
COPY_PASTE = 0.08       # Copy-paste: 8% probability
ERASE = 0.08            # Random erasing: 8% probability
MIXUP = 0.04            # MixUp: 4% probability
```

**Copy-Paste Augmentation (8%):**
- Copies sea lion instances from one image to another
- Increases sample diversity and handles occlusion scenarios
- Particularly effective for rare classes (adult males, pups)

**Random Erasing (8%):**
- Randomly masks rectangular regions
- Improves robustness to partial occlusions
- Forces model to rely on partial visual cues

**MixUp (4%):**
- Blends two images with corresponding labels
- Enhances model's decision boundary smoothness
- Applied conservatively to preserve detection accuracy

**Disabled Augmentations:**
- **Mosaic (0%)**: Disabled to maintain spatial relationships critical for counting
- **Shear (0%)**: Avoided to preserve natural sea lion proportions

#### 4.4 Training Strategy
- **Validation split**: 5% of training data
- **Early stopping**: Monitor validation mAP
- **Model checkpointing**: Save best weights based on validation performance
- **Gradient clipping**: Enabled for training stability
- **Warmup epochs**: 3 epochs with linear learning rate warmup

This augmentation strategy balances data diversity with task-specific requirements, ensuring the model learns robust features while maintaining high counting accuracy for aerial sea lion detection.

# Results
![image](https://github.com/user-attachments/assets/b9a477ca-b636-4545-8a3c-22737bd78bcc)

**RMSE** on Kaggle
- Score: 11.65312
- Public score: 11.68292

**Rank**: 2nd place
- Rank1: 10.85644
- Rank2 (**Ours**): 11.65312
- Rank3: 12.50888
- Rank4: 13.03257
- Rank5: 13.18968
- Rank6: 13.92760

