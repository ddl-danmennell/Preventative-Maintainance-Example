#!/usr/bin/env python3
"""
Generate synthetic thermal images for equipment monitoring
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json

np.random.seed(42)

# Configuration
OUTPUT_DIR = Path("/mnt/data/Preventative-Maintainance-Example/epoch002-data-wrangling/thermal_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORMAL_DIR = OUTPUT_DIR / "normal"
DEGRADED_DIR = OUTPUT_DIR / "degraded"
NORMAL_DIR.mkdir(exist_ok=True)
DEGRADED_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
TEMP_RANGE = (20, 150)  # Celsius

def temp_to_rgb(temp_field):
    """Convert temperature field to RGB thermal colormap"""
    # Normalize to 0-1
    normalized = (temp_field - TEMP_RANGE[0]) / (TEMP_RANGE[1] - TEMP_RANGE[0])
    normalized = np.clip(normalized, 0, 1)

    # Create RGB (blue -> green -> yellow -> red)
    rgb = np.zeros((*temp_field.shape, 3))
    rgb[:, :, 2] = 1 - normalized  # Blue decreases
    rgb[:, :, 1] = 1 - 2 * np.abs(normalized - 0.5)  # Green peaks in middle
    rgb[:, :, 0] = normalized  # Red increases

    return (rgb * 255).astype(np.uint8)

def generate_normal_pattern():
    """Generate normal equipment thermal pattern"""
    temp_field = np.random.uniform(30, 50, IMAGE_SIZE)

    # Add 1-3 normal hotspots
    num_hotspots = np.random.randint(1, 4)
    for _ in range(num_hotspots):
        x, y = np.random.randint(40, IMAGE_SIZE[0]-40), np.random.randint(40, IMAGE_SIZE[1]-40)
        size = np.random.randint(20, 40)

        xx, yy = np.meshgrid(range(IMAGE_SIZE[0]), range(IMAGE_SIZE[1]))
        hotspot = 20 * np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * size**2))
        temp_field += hotspot

    temp_field += np.random.normal(0, 2, IMAGE_SIZE)
    return np.clip(temp_field, *TEMP_RANGE)

def generate_degraded_pattern(severity):
    """Generate degraded equipment thermal pattern"""
    temp_field = generate_normal_pattern()

    # Add anomalous heat
    anomaly_temp = 40 + severity * 60
    x, y = np.random.randint(50, IMAGE_SIZE[0]-50), np.random.randint(50, IMAGE_SIZE[1]-50)
    size = int(30 + severity * 30)

    xx, yy = np.meshgrid(range(IMAGE_SIZE[0]), range(IMAGE_SIZE[1]))
    anomaly = anomaly_temp * np.exp(-((xx-x)**2 + (yy-y)**2) / (2 * size**2))
    temp_field += anomaly

    temp_field += np.random.normal(0, 3 * (1 + severity), IMAGE_SIZE)
    return np.clip(temp_field, *TEMP_RANGE)

print("Generating thermal images...")
print(f"Output directory: {OUTPUT_DIR}")

# Generate normal images
print("\nGenerating normal images...")
for i in range(250):
    temp_field = generate_normal_pattern()
    rgb = temp_to_rgb(temp_field)
    img = Image.fromarray(rgb)

    filepath = NORMAL_DIR / f"normal_{i:04d}.png"
    img.save(filepath)

    if (i + 1) % 50 == 0:
        print(f"  Generated {i+1}/250 normal images")

# Generate degraded images
print("\nGenerating degraded images...")
for i in range(250):
    severity = np.random.uniform(0.3, 1.0)
    temp_field = generate_degraded_pattern(severity)
    rgb = temp_to_rgb(temp_field)
    img = Image.fromarray(rgb)

    filepath = DEGRADED_DIR / f"degraded_{i:04d}.png"
    img.save(filepath)

    # Save metadata
    metadata = {
        'class': 'degraded',
        'severity': float(severity),
        'mean_temp': float(np.mean(temp_field)),
        'max_temp': float(np.max(temp_field))
    }
    with open(filepath.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f)

    if (i + 1) % 50 == 0:
        print(f"  Generated {i+1}/250 degraded images")

# Calculate total size
total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file())
total_size_mb = total_size / 1024 / 1024

# Create summary
summary = {
    'total_images': 500,
    'normal': 250,
    'degraded': 250,
    'size_mb': round(total_size_mb, 2),
    'image_size': IMAGE_SIZE
}

summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("IMAGE GENERATION COMPLETE")
print("="*60)
print(f"Total images: {summary['total_images']}")
print(f"Normal: {summary['normal']}")
print(f"Degraded: {summary['degraded']}")
print(f"Total size: {summary['size_mb']:.2f} MB")
