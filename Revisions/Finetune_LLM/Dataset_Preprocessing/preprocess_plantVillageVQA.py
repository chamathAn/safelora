import os
import shutil
import random

# Path to your main dataset folder
# Example: each subfolder inside 'PlantVillage' is a disease class
base_dir = "D:/safelora/Datasets/PlantVillage/train"
output_dir = "D:/safelora/Datasets"

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Recreate output folders
for split in ['train', 'val', 'test']:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Loop through every class folder
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    # Split indexes
    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # Copy to respective folders
    for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        dest_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for img in img_list:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dest_dir, img)
            shutil.copy2(src_path, dst_path)

print("✅ Dataset successfully split into train, val, and test folders!")
