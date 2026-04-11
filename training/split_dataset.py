import os
import shutil
import random

SOURCE_DIR = "dataset/all"
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

random.seed(42)

for class_name in os.listdir(SOURCE_DIR):
    images = os.listdir(os.path.join(SOURCE_DIR, class_name))
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(SOURCE_DIR, class_name, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(SOURCE_DIR, class_name, img),
            os.path.join(VAL_DIR, class_name, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(SOURCE_DIR, class_name, img),
            os.path.join(TEST_DIR, class_name, img)
        )

print("✅ Proper split completed")
