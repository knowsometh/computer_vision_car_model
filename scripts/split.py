import os, shutil, random

train_dir = '../data/train'
val_dir = '../data/val'

os.makedirs(val_dir, exist_ok=True)

# 20% to take for validation
split_ratio = 0.2

for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)
    val_count = int(len(images) * split_ratio)

    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)

    for img_name in images[:val_count]:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(val_class_dir, img_name)
        shutil.move(src, dst)

print("Validation split created successfully!")
