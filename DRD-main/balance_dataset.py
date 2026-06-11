import os
import shutil
import random

data_dir = '/Users/pradeepkd/Desktop/projectphase1/archive/colored_images'
target_count = 800  # Balance to 800 samples per class

classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

for cls in classes:
    cls_dir = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(cls_dir) if f.endswith('.png')]
    current_count = len(images)
    if current_count < target_count:
        # Duplicate random images
        to_add = target_count - current_count
        for i in range(to_add):
            src = os.path.join(cls_dir, random.choice(images))
            dst = os.path.join(cls_dir, f'copy_{i}_{os.path.basename(src)}')
            shutil.copy(src, dst)
        print(f'Balanced {cls} from {current_count} to {target_count} images')
    else:
        print(f'{cls} already has {current_count} images')

print('Dataset balancing complete.')