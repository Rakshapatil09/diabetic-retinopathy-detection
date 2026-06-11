import pandas as pd
import os
import shutil
import glob

# Load CSV
csv_path = '/Users/pradeepkd/Desktop/projectphase1/archive/train.csv'
df = pd.read_csv(csv_path)

# Mapping diagnosis to folder names
diagnosis_to_folder = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR'
}

# Source directory
source_dir = '/Users/pradeepkd/Desktop/projectphase1/archive/colored_images'

# Create folders if not exist
for folder in diagnosis_to_folder.values():
    os.makedirs(os.path.join(source_dir, folder), exist_ok=True)

# Find all images
image_files = glob.glob(os.path.join(source_dir, '**', '*.png'), recursive=True)

# Create a dict of id_code to path
image_dict = {}
for path in image_files:
    filename = os.path.basename(path)
    id_code = filename.replace('.png', '')
    image_dict[id_code] = path

# Move images based on CSV
for _, row in df.iterrows():
    id_code = row['id_code']
    diagnosis = row['diagnosis']
    folder = diagnosis_to_folder[diagnosis]
    if id_code in image_dict:
        src = image_dict[id_code]
        dst = os.path.join(source_dir, folder, f'{id_code}.png')
        shutil.move(src, dst)
        print(f'Moved {id_code}.png to {folder}')
    else:
        print(f'Image {id_code}.png not found')

print('Reorganization complete.')