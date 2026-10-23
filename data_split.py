"""
Leak-free train/validation generators for the DR image folders.

balance_dataset.py balances classes by writing duplicates named
"copy_<n>_<original>.png" into the same folder. Keras' validation_split
splits by file order, so a duplicate can land in training while its original
is in validation, which inflates validation accuracy. Here the split is done
per original image: an original and all of its copies always go to the same
side, and validation only keeps the originals.
"""
import os
import re
import random

import pandas as pd

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')
COPY_PREFIX = re.compile(r'^copy_\d+_')


def _original_name(filename):
    return COPY_PREFIX.sub('', filename)


def split_dataframes(data_dir, val_fraction=0.2, seed=42):
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMAGE_EXTS))
        originals = sorted({_original_name(f) for f in files})
        rng.shuffle(originals)
        val_originals = set(originals[:int(len(originals) * val_fraction)])
        for f in files:
            row = {'filename': os.path.join(cls_dir, f), 'class': cls}
            if _original_name(f) in val_originals:
                if f == _original_name(f):  # keep only originals in validation
                    val_rows.append(row)
            else:
                train_rows.append(row)
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), class_names


def make_generators(data_dir, train_datagen, val_datagen, target_size, batch_size,
                    val_fraction=0.2, seed=42):
    """Return (train_generator, validation_generator).

    The validation generator is not shuffled, so its predictions line up with
    validation_generator.classes.
    """
    train_df, val_df, class_names = split_dataframes(data_dir, val_fraction, seed)
    common = dict(x_col='filename', y_col='class', classes=class_names,
                  target_size=target_size, batch_size=batch_size,
                  class_mode='categorical')
    train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, seed=seed, **common)
    val_gen = val_datagen.flow_from_dataframe(val_df, shuffle=False, **common)
    return train_gen, val_gen
