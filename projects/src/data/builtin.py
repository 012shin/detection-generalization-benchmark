# -*- coding: utf-8 -*-
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from .builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data.datasets.pascal_voc import register_pascal_voc

def register_all_pascal_voc(root):
    SPLITS = [
        ("clipart_2012_val", "clipart", "test"),
        ("clipart_2012_test", "clipart", "test"),
        ("clipart_2012_train", "clipart", "train"),
        ("water_2012_train", "VOC_Water", "train"),
        ("water_2012_test", "VOC_Water", "test"),
        ("comic_2012_val", "comic", "test"),
        ("comic_2012_train", "comic", "train"),

    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

_root = os.getenv("DETECTRON2_DATASETS", "/home/dataset/detectron2")
register_all_pascal_voc(_root)