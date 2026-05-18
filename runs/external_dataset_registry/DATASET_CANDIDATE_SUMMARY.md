# External Dataset Candidate Summary

Generated from `docs/02_dataset_training/external_dataset_registry.json`.

| Dataset | Format | Classes | Images | TACO Similarity | Decision |
|---|---|---:|---:|---|---|
| [TACO: Trash Annotations in Context](https://github.com/pedropro/TACO) | COCO instance segmentation; can derive boxes from segmentation/bbox fields | 60 | 1500 | reference dataset | Download official annotations/images, run COCO adapter, map 60 classes to project classes and/or TACO-10 |
| [MJU-Waste v1.0](https://github.com/realwecan/mju-waste) | PASCAL VOC; COCO instance annotations also available | 1 | 2475 | medium | Use for object-localization/segmentation evidence and feature distribution, not final multi-class sorting alone |
| [ZeroWaste](https://arxiv.org/abs/2106.02740) | Detection/semantic segmentation in cluttered industrial waste scenes | TBD | TBD | medium-high | Treat as external-domain stress test, not direct class-balance replacement |
| [TrashCan 1.0](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7) | Instance segmentation masks; material and instance label versions | 22 | 7212 | medium | Use as 'similar annotated debris dataset' in comparison table; avoid merging into final household sorting unless domain is justified |
| [TrashNet](https://github.com/garythung/trashnet) | Image classification folders, no boxes | 6 | 2527 | low-medium | Use for classification baseline only and explicitly mark as clean-background comparison |
| [Garbage in Images (GINI)](https://github.com/spotgarbage/spotgarbage-GINI) | Bounding boxes for one trash class | 1 | 2561 | medium | EDA/features only or binary trash-vs-background; do not use as standalone multi-class model |

## Risks
- **TACO: Trash Annotations in Context**: No official train/val/test split; many rare classes; needs COCO-to-YOLO/class-crop conversion
- **MJU-Waste v1.0**: Mostly waste-vs-background, not fine-grained material classes; Google Drive download/manual access
- **ZeroWaste**: Different domain from household/public litter; download page must be verified before use
- **TrashCan 1.0**: Underwater domain is very different from phone-camera household waste; academic/research-use license
- **TrashNet**: Clean/staged backgrounds, not TACO-like detection; cannot directly train detector without synthetic/full-image boxes
- **Garbage in Images (GINI)**: Single class; not enough for material sorting