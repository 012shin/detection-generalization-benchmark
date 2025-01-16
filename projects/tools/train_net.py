#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import itertools
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader,DatasetMapper
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from src.evaluator import PascalVOCDetectionEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

# from src.data import SparseRCNNDatasetMapper
from src.config import get_cfg
from src.data.dataset_mapper import SparseRCNNDatasetMapper

from typing import Any, Dict, List, Set
from detectron2.solver.build import maybe_add_gradient_clipping

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)    
    
    @classmethod
    def build_train_loader(cls, cfg): # 이 부분으로 detector 별 datasetmapper를 적용시킬 수 있을거같음
        if cfg.MODEL.META_ARCHITECTURE in ["SparseRCNN", "DiffusionDet"]:
            mapper = SparseRCNNDatasetMapper(cfg, is_train=True)
            return build_detection_train_loader(cfg, mapper=mapper) 
        elif "vit" in cfg.MODEL.WEIGHTS:
            mapper = DatasetMapper(cfg,is_train=True, augmentations=[
                T.RandomFlip(horizontal=True),
                T.ResizeScale( min_scale=0.1, max_scale=2.0, target_height=cfg.MODEL.VIT.INPUT_SIZE, target_width=cfg.MODEL.VIT.INPUT_SIZE),
                T.FixedSizeCrop(crop_size=(cfg.MODEL.VIT.INPUT_SIZE, cfg.MODEL.VIT.INPUT_SIZE), pad=False)
            ])
            return build_detection_train_loader(cfg,mapper = mapper)
        else:
            return build_detection_train_loader(cfg)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        # fully fine-tuning
        if cfg.MODEL.BACKBONE.ADAPTER.MODE == 'ft':
            for key, value in model.named_parameters(recurse=True):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "backbone" in key:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                print(key) 
        # adapter
        elif cfg.MODEL.BACKBONE.ADAPTER.MODE == 'peft':
            for key, value in model.named_parameters(recurse=True):
                if value in memo:
                    continue
                if not value.requires_grad:
                    continue
                if (
                    any(part in key for part in [
                        'adapter','proposal_generator','roi_heads', 'top_block', 'fpn_lateral', 'fpn_output'
                    ])):
                    memo.add(value)
                    lr = cfg.SOLVER.BASE_LR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY
                    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                    print(key)
        # backbone layer
        elif cfg.MODEL.BACKBONE.ADAPTER.MODE == 'layer':
            for key, value in model.named_parameters(recurse=True):
                if value in memo:
                    continue
                if not value.requires_grad:
                    continue
                # 조건 수정: 'res2', 'res3'가 포함된 경우를 확인
                if (
                    any(block in key for block in ['res5.']) or
                    any(part in key for part in [
                        'proposal_generator','roi_heads', 'top_block', 'fpn_lateral', 'fpn_output'
                    ])
                ):
                    memo.add(value)
                    lr = cfg.SOLVER.BASE_LR
                    if "backbone" in key:
                        lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY
                    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                    print(key)
        # decoder probing
        else:
            for key, value in model.named_parameters(recurse=True):
                if value in memo:
                    continue
                if not value.requires_grad:
                    continue
                if (
                    any(part in key for part in [
                        'proposal_generator','roi_heads', 'top_block', 'fpn_lateral', 'fpn_output'
                    ])):
                    memo.add(value)
                    lr = cfg.SOLVER.BASE_LR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY
                    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
                    print(key)

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
