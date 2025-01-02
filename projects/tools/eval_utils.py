import logging
import os
from collections import OrderedDict
import csv


import detectron2.utils.comm as comm
import torch
from collections.abc import Mapping
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    PascalVOCDetectionEvaluator,
    LVISEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
)
from detectron2.evaluation import DatasetEvaluators, print_csv_format

from src.data.build import build_detection_test_loader
from src.data import CorruptionMapper



def get_evaluator(cfg, dataset_name, output_folder=None):
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
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, tasks=("bbox",), distributed=True))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def apply_corruption_test(cfg, model):
    results = OrderedDict()
    logger = logging.getLogger("gdet_training")
    results_file = "./inference/" + '_'.join(cfg.OUTPUT_DIR.split('/')[1:]) + ".csv"

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Prepare CSV header
    with open(results_file, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Dataset", "Noise Type", "Severity", "Metric", "Value"])

    if cfg.TEST.NOISE_ALL:
        corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                       "glass_blur", "zoom_blur", "motion_blur", "frost", "fog",
                       "brightness", "contrast", "elastic_transform", "pixelate",
                       "jpeg_compression"] #"snow" ImageMagick issue 로 삭제
        range_sev = [1, 2, 3, 4, 5]
        for corrupt in corruptions:
            for sev in range_sev:
                data_mapper = CorruptionMapper(cfg, is_train=False, eval_aug=corrupt, severity=sev)
                for dataset_name in cfg.DATASETS.TEST:
                    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper)
                    evaluator = get_evaluator(
                        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                    )
                    results_i = inference_on_dataset(model, data_loader, evaluator)
                    results[dataset_name + "{}_{}".format(corrupt, str(sev))] = results_i
                    if comm.is_main_process():
                        logger.info("Evaluation results for {} in csv format:".format(
                            dataset_name + "{}_{}".format(corrupt, str(sev))))
                        print_csv_format(results_i)
                    # Save results to CSV
                    with open(results_file, "a", newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        for task, res in results_i.items():
                            if isinstance(res, Mapping):
                                for k, v in res.items():
                                    if "-" not in k:  # Exclude metrics like "AP-category"
                                        writer.writerow([dataset_name, corrupt, sev, k, v])
                            else:
                                writer.writerow([dataset_name, corrupt, sev, task, res])
        return results
    else:
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = get_evaluator(
                cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            )
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
            # Save results to CSV
            with open(results_file, "a", newline='') as csv_file:
                writer = csv.writer(csv_file)
                for task, res in results_i.items():
                    if isinstance(res, Mapping):
                        for k, v in res.items():
                            if "-" not in k:  # Exclude metrics like "AP-category"
                                writer.writerow([dataset_name, "", "", k, v])
                    else:
                        writer.writerow([dataset_name, "", "", task, res])
        return results


# def do_test_resume(cfg, model, results_file="./inference/output.csv"):
#     # Load completed tasks
#     completed_tasks = set()
#     if os.path.exists(results_file):
#         with open(results_file, "r") as csv_file:
#             reader = csv.reader(csv_file)
#             next(reader)  # Skip header
#             for row in reader:
#                 dataset, corruption, severity, _, _ = row
#                 completed_tasks.add((dataset, corruption, severity))

#     # Prepare for inference
#     results = OrderedDict()
#     logger = logging.getLogger("gdet_training")
#     corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
#                    "glass_blur", "zoom_blur", "motion_blur", "frost", "fog",
#                    "brightness", "contrast", "elastic_transform", "pixelate",
#                    "jpeg_compression"]
#     range_sev = [1, 2, 3, 4, 5]

#     for corrupt in corruptions:
#         for sev in range_sev:
#             for dataset_name in cfg.DATASETS.TEST:
#                 # Skip already completed tasks
#                 if (dataset_name, corrupt, str(sev)) in completed_tasks:
#                     continue

#                 # Build data loader and evaluator
#                 data_mapper = CorruptionMapper(cfg, is_train=False, eval_aug=corrupt, severity=sev)
#                 data_loader = build_detection_test_loader(cfg, dataset_name, mapper=data_mapper)
#                 evaluator = get_evaluator(
#                     cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#                 )
                
#                 # Perform inference
#                 results_i = inference_on_dataset(model, data_loader, evaluator)
#                 results[dataset_name + "{}_{}".format(corrupt, str(sev))] = results_i
                
#                 if is_main_process():
#                     logger.info("Evaluation results for {} in csv format:".format(
#                         dataset_name + "{}_{}".format(corrupt, str(sev))))
#                     print_csv_format(results_i)
                
#                 # Save results to CSV
#                 with open(results_file, "a", newline='') as csv_file:
#                     writer = csv.writer(csv_file)
#                     for task, res in results_i.items():
#                         if isinstance(res, Mapping):
#                             for k, v in res.items():
#                                 if "-" not in k:  # Exclude metrics like "AP-category"
#                                     writer.writerow([dataset_name, corrupt, sev, k, v])
#                         else:
#                             writer.writerow([dataset_name, corrupt, sev, task, res])
#     return results
