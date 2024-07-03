import json
import logging
import os

import datasets
from datasets import (
    DatasetDict,
    Image,
    Sequence,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from tqdm import tqdm

logger = logging.getLogger("load_dataset")


# Load hg interpret-mimic-cxr
def load_interpret_cxr(data_dir):
    return load_from_disk(data_dir)


# Load mimic-cxr
def load_mimic_cxr(data_dir, num_proc=4):

    dataset_mimic = load_dataset(
        "json",
        data_files={"train": os.path.join(data_dir, "train_mimic.json"), "validation": os.path.join(data_dir, "val_mimic.json")},
    )

    def add_prefix(example):
        example["images"] = [os.path.join(data_dir, i) for i in example["images"]]
        return example

    dataset_mimic = dataset_mimic.map(add_prefix, num_proc=num_proc)
    dataset_mimic = dataset_mimic.cast_column("images", Sequence(Image()))

    return dataset_mimic


# Load labels to dataset in memory
def load_labels_to_dataset(label_file_path, dataset):
    data2label_map = {}
    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Reading label file"):
            data = json.loads(line)
            data2label_map[data["data_key"]] = data

    for split in ["train", "validation"]:
        present = []
        absent = []
        uncertain = []

        ds = dataset[split]
        for idx, (source, images_path) in enumerate(tqdm(zip(ds["source"], ds["images_path"]), desc="Loading labels", total=len(ds))):
            label_dict = data2label_map[f"{split}#{idx}"]
            present.append(label_dict["present"])
            absent.append(label_dict["absent"])
            uncertain.append(label_dict["uncertain"])
            assert label_dict["vaild_key"] == f"{source}#{images_path[0]}"

        logger.debug("Adding label columns to the %s dataset", split)
        ds = ds.add_column(name="label_present", column=present)
        ds = ds.add_column(name="label_absent", column=absent)
        dataset[split] = ds.add_column(name="label_uncertain", column=uncertain)


def get_dataset(interpret_cxr_dir, mimic_cxr_dir, label_file_path):
    dataset_itp = load_interpret_cxr(interpret_cxr_dir)
    logger.debug("%s loaded from interpret_cxr", [f"{split}:{len(ds)}" for split, ds in dataset_itp.items()])
    dataset_mimic = load_mimic_cxr(mimic_cxr_dir)
    logger.debug("%s loaded from mimic-cxr", [f"{split}:{len(ds)}" for split, ds in dataset_mimic.items()])

    # Concat both
    dataset_final = DatasetDict({"train": concatenate_datasets([dataset_itp["train"], dataset_mimic["train"]]), "validation": concatenate_datasets([dataset_itp["validation"], dataset_mimic["validation"]])})
    logger.debug("%s have been concatenated to the final dataset", [f"{split}:{len(ds)}" for split, ds in dataset_final.items()])

    load_labels_to_dataset(label_file_path, dataset=dataset_final)

    return dataset_final


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    dataset = get_dataset(interpret_cxr_dir="/storage/4T_add/liao/arrg/interpret-cxr", mimic_cxr_dir="/storage/4T_add/liao/arrg/mimic-cxr-jpg-resized", label_file_path="/storage/4T_add/liao/arrg/img_label_ids.json")
    print(dataset["train"])
