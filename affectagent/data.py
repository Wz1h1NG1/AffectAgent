"""MER-UniBench / MER-Caption+ sample loading shared by SFT, MAPPO and evaluation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from .labels import get_task_type
from .schemas import RolloutSample


def _dataset_classes():
    from my_affectgpt.datasets.datasets.cmumosei import CMUMOSEI_Dataset
    from my_affectgpt.datasets.datasets.cmumosi import CMUMOSI_Dataset
    from my_affectgpt.datasets.datasets.iemocap import IEMOCAPFour_Dataset
    from my_affectgpt.datasets.datasets.meld import MELD_Dataset
    from my_affectgpt.datasets.datasets.mer2023 import MER2023_Dataset
    from my_affectgpt.datasets.datasets.mer2024 import MER2024_Dataset
    from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
    from my_affectgpt.datasets.datasets.ovmerd_dataset import OVMERD_Dataset
    from my_affectgpt.datasets.datasets.ovmerdplus_dataset import OVMERDPlus_Dataset
    from my_affectgpt.datasets.datasets.sims import SIMS_Dataset
    from my_affectgpt.datasets.datasets.simsv2 import SIMSv2_Dataset

    return {
        "mercaptionplus": MERCaptionPlus_Dataset,
        "ovmerd": OVMERD_Dataset,
        "ovmerdplus": OVMERDPlus_Dataset,
        "mer2023": MER2023_Dataset,
        "mer2024": MER2024_Dataset,
        "meld": MELD_Dataset,
        "iemocapfour": IEMOCAPFour_Dataset,
        "cmumosi": CMUMOSI_Dataset,
        "cmumosei": CMUMOSEI_Dataset,
        "sims": SIMS_Dataset,
        "simsv2": SIMSv2_Dataset,
    }


DATASET_ALIASES = {"iemocap": "iemocapfour"}


def canonical_dataset_name(name: str) -> str:
    name = name.lower()
    return DATASET_ALIASES.get(name, name)


def load_dataset_cls(name: str):
    name = canonical_dataset_name(name)
    classes = _dataset_classes()
    if name not in classes:
        raise ValueError(f"Unknown dataset: {name}. Supported: {sorted(classes)}")
    return classes[name]()


def resolve_face_or_frame(cfg) -> str:
    datasets_cfg = cfg.datasets_cfg
    for name in ("mercaptionplus", "ovmerd"):
        if name in datasets_cfg:
            return str(datasets_cfg[name].face_or_frame)
    return "multiframe_audio_frame_text"


def configure_dataset(dataset_cls, cfg, face_or_frame: str) -> None:
    """Attach the inference-time processors and modality requirements to a dataset helper."""

    from my_affectgpt.common.registry import registry
    from my_affectgpt.processors.base_processor import BaseProcessor

    dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
    dataset_cls.vis_processor = BaseProcessor()
    dataset_cls.img_processor = BaseProcessor()
    inference_cfg = cfg.inference_cfg
    for attribute, key in (("vis_processor", "vis_processor"), ("img_processor", "img_processor")):
        processor_cfg = inference_cfg.get(key) if inference_cfg is not None else None
        if processor_cfg is None:
            processor_cfg = cfg.model_cfg.get(key)
        if processor_cfg is not None:
            processor = registry.get_processor_class(processor_cfg.train.name).from_config(processor_cfg.train)
            setattr(dataset_cls, attribute, processor)
    dataset_cls.n_frms = cfg.model_cfg.vis_processor.train.n_frms


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def _polarity(valence) -> str:
    valence = float(valence)
    return "positive" if valence > 0 else "negative" if valence < 0 else "neutral"


def read_split(dataset_cls, dataset_name: str, split: str) -> List[Dict[str, str]]:
    """Return ``[{name, subtitle, ground_truth, description}]`` for the train or test split."""

    task_type = get_task_type(dataset_name)
    name2subtitle = getattr(dataset_cls, "name2subtitle", {})
    records = []
    if split == "test":
        if not hasattr(dataset_cls, "read_test_names"):
            raise ValueError(f"{dataset_name} has no test split")
        name2gt = dataset_cls.get_test_name2gt()
        idx2emo = None
        for name in dataset_cls.read_test_names():
            gt = name2gt.get(name, "")
            valence = None
            if task_type == "sentiment":
                valence = float(gt)  # raw score, needed by the WAF protocol
                gt = _polarity(gt)
            elif isinstance(gt, (int, np.integer)):
                # MELD stores class indices for its test split.
                if idx2emo is None:
                    idx2emo = dataset_cls.get_emo2idx_idx2emo()[1]
                gt = idx2emo[int(gt)]
            records.append({"name": name, "subtitle": _clean_text(name2subtitle.get(name, "")),
                            "ground_truth": str(gt), "description": "", "valence": valence})
    else:
        for annotation in dataset_cls.annotation:
            gt = annotation.get("ovlabel", annotation.get("onehot", annotation.get("sentiment", "")))
            records.append({
                "name": annotation["name"],
                "subtitle": _clean_text(annotation.get("subtitle", name2subtitle.get(annotation["name"], ""))),
                "ground_truth": str(gt),
                "description": _clean_text(annotation.get("description", "")),
            })
    return [record for record in records if record["name"] and record["ground_truth"]]


def read_sample_data(dataset_cls, name: str) -> Optional[dict]:
    """Read every available modality; a modality that fails to load is set to None."""

    sample = {"name": name}
    paths = {
        "video_path": dataset_cls._get_video_path(sample) if hasattr(dataset_cls, "_get_video_path") else None,
        "face_npy": dataset_cls._get_face_path(sample) if hasattr(dataset_cls, "_get_face_path") else None,
        "audio_path": dataset_cls._get_audio_path(sample) if hasattr(dataset_cls, "_get_audio_path") else None,
    }
    sample_data = {key: None for key in ("frame", "raw_frame", "face", "raw_face", "audio", "raw_audio", "image", "raw_image")}
    loaded = False
    for argument, path in paths.items():
        if not path:
            continue
        try:
            partial = dataset_cls.read_frame_face_audio_text(**{argument: path})
        except Exception as error:
            print(f"[AffectAgent] {name}: failed to read {argument} ({error})")
            continue
        for key, value in partial.items():
            if value is not None:
                sample_data[key] = value
                loaded = True
    return sample_data if loaded else None


class AffectAgentDataset(Dataset):
    """Yields ``(RolloutSample, sample_data)``; samples whose media cannot be read are dropped."""

    def __init__(
        self,
        dataset_cls,
        dataset_name: str,
        candidate_labels: List[str],
        split: str = "train",
        lang: str = "en",
        max_samples: int = -1,
    ):
        self.dataset_cls = dataset_cls
        self.dataset_name = canonical_dataset_name(dataset_name)
        self.task_type = get_task_type(self.dataset_name)
        self.candidate_labels = list(candidate_labels)
        self.lang = lang
        self.records = read_split(dataset_cls, self.dataset_name, split)
        if 0 < max_samples < len(self.records):
            self.records = self.records[:max_samples]
        self.by_name = {record["name"]: record for record in self.records}
        print(f"[AffectAgentDataset] {self.dataset_name}/{split}: {len(self.records)} samples")

    def __len__(self) -> int:
        return len(self.records)

    def make_sample(self, record: Dict[str, str]) -> RolloutSample:
        # The annotated description is a training target, so it is not attached to the sample.
        return RolloutSample(
            sample_id=record["name"],
            subtitle=record["subtitle"],
            ground_truth=record["ground_truth"],
            candidate_labels=self.candidate_labels,
            dataset=self.dataset_name,
            lang=self.lang,
            task_type=self.task_type,
        )

    def __getitem__(self, index: int) -> Tuple[RolloutSample, Optional[dict]]:
        record = self.records[index]
        return self.make_sample(record), read_sample_data(self.dataset_cls, record["name"])

    @staticmethod
    def collate_fn(batch):
        kept = [(sample, data) for sample, data in batch if data is not None]
        return [sample for sample, _ in kept], [data for _, data in kept]
