import os
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from .schemas import GROUPS


class DualChannelRetriever:
    """
    Frozen AffectAgent dual-channel Retriever (part of the environment, no gradients).

    1. Cognitive channel: E5 semantic retrieval over the MER-Caption+ library, driven by the
       Query Planner's support / confusion / counter queries -> E_cog.
    2. Perceptual channel: FAISS audiovisual retrieval with the current sample's pooled
       AffectGPT features -> E_perc (tokens for RAAF, plus the partner's metadata/subtitle).
    """

    def __init__(self, semantic_index_dir=None, multimodal_index_dir=None, token_store_path=None):
        self.semantic_index_dir = semantic_index_dir
        self.multimodal_index_dir = multimodal_index_dir
        self.metadata = []
        self.semantic_faiss = None
        self.encoder = None

        if self.semantic_index_dir:
            print(f"Loading Semantic Index from {semantic_index_dir}...")
            self._load_semantic_index()

        self.video_index = None
        self.audio_index = None
        self.video_ids = None
        self.audio_ids = None
        self.multimodal_meta = {}
        self.multimodal_id_to_row = {}
        self.name_to_global_id = {}
        self.multimodal_ready = False
        self.token_store_path = token_store_path
        self._token_store = None
        if self.multimodal_index_dir:
            self._load_multimodal_index()

    @staticmethod
    def _first_existing_path(candidates, expect_dir=False):
        for path in candidates:
            if not path:
                continue
            if expect_dir and os.path.isdir(path):
                return path
            if not expect_dir and os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _normalize_vec(vec):
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32)

    # ══════════════════════════════════════════════════════
    # Loading
    # ══════════════════════════════════════════════════════

    def _load_semantic_index(self):
        """加载 Channel A 的 E5 编码器、FAISS 索引以及 metadata"""
        metadata_path = os.path.join(self.semantic_index_dir, "metadata.jsonl")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f if line.strip()]

        index_path = self._first_existing_path([
            os.path.join(self.semantic_index_dir, "semantic_faiss.index"),
            os.path.join(self.semantic_index_dir, "e5_semantic.index"),
        ])
        if index_path is None:
            raise FileNotFoundError(
                f"No semantic FAISS index found under {self.semantic_index_dir}. "
                "Expected semantic_faiss.index or e5_semantic.index"
            )
        self.semantic_faiss = faiss.read_index(index_path)

        self.e5_model_name = "intfloat/multilingual-e5-base"
        try:
            with open(os.path.join(self.semantic_index_dir, "build_config.json"), "r") as f:
                self.e5_model_name = json.load(f).get("model_name", self.e5_model_name)
        except (OSError, json.JSONDecodeError):
            print(f"Warning: Could not read build_config.json, using default {self.e5_model_name}")

        resolved_path = self._resolve_local_model_path(self.e5_model_name)
        print(f"Loading E5 Model: {resolved_path}...")
        self.encoder = SentenceTransformer(resolved_path)

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> str:
        """尝试将模型名解析为本地路径，找不到则原样返回（让 sentence-transformers 走在线下载）。"""
        if os.path.isdir(model_name):
            return os.path.abspath(model_name)
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        project_root = os.path.join(os.path.dirname(__file__), "..")
        candidates = [
            short_name,
            os.path.join(os.environ.get("AFFECTAGENT_MODEL_ROOT", "models"), short_name),
            os.path.join("models", short_name),
            os.path.join(project_root, short_name),
            os.path.join(project_root, "models", short_name),
        ]
        for path in candidates:
            if os.path.isdir(path):
                resolved = os.path.abspath(path)
                print(f"  -> Resolved E5 to local path: {resolved}")
                return resolved
        return model_name

    def _load_multimodal_index(self):
        """加载 Channel B 的多模态 FAISS 索引、ID 映射、metadata 和 token store。"""
        root = self.multimodal_index_dir
        index_dir = self._first_existing_path([os.path.join(root, "index"), root], expect_dir=True)
        vector_dir = self._first_existing_path([os.path.join(root, "vector"), root], expect_dir=True)
        meta_dir = self._first_existing_path([os.path.join(root, "meta"), root], expect_dir=True)
        token_dir = self._first_existing_path([os.path.join(root, "token_store"), root], expect_dir=True)
        if index_dir is None or vector_dir is None or meta_dir is None:
            raise FileNotFoundError(
                f"Invalid multimodal index dir: {root}. Expected subdirs index/, vector/, meta/"
            )

        video_index_path = os.path.join(index_dir, "video.index")
        audio_index_path = os.path.join(index_dir, "audio.index")
        video_ids_path = os.path.join(vector_dir, "video_ids.npy")
        audio_ids_path = os.path.join(vector_dir, "audio_ids.npy")
        meta_path = self._first_existing_path([
            os.path.join(meta_dir, "metadata.jsonl"),
            os.path.join(root, "metadata.jsonl"),
        ])

        if os.path.isfile(video_index_path):
            self.video_index = faiss.read_index(video_index_path)
        if os.path.isfile(audio_index_path):
            self.audio_index = faiss.read_index(audio_index_path)
        if os.path.isfile(video_ids_path):
            self.video_ids = np.load(video_ids_path)
        if os.path.isfile(audio_ids_path):
            self.audio_ids = np.load(audio_ids_path)
        if meta_path is not None:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    gid = item.get("global_id")
                    if gid is not None:
                        self.multimodal_meta[int(gid)] = item
        self.name_to_global_id = {
            item["name"]: gid for gid, item in self.multimodal_meta.items() if item.get("name")
        }

        if token_dir is not None:
            id_to_row_path = os.path.join(token_dir, "id_to_row.json")
            if os.path.isfile(id_to_row_path):
                with open(id_to_row_path, "r", encoding="utf-8") as f:
                    self.multimodal_id_to_row = {int(k): v for k, v in json.load(f).items()}
            if self.token_store_path is None:
                self.token_store_path = os.path.join(token_dir, "tokens.h5")

        self.multimodal_ready = bool(
            (self.video_index is not None and self.video_ids is not None)
            or (self.audio_index is not None and self.audio_ids is not None)
        )

    # ══════════════════════════════════════════════════════
    # Cognitive channel (E_cog)
    # ══════════════════════════════════════════════════════

    def retrieve_channel_A(self, queries_dict, top_k=3, exclude_sample_id=None, groups: Optional[Sequence[str]] = None):
        """
        Retrieve Top-K cognitive candidates for each Planner query.

        Args:
            queries_dict: {"support": {"query_text": ...}, "confusion": {...}, "counter": {...}}
            top_k: candidates per group (the K items judged by the Evidence Filter)
            groups: evidence groups to retrieve (used by the evidence-structure ablations)

        Returns:
            {"support": [{"id", "global_id", "text", "label_hint", "score", ...}], ...}
        """
        if self.semantic_faiss is None or self.encoder is None:
            raise RuntimeError(
                "Cognitive semantic index is not loaded. Pass semantic_index_dir to DualChannelRetriever."
            )
        groups = tuple(groups) if groups is not None else GROUPS
        candidates = {group: [] for group in groups}
        if not self.metadata:
            return candidates

        requests = []
        for q_type, q_info in queries_dict.items():
            q_type = "support" if q_type == "primary" else q_type
            query_text = str((q_info or {}).get("query_text", "")).strip()
            if q_type in candidates and query_text:
                requests.append((q_type, query_text))
        if not requests:
            return candidates

        # E5 编码约定：query 端加 "query: " 前缀；三条查询一次编码
        embeddings = self.encoder.encode(
            [f"query: {text}" for _, text in requests], normalize_embeddings=True
        )
        search_k = min(max(top_k * 3, top_k + 1), len(self.metadata))
        scores, indices = self.semantic_faiss.search(np.asarray(embeddings, dtype=np.float32), search_k)

        for row, (q_type, _) in enumerate(requests):
            seen = set()
            for score, idx in zip(scores[row], indices[row]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                name = meta.get("name")
                if (exclude_sample_id and name == exclude_sample_id) or name in seen:
                    continue
                seen.add(name)
                description = meta.get("description", meta.get("reason", ""))
                candidates[q_type].append({
                    "id": name,
                    "global_id": meta.get("global_id", int(idx)),
                    "label_hint": meta.get("ovlabel", ""),
                    "score": float(score),
                    # 检索出的文本证据包含 subtitle 和 description
                    "text": f"{meta.get('subtitle', '')}。{description}".strip("。"),
                    "subquery_type": q_type,
                    "description": description,
                })
                if len(candidates[q_type]) >= top_k:
                    break
        return candidates

    # ══════════════════════════════════════════════════════
    # Perceptual channel (E_perc)
    # ══════════════════════════════════════════════════════

    def _search_single_modality(self, index, id_array, query_vec, top_k):
        if index is None or id_array is None or query_vec is None:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, indices = index.search(q, min(top_k, index.ntotal))
        return [
            {"global_id": int(id_array[idx]), "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def retrieve_channel_B_candidates(self, current_sample_features, top_k=1, exclude_sample_id=None) -> List[Dict]:
        """Rank library samples by mean audiovisual cosine similarity over the available modalities."""
        if not self.multimodal_ready:
            raise RuntimeError(
                "Channel B multimodal index is not loaded. Please pass a valid multimodal_index_dir "
                "built by retrieval/faiss/build_mercaptionplus_faiss.py"
            )
        if isinstance(current_sample_features, dict):
            video_vec = self._normalize_vec(current_sample_features.get("video"))
            audio_vec = self._normalize_vec(current_sample_features.get("audio"))
        else:
            video_vec, audio_vec = self._normalize_vec(current_sample_features), None
        queries = [
            (self.video_index, self.video_ids, video_vec),
            (self.audio_index, self.audio_ids, audio_vec),
        ]
        active = [query for query in queries if query[0] is not None and query[2] is not None]
        if not active:
            return []

        score_map: Dict[int, float] = {}
        for index, ids, vec in active:
            for result in self._search_single_modality(index, ids, vec, top_k * 3 + 1):
                gid = result["global_id"]
                score_map[gid] = score_map.get(gid, 0.0) + result["score"] / len(active)

        ranked = []
        for gid, score in sorted(score_map.items(), key=lambda item: -item[1]):
            name = self.multimodal_meta.get(gid, {}).get("name") or str(gid)
            if exclude_sample_id and name == exclude_sample_id:
                continue
            ranked.append({"name": name, "global_id": gid, "score": score})
            if len(ranked) >= top_k:
                break
        return ranked

    def retrieve_channel_B(self, current_sample_features, top_k=1, exclude_sample_id=None):
        """Channel B: names of the Top-K perceptually closest library samples."""
        return [
            item["name"]
            for item in self.retrieve_channel_B_candidates(current_sample_features, top_k, exclude_sample_id)
        ]

    def _open_token_store(self):
        if self._token_store is None:
            if not self.token_store_path or not os.path.isfile(self.token_store_path):
                return None
            import h5py

            self._token_store = h5py.File(self.token_store_path, "r")
        return self._token_store

    @staticmethod
    def _trim_padding(tokens: np.ndarray) -> np.ndarray:
        # The token store zero-pads every sequence to the longest one in the library.
        keep = np.abs(tokens).sum(axis=-1) > 0
        return tokens[keep] if keep.any() else tokens[:1]

    def load_partner_tokens(self, global_id: int) -> Dict[str, Optional[torch.Tensor]]:
        """Load E_perc^v / E_perc^a token sequences [1, T, D] of one library sample."""
        tokens = {"video": None, "audio": None}
        store = self._open_token_store()
        row_info = self.multimodal_id_to_row.get(int(global_id))
        if store is None or row_info is None:
            return tokens
        for key in ("video", "audio"):
            row = row_info if isinstance(row_info, int) else row_info.get(f"{key}_row", -1)
            dataset_name = f"{key}_tokens"
            if dataset_name in store and row is not None and row >= 0:
                array = self._trim_padding(np.asarray(store[dataset_name][row], dtype=np.float32))
                tokens[key] = torch.from_numpy(array).unsqueeze(0)
        return tokens


# Backward-compatible pre-release class name.
DoubleChannelRetriever = DualChannelRetriever
