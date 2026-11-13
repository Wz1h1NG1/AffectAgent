"""Role-specific multimodal prompts and robust structured-output parsers.

These builders are the single source of truth for prompt text: role-specific SFT, MAPPO and
evaluation all render prompts through them (see ``orchestrator.*_prompt_ids``).
"""

import json
import re
from typing import Dict, Iterable, List, Optional, Sequence

from .labels import normalize_label, split_labels
from .schemas import (
    GROUPS,
    EmotionGeneratorOutput,
    EvidenceFilterOutput,
    QueryPlannerOutput,
)


def _resolve_video_patch_token(face_or_frame: str = "face") -> str:
    return "<FaceHere>" if "face" in face_or_frame else "<FrameHere>"


def render_prompt(messages: List[Dict]) -> str:
    """Render chat messages in AffectGPT's ``###Human: ... ###Assistant:`` format."""

    parts = []
    for message in messages:
        if message["role"] == "system":
            parts.append(f"System: {message['content']}")
        elif message["role"] == "user":
            parts.append(f"###Human: {message['content']}")
    parts.append("###Assistant:")
    return "\n".join(parts)


def _clip(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


# ════════════════════════════════════════════════════════════════════
# Query Planner
# ════════════════════════════════════════════════════════════════════

QUERY_PLANNER_SYSTEM_ZH = (
    "你是AffectAgent的多模态查询规划器。请同时观察字幕、视频和音频，并结合候选标签，"
    "生成支持、易混淆和反证三类认知检索查询。查询应描述可检索的语言、表情、动作、语气和"
    "情感线索，不能只输出抽象意图。严格输出JSON。"
)
QUERY_PLANNER_SYSTEM_EN = (
    "You are AffectAgent's multimodal Query Planner. Observe the subtitle, video, and audio "
    "together with the candidate labels, then formulate support, confusion, and counter "
    "cognitive retrieval queries. Describe retrievable linguistic, facial, behavioral, vocal, "
    "and affective cues rather than an abstract intent. Output JSON only."
)
QUERY_OUTPUT_SCHEMA = """{
  "support":   {"query_text": "...", "target_label": "..."},
  "confusion": {"query_text": "...", "confusing_label": "..."},
  "counter":   {"query_text": "...", "opposite_label": "..."}
}"""

# Label field carried by each query group; used by the Score_label counterfactual.
QUERY_LABEL_KEYS = {
    "support": "target_label",
    "confusion": "confusing_label",
    "counter": "opposite_label",
}


def build_query_planner_messages(
    subtitle: str,
    candidate_labels: List[str],
    lang: str = "en",
    face_or_frame: str = "face",
) -> List[Dict]:
    labels = ", ".join(candidate_labels)
    video_token = _resolve_video_patch_token(face_or_frame)
    if lang == "zh":
        content = (
            f'字幕: "{subtitle}"\n'
            f"视觉: <Video>{video_token}</Video>\n"
            f"听觉: <Audio><AudioHere></Audio>\n"
            f"候选标签: {labels}\n\n请输出如下JSON:\n{QUERY_OUTPUT_SCHEMA}"
        )
        system = QUERY_PLANNER_SYSTEM_ZH
    else:
        content = (
            f'Subtitle: "{subtitle}"\n'
            f"Visual: <Video>{video_token}</Video>\n"
            f"Audio: <Audio><AudioHere></Audio>\n"
            f"Candidate labels: {labels}\n\nOutput this JSON schema:\n{QUERY_OUTPUT_SCHEMA}"
        )
        system = QUERY_PLANNER_SYSTEM_EN
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_q_messages(
    subtitle: str,
    candidate_labels: List[str],
    lang: str = "en",
    face_or_frame: str = "face",
) -> List[Dict]:
    """Deprecated function name retained for existing integrations."""

    return build_query_planner_messages(subtitle, candidate_labels, lang, face_or_frame)


# ════════════════════════════════════════════════════════════════════
# Evidence Filter: one Yes/No decision per retrieved candidate (action a_F)
# ════════════════════════════════════════════════════════════════════

EVIDENCE_FILTER_SYSTEM_ZH = (
    "你是AffectAgent的多模态证据过滤器。请用当前样本的字幕、视频和音频逐条交叉核验检索到的认知证据，"
    "对每条候选证据判断它是否与当前样本一致、能为情感判断提供可靠支撑，回答Yes或No。"
    "严格按候选顺序输出JSON。"
)
EVIDENCE_FILTER_SYSTEM_EN = (
    "You are AffectAgent's multimodal Evidence Filter. Cross-verify every retrieved cognitive "
    "candidate against the current subtitle, video, and audio, and decide for each candidate "
    "whether it is consistent with the current sample and reliable for the emotion decision "
    "(Yes or No). Output the decisions as JSON, in candidate order."
)


def _active_groups(candidates: Dict[str, list]) -> List[str]:
    """Evidence groups present in ``candidates`` (``primary`` is accepted as an alias of ``support``)."""

    return [group for group in GROUPS if group in candidates or (group == "support" and "primary" in candidates)]


def _canonical_candidates(candidates: Dict[str, list]) -> Dict[str, list]:
    aliases = {"support": "primary", "confusion": "contrast"}
    canonical = {}
    for group in _active_groups(candidates):
        items = candidates.get(group)
        if items is None:
            items = candidates.get(aliases.get(group, ""))
        canonical[group] = list(items or [])
    return canonical


def _format_candidates_text(candidates: Dict[str, list], lang: str, max_chars: int) -> str:
    """List every candidate with its library emotion annotation, which the Filter checks
    against the current sample."""

    chunks = []
    label_key = "标签" if lang == "zh" else "labels"
    for group, items in _canonical_candidates(candidates).items():
        rows = [f"【{group}组】" if lang == "zh" else f"[{group.capitalize()} group]"]
        for index, item in enumerate(items, 1):
            rows.append(
                f'{index}. id={item.get("id", "")}, {label_key}={item.get("label_hint", "")}, '
                f'"{_clip(item.get("text", ""), max_chars)}"'
            )
        if not items:
            rows.append("(空)" if lang == "zh" else "(empty)")
        chunks.append("\n".join(rows))
    return "\n".join(chunks)


def filter_output_schema(candidates: Dict[str, list]) -> str:
    fields = []
    for group, items in _canonical_candidates(candidates).items():
        fields.append(f'"{group}": [' + ", ".join('"Yes|No"' for _ in items) + "]")
    return "{" + ", ".join(fields) + "}"


def format_filter_target(decisions: Dict[str, Sequence[bool]]) -> str:
    """SFT target for the Evidence Filter in the layout ``parse_evidence_filter_output`` reads."""

    return json.dumps(
        {group: ["Yes" if keep else "No" for keep in values] for group, values in decisions.items()},
        ensure_ascii=False,
    )


def build_evidence_filter_messages(
    subtitle: str,
    candidates: Dict[str, list],
    lang: str = "en",
    face_or_frame: str = "face",
    max_evidence_chars: int = 400,
) -> List[Dict]:
    video_token = _resolve_video_patch_token(face_or_frame)
    candidates_text = _format_candidates_text(candidates, lang, max_evidence_chars)
    schema = filter_output_schema(candidates)
    if lang == "zh":
        content = (
            f'当前样本字幕: "{subtitle}"\n'
            f"视觉: <Video>{video_token}</Video>\n听觉: <Audio><AudioHere></Audio>\n\n"
            f"候选认知证据:\n{candidates_text}\n\n对每条候选回答Yes或No，输出JSON:\n{schema}"
        )
        system = EVIDENCE_FILTER_SYSTEM_ZH
    else:
        content = (
            f'Current subtitle: "{subtitle}"\n'
            f"Visual: <Video>{video_token}</Video>\nAudio: <Audio><AudioHere></Audio>\n\n"
            f"Cognitive candidates:\n{candidates_text}\n\n"
            f"Answer Yes or No for every candidate and output JSON:\n{schema}"
        )
        system = EVIDENCE_FILTER_SYSTEM_EN
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


build_s_messages = build_evidence_filter_messages


# ════════════════════════════════════════════════════════════════════
# Emotion Generator: rationale r and open-vocabulary prediction ŷ
# ════════════════════════════════════════════════════════════════════

EMOTION_GENERATOR_SYSTEM_ZH = (
    "你是AffectAgent的多模态情感生成器。结合原始文本、过滤后的认知证据以及经RAAF和MB-MoE"
    "增强的视听表示，先给出解释，再输出最终情感标签。严格输出JSON。"
)
EMOTION_GENERATOR_SYSTEM_EN = (
    "You are AffectAgent's multimodal Emotion Generator. Combine the raw text, the filtered "
    "cognitive evidence, and the audiovisual representation enhanced by RAAF and MB-MoE. Give "
    "your rationale first and then the final emotion prediction. Output JSON only."
)
GENERATOR_INSTRUCTION_EN = (
    "Give a short rationale (at most three sentences), then list all emotion words that describe "
    "the person (usually one to five); prefer words from the candidate labels. "
    'Output JSON: {"reasoning": "...", "prediction": ["<emotion>", "..."]}'
)
GENERATOR_INSTRUCTION_ZH = (
    "先给出不超过三句的简短解释，再列出描述人物情感的全部英文情感词（通常1到5个），优先使用候选标签。"
    '输出JSON: {"reasoning": "...", "prediction": ["<emotion>", "..."]}'
)


def _iter_evidence(value: object) -> Iterable[object]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _item_value(item: object, key: str, default: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get(key, default))
    return str(getattr(item, key, default))


def _format_evidence_text(evidence_items: Dict[str, object], lang: str, max_chars: int) -> str:
    rows = []
    for role in GROUPS:
        values = evidence_items.get(role)
        # Accept the old implementation term ``contrast`` during migration.
        if role == "confusion" and values is None:
            values = evidence_items.get("contrast")
        for index, item in enumerate(_iter_evidence(values), 1):
            text = _clip(_item_value(item, "text"), max_chars)
            hint = _item_value(item, "label_hint")
            if lang == "zh":
                rows.append(f'[{role}证据 {index}{", " + hint if hint else ""}]\n"{text}"')
            else:
                rows.append(f'[{role.capitalize()} evidence {index}{", " + hint if hint else ""}]\n"{text}"')
    if not rows:
        rows.append("[无可靠证据]" if lang == "zh" else "[No reliable evidence]")
    return "\n".join(rows)


def build_emotion_generator_messages(
    subtitle: str,
    evidence_items: Dict[str, object],
    candidate_labels: List[str],
    lang: str = "en",
    face_or_frame: str = "face",
    max_evidence_chars: int = 400,
) -> List[Dict]:
    evidence = _format_evidence_text(evidence_items, lang, max_evidence_chars)
    labels = ", ".join(candidate_labels)
    video_token = _resolve_video_patch_token(face_or_frame)
    if lang == "zh":
        content = (
            f"{evidence}\n[待分析样本]\n字幕: \"{subtitle}\"\n"
            f"视听融合表示: <Video>{video_token}</Video><Audio><AudioHere></Audio>\n"
            f"候选标签: {labels}\n\n{GENERATOR_INSTRUCTION_ZH}"
        )
        system = EMOTION_GENERATOR_SYSTEM_ZH
    else:
        content = (
            f"{evidence}\n[Current sample]\nSubtitle: \"{subtitle}\"\n"
            f"Fused audiovisual representation: <Video>{video_token}</Video><Audio><AudioHere></Audio>\n"
            f"Candidate labels: {labels}\n\n{GENERATOR_INSTRUCTION_EN}"
        )
        system = EMOTION_GENERATOR_SYSTEM_EN
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


build_g_messages = build_emotion_generator_messages


def format_generator_target(reasoning: str, labels: Sequence[str]) -> str:
    """SFT target for the Emotion Generator in the layout ``parse_emotion_generator_output`` reads."""

    return json.dumps({"reasoning": reasoning, "prediction": list(labels)}, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════
# Parsers
# ════════════════════════════════════════════════════════════════════

def _clean_json_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    # Keep the outermost JSON object when the model adds text around it.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return cleaned


def parse_query_planner_output(raw_text: str) -> QueryPlannerOutput:
    output = QueryPlannerOutput(raw_text=raw_text)
    try:
        data = json.loads(_clean_json_text(raw_text))
        output.support = data.get("support", data.get("primary", {}))
        output.confusion = data.get("confusion", data.get("contrast", {}))
        output.counter = data.get("counter", {})
        # Normalize pre-release field names while keeping old SFT outputs loadable.
        if "confusing_label" not in output.confusion and "contrast_label" in output.confusion:
            output.confusion["confusing_label"] = output.confusion["contrast_label"]
        if "opposite_label" not in output.counter and "counter_direction" in output.counter:
            output.counter["opposite_label"] = output.counter["counter_direction"]
        output.valid = all(
            isinstance(getattr(output, group), dict)
            and bool(getattr(output, group).get("query_text"))
            for group in GROUPS
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        output.valid = False
    return output


parse_q_output = parse_query_planner_output


def _to_decision(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "keep", "是"}:
        return True
    if text in {"no", "n", "false", "0", "drop", "否"}:
        return False
    return None


def parse_evidence_filter_output(raw_text: str, candidates: Dict[str, list]) -> EvidenceFilterOutput:
    """Parse per-candidate Yes/No decisions.

    Accepted layouts per group: ``["Yes", "No", ...]`` in candidate order, ``{"<id>": "Yes"}``,
    or a single ``{"id": "<id>"}`` (that ID is kept, the others dropped).
    The output is valid only if every candidate received a decision; groups that could not be
    read completely are left out of ``decisions`` and ``kept_ids``.
    """

    output = EvidenceFilterOutput(raw_text=raw_text)
    try:
        data = json.loads(_clean_json_text(raw_text))
    except (json.JSONDecodeError, TypeError):
        data = None
    if not isinstance(data, dict):
        output.valid = False
        return output

    valid = True
    for group, items in _canonical_candidates(candidates).items():
        ids = [str(item.get("id", "")) for item in items]
        source = data.get(group, data.get("primary") if group == "support" else None)
        decisions: List[Optional[bool]] = [None] * len(items)
        if isinstance(source, list):
            for index, value in enumerate(source[:len(items)]):
                decisions[index] = _to_decision(value)
        elif isinstance(source, dict) and set(source) == {"id"}:
            decisions = [candidate_id == str(source["id"]) for candidate_id in ids]
            if str(source["id"]) not in ids:  # an identifier outside the group is malformed
                valid = False
        elif isinstance(source, dict):
            for index, candidate_id in enumerate(ids):
                if candidate_id in source:
                    decisions[index] = _to_decision(source[candidate_id])
        if any(decision is None for decision in decisions):
            valid = False
            continue
        output.decisions[group] = list(decisions)
        output.kept_ids[group] = [candidate_id for candidate_id, keep in zip(ids, decisions) if keep]
    output.valid = valid
    return output


parse_s_output = parse_evidence_filter_output


NEGATION = re.compile(r"\b(not|no|never|without|lack|less)\b")


def _snap_to_vocabulary(label: str, vocabulary: Dict[str, str]) -> str:
    """Map a phrase such as "happy emotion" to the vocabulary word it contains."""

    if not vocabulary or label in vocabulary or NEGATION.search(label):
        return label
    best, best_position = label, len(label) + 1
    for key, word in vocabulary.items():
        match = re.search(rf"\b{re.escape(key)}\b", label)
        if match and match.start() < best_position:
            best, best_position = word, match.start()
    return best


def _regex_prediction(raw_text: str) -> str:
    match = re.search(r'"prediction"\s*:\s*(\[[^\]]*\]|"[^"]*")', str(raw_text or ""))
    if not match:
        return ""
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ""
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def parse_emotion_generator_output(
    raw_text: str,
    candidate_labels: Sequence[str] = (),
) -> EmotionGeneratorOutput:
    """Parse the rationale and every predicted emotion word (open vocabulary)."""

    output = EmotionGeneratorOutput(raw_text=raw_text)
    prediction: object = ""
    try:
        data = json.loads(_clean_json_text(raw_text))
        prediction = data.get("prediction", "")
        if isinstance(prediction, dict):
            prediction = prediction.get("label", prediction.get("text", ""))
        output.reasoning = str(data.get("reasoning", ""))
        try:
            output.confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            output.confidence = 0.0
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Truncated generations often still contain a well-formed prediction field.
        prediction = _regex_prediction(raw_text)

    vocabulary = {normalize_label(label): str(label) for label in candidate_labels}
    labels = []
    for label in split_labels(prediction):
        label = _snap_to_vocabulary(label, vocabulary)
        if label not in labels:
            labels.append(label)
    output.labels = labels
    output.prediction = ", ".join(labels)
    output.valid = bool(labels)
    if not 0.0 <= output.confidence <= 1.0:
        output.confidence = 0.5
    return output


parse_g_output = parse_emotion_generator_output
