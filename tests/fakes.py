"""Tiny CPU stand-ins for AffectGPT, its tokenizer and the retriever, used by the unit tests."""

import contextlib
import re
from types import SimpleNamespace

import torch
import torch.nn as nn

from affectagent.orchestrator import AffectAgentPipeline

SPECIAL_TOKENS = ["<ImageHere>", "<AudioHere>", "<FrameHere>", "<FaceHere>", "<MultiHere>"]
VOCAB_SIZE = 264
HIDDEN = 16


class FakeTokenizer:
    """Character-level tokenizer; the AffectGPT placeholders are single tokens."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {token: 256 + index for index, token in enumerate(SPECIAL_TOKENS)}
        self.pattern = re.compile("(" + "|".join(map(re.escape, SPECIAL_TOKENS)) + ")")

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text):
        ids = []
        for part in self.pattern.split(text):
            if part in self.vocab:
                ids.append(self.vocab[part])
            else:
                ids.extend(min(ord(char), 255) for char in part)
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids if 2 <= int(i) < 256)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()


class FakeLM(nn.Module):
    """Causal toy LM exposing ``model.model.embed_tokens`` like a PEFT-wrapped HF model."""

    def __init__(self):
        super().__init__()
        self.model = _Wrapper()
        self.base = nn.Linear(HIDDEN, HIDDEN)
        self.adapter = nn.Linear(HIDDEN, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        for module in (self.model, self.base, self.lm_head):
            module.requires_grad_(False)
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.adapter_enabled = True

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False, use_cache=None, return_dict=True):
        steps = torch.arange(1, inputs_embeds.size(1) + 1, device=inputs_embeds.device).view(1, -1, 1)
        hidden = torch.cumsum(inputs_embeds, dim=1) / steps  # causal prefix mean
        update = self.base(hidden)
        if self.adapter_enabled:
            update = update + self.adapter(hidden)
        hidden = torch.tanh(update)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(inputs_embeds, hidden))


class FakeAffectGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.llama_model = FakeLM()


class FakeChat:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.device = "cpu"
        self.model = FakeAffectGPT()
        self.num_video_query_token = 1
        self.num_audio_query_token = 1
        self.num_multi_query_token = 1
        self.num_image_query_token = 1
        self.stopping_criteria = None

    def replace_token_for_multimodal(self, prompt):
        return prompt  # one query token per modality

    def postprocess_audio(self, sample_data):
        return None, sample_data.get("audio")

    def postprocess_frame(self, sample_data):
        return None, sample_data.get("video")

    def postprocess_face(self, sample_data):
        return None, None


class FakeRetriever:
    multimodal_ready = True

    def __init__(self):
        generator = torch.Generator().manual_seed(1)
        self.tokens = {
            "video": torch.randn(1, 1, HIDDEN, generator=generator),
            "audio": torch.randn(1, 1, HIDDEN, generator=generator),
        }
        self.multimodal_meta = {7: {"name": "lib7", "ovlabel": "happy", "subtitle": "what a great day"}}
        self.calls = []

    def retrieve_channel_A(self, queries, top_k, exclude_sample_id=None, groups=None):
        self.calls.append(queries)
        return {
            group: [
                {"id": f"{group}{index}", "global_id": index, "text": f"{group} evidence {index}",
                 "label_hint": "happy" if group == "support" else "sad"}
                for index in range(top_k)
            ]
            for group in groups
        }

    def retrieve_channel_B_candidates(self, vectors, top_k=1, exclude_sample_id=None):
        return [{"name": "lib7", "global_id": 7, "score": 0.9}]

    def load_partner_tokens(self, global_id):
        return dict(self.tokens)


PLANNER_RESPONSE = (
    '{"support": {"query_text": "smiling warmly", "target_label": "happy"}, '
    '"confusion": {"query_text": "calm voice", "confusing_label": "neutral"}, '
    '"counter": {"query_text": "crying", "opposite_label": "sad"}}'
)
FILTER_RESPONSE = '{"support": ["Yes", "No"], "confusion": ["No", "Yes"], "counter": ["No", "No"]}'


class ScriptedPipeline(AffectAgentPipeline):
    """Real pipeline whose generation step returns scripted text (prompts are still embedded)."""

    def __init__(self, *args, generator_predictions=("happy", "sad", "happy"), **kwargs):
        super().__init__(*args, **kwargs)
        self.generator_predictions = list(generator_predictions)
        self.generator_calls = 0
        self.prompts = {"q": [], "f": [], "g": []}

    def _act(self, prompt_ids, features, agent):
        with torch.no_grad():
            self.encoder.embed(prompt_ids, features, self.model.llama_model)
        self.prompts[agent].append(self.chat.tokenizer.decode(prompt_ids))
        if agent == "q":
            text = PLANNER_RESPONSE
        elif agent == "f":
            text = FILTER_RESPONSE
        else:
            label = self.generator_predictions[self.generator_calls % len(self.generator_predictions)]
            self.generator_calls += 1
            text = '{"reasoning": "visible cues", "prediction": ["%s"]}' % label
        response_ids = torch.tensor(self.chat.tokenizer.encode(text + "###"), dtype=torch.long)
        return response_ids, text


def sample_data():
    generator = torch.Generator().manual_seed(0)
    return {"video": torch.randn(1, 1, HIDDEN, generator=generator), "audio": torch.randn(1, 1, HIDDEN, generator=generator)}
