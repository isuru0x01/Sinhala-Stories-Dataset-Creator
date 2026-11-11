#!/usr/bin/env python3
from __future__ import annotations
# ==============================================================================
# sinhala_tokenizer_research.py
#
# RESEARCH REPLICATION PIPELINE:
# "Bridging the Linguistic Divide: An Akshara-Preserving Subword Tokenizer for
#  Efficient Sinhala Language Modeling" (Targeted for Q1 Computational Linguistics)
#
# ARCHITECTURAL INNOVATION:
# Akshara-to-Private-Use Mapping (Akshara-as-Atomic-Unit):
# 1. Seeds mapper with the comprehensive finite inventory of Sinhala aksharas (3,890 forms:
#    consonant x vowel-sign and hal combinations, non-virama C+anusvaraya/visargaya, productive
#    Yansaya and Rakaransaya conjuncts, ksha clusters, independent vowels, and joiners; capped to strictly fit
#    within Unicode BMP PUA U+E000 to U+F8FF, capacity 6,400).
#    * Trade-off: Allocating 3,890 atomic base units reserves ~12.2% of a 32k vocabulary budget,
#      meaning slightly fewer learned subword merges than plain BPE at the same nominal size.
# 2. Decomposes any still-unseen complex conjuncts (e.g. non-ya/ra ZWJ forms like ස්වා, ද්වී) greedily into
#    known akshara components before falling back to raw UTF-8 bytes, preventing fertility explosion.
# 3. Runs Byte-Fallback BPE and Unigram LM over the PUA alphabet with Metaspace pre-tokenization
#    and byte fallback (<0x00>..<0xFF> + ByteFallback decoder).
# 4. Merges freely across akshara boundaries to form rich subwords and full words,
#    preserving akshara integrity (AIR is evaluated empirically on held-out text splits).
# 5. Decodes bijectively back to original Sinhala Unicode sequences with zero loss and zero <unk>.
#
# RIGOROUS SCIENTIFIC CONTROLS & ABLATIONS:
# - In-Domain Ablations: Plain Metaspace BPE & Plain Unigram with byte fallback trained on the exact same corpus
#   at matched nominal vocabulary size to isolate the linguistic akshara contribution from domain/corpus effects.
# - Verified Baselines: XLM-RoBERTa, Gemma-2, Qwen-2.5, mT5, Sinhala-GPT2, LLaMA-3 (if token provided).
# - Empirical AIR, BFR & UNK: Akshara Integrity Rate (AIR %) measured across all phonetic aksharas;
#   Byte-Fallback Rate (BFR %) tracks the share of generated tokens that are raw byte tokens (<0xXX>),
#   with Byte-Level BPE models (Qwen, LLaMA) distinguished; UNK (%) reports unknown token failure rates.
# - Offset-Based BPC & BPB: Evaluated across all validation chunks using exact source-text character offsets,
#   eliminating decoder spacing rewrites, boundary space stripping, and split-byte artifacts.
#
# GOOGLE COLAB INSTRUCTIONS:
# 1. Runtime -> Change runtime type -> T4 GPU
# 2. !pip install -q tokenizers transformers datasets sentencepiece torch scipy tabulate
# 3. (Optional for gated models like LLaMA-3): os.environ["HF_TOKEN"] = "your_token"
# 4. !python sinhala_tokenizer_research.py --dataset Isuru0x01/sinhala_stories --sample_size 10000 --vocab_size 32000
# ==============================================================================

import os
import sys
import re
import math
import time
import json
import random
import unicodedata
import argparse
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter

# Safe import validation
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from tokenizers import (
        Tokenizer,
        models,
        normalizers,
        pre_tokenizers,
        decoders,
        trainers,
    )
    from transformers import (
        PreTrainedTokenizerFast,
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
    )
    HAS_CORE_DEPS = True
    IMPORT_ERR = ""
except ImportError as err:
    HAS_CORE_DEPS = False
    IMPORT_ERR = str(err)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ==============================================================================
# 1. LINGUISTIC SPECIFICATION: SINHALA AKSHARA FORMULATION
# ==============================================================================

SINHALA_START = 0x0D80
SINHALA_END = 0x0DFF
ZWJ = "\u200D"      # Zero-Width Joiner (used for yansaya, rakaransaya, bandi akuru)
ZWNJ = "\u200C"     # Zero-Width Non-Joiner
VIRAMA = "\u0DCA"   # Hal-kirima / Al-lakuna

# Complete, linguistically grounded Sinhala Akshara (Grapheme Cluster) Regex:
# 1. Independent Vowel: [\u0D85-\u0D96] + optional [anusvaraya \u0D82 / visargaya \u0D83]
# 2. Base Consonant: [\u0D9A-\u0DC6]
#    - Optional conjunct clusters via ZWJ: (?:\u0DCA\u200D[\u0D9A-\u0DC6])*
#    - Followed by EITHER:
#        a) Pure virama / hal-kirima: \u0DCA (e.g. ක්, න්, ත්)
#        b) OR dependent vowel signs (pili) [\u0DCF-\u0DDF\u0DF2\u0DF3]* + optional anusvaraya/visargaya
SINHALA_AKSHARA_PATTERN = (
    r"(?:[\u0D85-\u0D96][\u0D82\u0D83]?|"
    r"[\u0D9A-\u0DC6](?:\u0DCA\u200D[\u0D9A-\u0DC6])*(?:\u0DCA|[\u0DCF-\u0DDF\u0DF2\u0DF3]*[\u0D82\u0D83]?))"
)
SINHALA_AKSHARA_REGEX = re.compile(SINHALA_AKSHARA_PATTERN)


def normalize_sinhala_text(text: str) -> str:
    """
    Performs canonical Unicode NFC normalization while strictly preserving
    ZWJ conjuncts, combining marks, and narrative ellipses (.. or ...).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_akshara_spans(text: str) -> List[Tuple[int, int, str]]:
    """Extracts all phonetic akshara spans (start, end, substring) from text."""
    return [(m.start(), m.end(), m.group(0)) for m in SINHALA_AKSHARA_REGEX.finditer(text)]


def generate_sinhala_akshara_inventory() -> List[str]:
    """
    Generates the comprehensive finite inventory of Sinhala aksharas (3,890 forms):
    - Independent vowels (18 forms) + anusvaraya/visargaya (36 forms) = 54 forms
    - Base consonants (41 forms) x all 18 pili/vowel-signs + hal/virama (19 forms) = 779 forms
    - Non-virama consonant + pili + anusvaraya (ං) / visargaya (ඃ) = 1,476 forms
    - Productive Yansaya conjuncts (C + virama + ZWJ + ya + pili) = 779 forms
    - Productive Rakaransaya conjuncts (C + virama + ZWJ + ra + pili) = 779 forms
    - Bandi akuru: ksha (C + virama + ZWJ + sha + pili) = 19 forms
    - Standalone anusvaraya (ං), visargaya (ඃ), Zero-Width Joiner (ZWJ), and Non-Joiner (ZWNJ) = 4 forms
    Total: exactly 3,890 forms (safely bounded under 6,400 BMP PUA slots).
    Impossible combinations (virama + anusvaraya, e.g. ක්ං) are strictly excluded.
    Unseen complex conjuncts (e.g. non-ya/ra forms like ස්වා, ද්වී) decompose greedily into these
    known components before raw byte fallback.
    """
    inventory = set()

    # 1. Independent vowels (18 forms) + anusvaraya / visargaya
    for cp in range(0x0D85, 0x0D97):
        v = chr(cp)
        inventory.add(v)
        inventory.add(v + "\u0D82")
        inventory.add(v + "\u0D83")

    # 2. Base consonants x pili + hal
    consonants = [chr(cp) for cp in range(0x0D9A, 0x0DC7) if unicodedata.category(chr(cp)) != "Cn"]
    pili_list = [
        "", "\u0DCA", "\u0DCF", "\u0DD0", "\u0DD1", "\u0DD2", "\u0DD3", "\u0DD4",
        "\u0DD6", "\u0DD8", "\u0DF2", "\u0DDF", "\u0DF3", "\u0DD9", "\u0DDA",
        "\u0DDB", "\u0DDC", "\u0DDD", "\u0DDE"
    ]

    for c in consonants:
        # Base consonant + pili
        for p in pili_list:
            ak = c + p
            inventory.add(ak)
            # Anusvaraya and visargaya only attach to non-virama vowel-signs / base consonant (p != "\u0DCA")
            if p != "\u0DCA":
                inventory.add(ak + "\u0D82")
                inventory.add(ak + "\u0D83")

        # Productive Yansaya conjuncts (C + virama + ZWJ + ya + pili)
        for p in pili_list:
            inventory.add(c + "\u0DCA\u200D\u0DBA" + p)

        # Productive Rakaransaya conjuncts (C + virama + ZWJ + ra + pili)
        for p in pili_list:
            inventory.add(c + "\u0DCA\u200D\u0DBB" + p)

    # Bandi akuru: ksha (ක්\u200Dෂ + pili)
    for p in pili_list:
        inventory.add("\u0D9A\u0DCA\u200D\u0DC2" + p)

    inventory.add("\u0D82")
    inventory.add("\u0D83")
    inventory.add("\u200D")
    inventory.add("\u200C")

    return sorted(list(inventory))


# ==============================================================================
# 2. AKSHARA-AS-ATOMIC-UNIT (PUA MAPPING & BYTE-FALLBACK ENGINE)
# ==============================================================================

class AksharaPUAMapper:
    """
    Bijectively maps distinct Sinhala phonetic Aksharas to single Unicode
    Private Use Area (PUA) codepoints starting at U+E000.
    Seeded with the complete systematic inventory and decomposes unseen aksharas
    into known sub-components before raw byte fallback.
    """

    def __init__(self, akshara_list: Optional[List[str]] = None):
        self.akshara2pua: Dict[str, str] = {}
        self.pua2akshara: Dict[str, str] = {}
        self.fit(akshara_list)

    def fit(self, akshara_list: Optional[List[str]] = None):
        """Builds bijection table seeded with full finite inventory plus corpus aksharas."""
        unique_aksharas = set(generate_sinhala_akshara_inventory())
        if akshara_list:
            unique_aksharas.update(akshara_list)

        sorted_aksharas = sorted(list(unique_aksharas))
        max_pua_capacity = 0xF8FF - 0xE000 + 1  # Exactly 6,400 codepoints in Unicode BMP PUA
        assert len(sorted_aksharas) <= max_pua_capacity, (
            f"PUA alphabet size ({len(sorted_aksharas):,}) exceeds Unicode BMP Private Use Area "
            f"capacity (U+E000 to U+F8FF: {max_pua_capacity:,} codepoints). "
            f"Preventing codepoint spillage into CJK compatibility area."
        )
        self.akshara2pua = {a: chr(0xE000 + i) for i, a in enumerate(sorted_aksharas)}
        self.pua2akshara = {chr(0xE000 + i): a for i, a in enumerate(sorted_aksharas)}
        print(f"[+] Fitted Akshara PUA Alphabet: {len(self.akshara2pua):,} atomic aksharas mapped (Capacity: {max_pua_capacity:,}).")

    def text_to_pua(self, text: str) -> str:
        """Converts known Sinhala aksharas to PUA characters with component fallback."""
        pua_text, _ = self.text_to_pua_with_spans(text)
        return pua_text

    def text_to_pua_with_spans(self, text: str) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Converts text to PUA representation while tracking the exact (orig_start, orig_end)
        character span in the original text for each character in the returned PUA string.
        Unseen complex aksharas are decomposed greedily into known components before byte fallback.
        """
        pua_chars: List[str] = []
        pua_spans: List[Tuple[int, int]] = []
        cursor = 0
        for m in SINHALA_AKSHARA_REGEX.finditer(text):
            m_s, m_e = m.start(), m.end()
            while cursor < m_s:
                pua_chars.append(text[cursor])
                pua_spans.append((cursor, cursor + 1))
                cursor += 1
            akshara = m.group(0)
            if akshara in self.akshara2pua:
                pua_chars.append(self.akshara2pua[akshara])
                pua_spans.append((m_s, m_e))
            else:
                # Decompose unseen complex akshara into known sub-components before raw byte fallback
                sub_idx = 0
                sub_n = len(akshara)
                while sub_idx < sub_n:
                    matched = None
                    match_end = sub_idx
                    for end in range(sub_n, sub_idx, -1):
                        candidate = akshara[sub_idx:end]
                        if candidate in self.akshara2pua:
                            matched = candidate
                            match_end = end
                            break
                    if matched is not None:
                        pua_chars.append(self.akshara2pua[matched])
                        pua_spans.append((m_s + sub_idx, m_s + match_end))
                        sub_idx = match_end
                    else:
                        pua_chars.append(akshara[sub_idx])
                        pua_spans.append((m_s + sub_idx, m_s + sub_idx + 1))
                        sub_idx += 1
            cursor = m_e
        while cursor < len(text):
            pua_chars.append(text[cursor])
            pua_spans.append((cursor, cursor + 1))
            cursor += 1
        return "".join(pua_chars), pua_spans

    def pua_to_text(self, pua_text: str) -> str:
        """Reconstructs original Sinhala Unicode text from PUA characters losslessly."""
        return "".join(self.pua2akshara.get(c, c) for c in pua_text)

    def to_dict(self) -> Dict[str, str]:
        return self.akshara2pua

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "AksharaPUAMapper":
        mapper = cls()
        mapper.akshara2pua = data
        mapper.pua2akshara = {v: k for k, v in data.items()}
        return mapper


def patch_tokenizer_byte_fallback(raw_tokenizer: Tokenizer) -> Tokenizer:
    """
    Ensures that the tokenizer vocabulary explicitly contains all 256 raw byte tokens
    <0x00> to <0xFF>, activates byte_fallback, and installs a ByteFallback+Metaspace
    decoder sequence. This guarantees 100% lossless coverage with zero <unk> tokens.
    Handles both BPE (dict vocab) and Unigram (list of [token, score]) models.
    """
    data = json.loads(raw_tokenizer.to_str())
    model_obj = data["model"]
    vocab_field = model_obj["vocab"]

    if isinstance(vocab_field, dict):
        # BPE model: vocab is a dict of {token: id}
        next_id = len(vocab_field)
        if "▁" not in vocab_field:
            vocab_field["▁"] = next_id
            next_id += 1
        for i in range(256):
            bt = f"<0x{i:02X}>"
            if bt not in vocab_field:
                vocab_field[bt] = next_id
                next_id += 1
    elif isinstance(vocab_field, list):
        # Unigram model: vocab is a list of [token, score]
        existing = {item[0] for item in vocab_field}
        if "▁" not in existing:
            vocab_field.append(["▁", -10.0])
            existing.add("▁")
        for i in range(256):
            bt = f"<0x{i:02X}>"
            if bt not in existing:
                vocab_field.append([bt, -20.0])
                existing.add(bt)

    model_obj["byte_fallback"] = True
    data["decoder"] = {
        "type": "Sequence",
        "decoders": [
            {"type": "ByteFallback"},
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
        ]
    }
    return Tokenizer.from_str(json.dumps(data))


class AksharaPreservingTokenizerWrapper:
    """
    Production-ready wrapper implementing the Akshara-as-Atomic-Unit architecture.
    Provides encode/decode, token reconstruction, and exact offset mappings.
    """

    def __init__(self, raw_tokenizer: Tokenizer, pua_mapper: AksharaPUAMapper, name: str = "Proposed-Akshara-Tokenizer"):
        self.raw_tokenizer = raw_tokenizer
        self.pua_mapper = pua_mapper
        self.name = name

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        pua_text = self.pua_mapper.text_to_pua(text)
        encoding = self.raw_tokenizer.encode(pua_text, add_special_tokens=add_special_tokens)
        return encoding.ids if hasattr(encoding, "ids") else encoding

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        decoded_pua = self.raw_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return self.pua_mapper.pua_to_text(decoded_pua)

    def encode_tokens_and_offsets(self, text: str, add_special_tokens: bool = False) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
        """
        Reconstructs exact token IDs, decoded Sinhala subword tokens, and character offsets
        in the original text such that doc[s:e] strictly matches the decoded token.
        """
        pua_text, pua_spans = self.pua_mapper.text_to_pua_with_spans(text)
        encoding = self.raw_tokenizer.encode(pua_text, add_special_tokens=add_special_tokens)
        ids = encoding.ids
        raw_tokens = encoding.tokens

        offsets = []
        sinhala_tokens = []

        for i, (t, (p_s, p_e)) in enumerate(zip(raw_tokens, encoding.offsets)):
            if p_s == p_e:
                orig_s, orig_e = (0, 0)
            else:
                orig_s = pua_spans[p_s][0]
                orig_e = pua_spans[min(len(pua_spans) - 1, p_e - 1)][1]
                # If token 0 is a standalone prepended '▁' and original text did not start with space
                if i == 0 and not text.startswith(" ") and t == "▁":
                    orig_s, orig_e = (0, 0)

            if t.startswith("<0x") and t.endswith(">"):
                clean_t = t
            else:
                clean_t = self.pua_mapper.pua_to_text(t).replace("▁", " ")
                if i == 0 and clean_t.startswith(" ") and not text.startswith(" "):
                    clean_t = clean_t[1:]

            offsets.append((orig_s, orig_e))
            sinhala_tokens.append(clean_t)

        return ids, sinhala_tokens, offsets

    def __len__(self) -> int:
        return self.raw_tokenizer.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return self.raw_tokenizer.get_vocab_size()

    def save_pretrained(self, save_dir: str):
        """Saves both the underlying Hugging Face tokenizer and the akshara PUA mapping table."""
        os.makedirs(save_dir, exist_ok=True)
        self.raw_tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
        with open(os.path.join(save_dir, "akshara_pua_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.pua_mapper.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_pretrained(cls, load_dir: str, name: str = "Proposed-Akshara-Tokenizer") -> "AksharaPreservingTokenizerWrapper":
        raw_tokenizer = Tokenizer.from_file(os.path.join(load_dir, "tokenizer.json"))
        with open(os.path.join(load_dir, "akshara_pua_map.json"), "r", encoding="utf-8") as f:
            pua_data = json.load(f)
        pua_mapper = AksharaPUAMapper.from_dict(pua_data)
        return cls(raw_tokenizer, pua_mapper, name=name)


# ==============================================================================
# 3. TRAINING ENGINE: PROPOSED & IN-DOMAIN ABLATION MODELS
# ==============================================================================

class SinhalaTokenizerTrainer:
    """
    Trains both:
    1. Proposed Akshara-Preserving Tokenizers (BPE & Unigram on PUA alphabet)
    2. In-Domain Ablation Controls (Plain BPE & Plain Unigram without Akshara mapping)
    on the exact same corpus at the exact same target vocabulary size.
    """

    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2):
        self.requested_vocab_size = vocab_size
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def enforce_vocab_parity(self, pua_mapper: AksharaPUAMapper) -> int:
        """
        Enforces strict nominal vocabulary parity across Proposed and Ablation models.
        The atomic akshara inventory (3,890 forms) sets an absolute lower bound on
        the base alphabet. If the requested vocab_size is below this floor (+ special tokens),
        both Proposed and Ablation controls are automatically configured with the exact same
        effective target vocabulary size.
        """
        alphabet_floor = len(pua_mapper.pua2akshara) + 100  # ~3,990
        if self.requested_vocab_size < alphabet_floor:
            self.vocab_size = alphabet_floor
            print(
                f"\n[!] ENFORCING TARGET VOCABULARY PARITY: Requested vocab_size ({self.requested_vocab_size:,}) "
                f"is below the atomic akshara alphabet requirement ({alphabet_floor:,}).\n"
                f"    -> Automatically adjusting nominal target vocab_size to {self.vocab_size:,} for BOTH "
                f"Proposed models and In-Domain Ablation controls."
            )
        else:
            self.vocab_size = self.requested_vocab_size
        return self.vocab_size

    def train_proposed_bpe(self, corpus: List[str], pua_mapper: AksharaPUAMapper, name: str = "Proposed Akshara-BPE") -> AksharaPreservingTokenizerWrapper:
        """Trains Proposed Akshara-BPE with byte fallback over atomic PUA aksharas."""
        self.enforce_vocab_parity(pua_mapper)
        pua_corpus = [pua_mapper.text_to_pua(doc) for doc in corpus]
        initial_alphabet = ["▁"] + list(pua_mapper.pua2akshara.keys())
        target_vocab_size = self.vocab_size
        print(f"\n[*] Training {name} (Nominal Target Vocab: {target_vocab_size:,})...")

        raw_tokenizer = Tokenizer(models.BPE(byte_fallback=True))
        raw_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

        bpe_trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
            initial_alphabet=initial_alphabet,
            min_frequency=self.min_frequency,
            show_progress=True,
        )
        raw_tokenizer.train_from_iterator(pua_corpus, trainer=bpe_trainer)
        patched_tokenizer = patch_tokenizer_byte_fallback(raw_tokenizer)
        return AksharaPreservingTokenizerWrapper(patched_tokenizer, pua_mapper, name=name)

    def train_proposed_unigram(self, corpus: List[str], pua_mapper: AksharaPUAMapper, name: str = "Proposed Akshara-Unigram") -> AksharaPreservingTokenizerWrapper:
        """Trains Proposed Akshara-Unigram with byte fallback over atomic PUA aksharas."""
        self.enforce_vocab_parity(pua_mapper)
        pua_corpus = [pua_mapper.text_to_pua(doc) for doc in corpus]
        initial_alphabet = ["<s>", "<pad>", "</s>", "<unk>", "▁"] + list(pua_mapper.pua2akshara.keys())
        target_vocab_size = self.vocab_size
        print(f"\n[*] Training {name} (Nominal Target Vocab: {target_vocab_size:,})...")

        raw_tokenizer = Tokenizer(models.Unigram(byte_fallback=True))
        raw_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        raw_tokenizer.decoder = decoders.Metaspace()

        unigram_trainer = trainers.UnigramTrainer(
            vocab_size=target_vocab_size,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
            unk_token="<unk>",
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )
        raw_tokenizer.train_from_iterator(pua_corpus, trainer=unigram_trainer)
        patched_tokenizer = patch_tokenizer_byte_fallback(raw_tokenizer)
        return AksharaPreservingTokenizerWrapper(patched_tokenizer, pua_mapper, name=name)

    def train_ablation_plain_bpe(self, corpus: List[str], name: str = "Ablation: Plain BPE (In-Domain)") -> Any:
        """Trains an in-domain Plain Byte-Fallback BPE control without Akshara mapping at matched nominal vocab."""
        target_vocab_size = self.vocab_size
        print(f"\n[*] Training Ablation Control: {name} (Nominal Target Vocab: {target_vocab_size:,})...")
        raw_tokenizer = Tokenizer(models.BPE(byte_fallback=True))
        raw_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

        bpe_trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
            initial_alphabet=["▁"],
            min_frequency=self.min_frequency,
            show_progress=True,
        )
        raw_tokenizer.train_from_iterator(corpus, trainer=bpe_trainer)
        patched_tokenizer = patch_tokenizer_byte_fallback(raw_tokenizer)

        return PreTrainedTokenizerFast(
            tokenizer_object=patched_tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            model_max_length=2048,
        )

    def train_ablation_plain_unigram(self, corpus: List[str], name: str = "Ablation: Plain Unigram (In-Domain)") -> Any:
        """Trains an in-domain Plain Unigram LM control without Akshara mapping at matched nominal vocab."""
        target_vocab_size = self.vocab_size
        print(f"\n[*] Training Ablation Control: {name} (Nominal Target Vocab: {target_vocab_size:,})...")
        raw_tokenizer = Tokenizer(models.Unigram(byte_fallback=True))
        raw_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        raw_tokenizer.decoder = decoders.Metaspace()

        unigram_trainer = trainers.UnigramTrainer(
            vocab_size=target_vocab_size,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
            unk_token="<unk>",
            initial_alphabet=["<s>", "<pad>", "</s>", "<unk>", "▁"],
            show_progress=True,
        )
        raw_tokenizer.train_from_iterator(corpus, trainer=unigram_trainer)
        patched_tokenizer = patch_tokenizer_byte_fallback(raw_tokenizer)

        return PreTrainedTokenizerFast(
            tokenizer_object=patched_tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            model_max_length=2048,
        )


# ==============================================================================
# 4. VERIFIED BASELINE LOADER
# ==============================================================================

def load_verified_baseline_tokenizers(hf_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads verified multilingual baselines with validated Sinhala coverage.
    Excludes non-Sinhala models (mBERT, IndicBERT) and duplicates (Aya-101 uses mT5).
    """
    baselines = {}
    candidates = [
        ("XLM-RoBERTa (Multilingual SOTA)", "xlm-roberta-base"),
        ("Gemma-2 (SentencePiece 256k)", "google/gemma-2-2b"),
        ("Qwen-2.5 (Multilingual BPE 151k)", "Qwen/Qwen2.5-0.5B"),
        ("mT5 (Multilingual T5)", "google/mt5-small"),
        ("Sinhala-GPT2 (Monolingual HF)", "keshan/Sinhala-GPT2"),
        ("LLaMA-3 (Frontier BPE 128k)", "meta-llama/Meta-Llama-3-8B"),
    ]

    print("\n[*] Loading Verified Multilingual Baselines:")
    for display_name, repo_id in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
            baselines[display_name] = tok
            print(f"  [+] Loaded: {display_name} ({repo_id}) | Vocab: {len(tok):,}")
        except Exception as e:
            err_msg = str(e).split("\n")[0]
            print(f"  [-] Skipped {display_name} ({repo_id}) [{err_msg[:65]}]")

    return baselines


# ==============================================================================
# 5. INTRINSIC & EMPIRICAL LINGUISTIC EVALUATOR
# ==============================================================================

def is_byte_token(tok_str: Any) -> bool:
    """Detects raw byte-fallback tokens (e.g. <0x0A>, <0xFF>) across tokenizers."""
    if not isinstance(tok_str, str):
        return False
    if tok_str.startswith("<0x") and tok_str.endswith(">") and len(tok_str) == 6:
        try:
            int(tok_str[3:5], 16)
            return True
        except ValueError:
            return False
    return False


class TokenizerEvaluator:
    """
    Calculates Information-Theoretic and Brahmic Linguistic metrics:
    - Fertility (Tokens per Word, add_special_tokens=False)
    - Bytes Per Token (BPT)
    - Sequence Length Reduction (SLR % against a fixed baseline anchor)
    - Shannon Token Entropy (bits)
    - Akshara Integrity Rate (AIR %): % of all phonetic aksharas preserved intact
      without boundary cuts or mid-character byte splits (empirically measured on held-out text)
    - Byte-Fallback Rate (BFR %): % share of emitted tokens that are raw <0xXX> byte tokens
    - Train Dead IDs (%): % of vocabulary tokens never emitted on the training split
    - Eval Active Vocab (%): % of vocabulary tokens observed on the evaluation split
    """

    def __init__(self, corpus: List[str], train_corpus: Optional[List[str]] = None):
        self.corpus = corpus
        self.train_corpus = train_corpus
        self.total_chars = sum(len(doc) for doc in self.corpus)
        self.total_bytes = sum(len(doc.encode("utf-8")) for doc in self.corpus)
        self.total_words = sum(len(doc.split()) for doc in self.corpus)

    def evaluate_tokenizer(self, name: str, tokenizer: Any) -> Dict[str, Any]:
        """Runs the full evaluation battery on a single tokenizer."""
        total_tokens = 0
        token_freq_counter = Counter()
        total_unk_tokens = 0

        total_phonetic_aksharas = 0
        preserved_phonetic_aksharas = 0
        total_audited_tokens = 0
        total_byte_tokens = 0
        offsets_available = False

        # Identify <unk> token ID if available
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if unk_token_id is None and hasattr(tokenizer, "raw_tokenizer"):
            unk_token_id = tokenizer.raw_tokenizer.token_to_id("<unk>")

        audit_corpus = self.corpus[:min(300, len(self.corpus))]

        for doc in self.corpus:
            # Tokenize strictly with add_special_tokens=False
            if hasattr(tokenizer, "encode"):
                try:
                    ids = tokenizer.encode(doc, add_special_tokens=False)
                except TypeError:
                    ids = tokenizer.encode(doc)
            else:
                ids = tokenizer(doc, add_special_tokens=False)["input_ids"]

            if hasattr(ids, "ids"):
                ids = ids.ids

            total_tokens += len(ids)
            token_freq_counter.update(ids)
            if unk_token_id is not None:
                total_unk_tokens += sum(1 for tid in ids if tid == unk_token_id)

        # Empirical Linguistic Akshara Boundary & Byte Fallback Audit
        for doc in audit_corpus:
            akshara_spans = extract_akshara_spans(doc)
            if not akshara_spans:
                continue

            # Obtain offsets and token strings
            offsets = []
            tokens_list = []
            if hasattr(tokenizer, "encode_tokens_and_offsets"):
                try:
                    _, tokens_list, offsets = tokenizer.encode_tokens_and_offsets(doc, add_special_tokens=False)
                except Exception:
                    offsets = []
            elif callable(tokenizer):
                try:
                    out = tokenizer(doc, return_offsets_mapping=True, add_special_tokens=False)
                    offsets = out.get("offset_mapping", [])
                    if hasattr(tokenizer, "convert_ids_to_tokens") and "input_ids" in out:
                        tokens_list = tokenizer.convert_ids_to_tokens(out["input_ids"])
                except Exception:
                    offsets = []

            # Track byte tokens in audited sample
            if tokens_list:
                total_audited_tokens += len(tokens_list)
                total_byte_tokens += sum(1 for t in tokens_list if is_byte_token(t))

            # Filter valid offsets
            valid_entries: List[Tuple[int, int, str]] = []
            if tokens_list and len(tokens_list) == len(offsets):
                valid_entries = [(s, e, t) for (s, e), t in zip(offsets, tokens_list) if s < e]
            elif offsets:
                valid_entries = [(s, e, "") for s, e in offsets if s < e]

            if not valid_entries:
                continue

            offsets_available = True
            doc_len = len(doc)
            total_phonetic_aksharas += len(akshara_spans)

            token_cuts: Set[int] = set()
            for s, e, _ in valid_entries:
                if 0 < s < doc_len:
                    token_cuts.add(s)
                if 0 < e < doc_len:
                    token_cuts.add(e)

            # Check which aksharas survived without being bisected or fragmented into byte tokens
            for a_s, a_e, _ in akshara_spans:
                # 1. Check for standard boundary cut strictly inside the akshara
                has_internal_cut = any(a_s < c < a_e for c in token_cuts)

                # 2. Check for mid-character byte splits among tokens overlapping this akshara
                overlapping = [entry for entry in valid_entries if max(entry[0], a_s) < min(entry[1], a_e)]
                has_byte_split = False
                for i in range(len(overlapping) - 1):
                    s1, e1, _ = overlapping[i]
                    s2, e2, _ = overlapping[i + 1]
                    if s2 < e1 or (s1 == s2 and e1 == e2):
                        has_byte_split = True
                        break

                if not has_internal_cut and not has_byte_split:
                    preserved_phonetic_aksharas += 1

        is_byte_level_bpe = any(k in name.lower() for k in ["qwen", "llama", "gpt2", "gpt-2"])

        if offsets_available and total_phonetic_aksharas > 0:
            air_pct: Optional[float] = round((preserved_phonetic_aksharas / total_phonetic_aksharas) * 100.0, 2)
        else:
            air_pct = None

        if is_byte_level_bpe:
            # Byte-level BPE maps individual bytes to unicode characters, not <0xXX> byte fallback tokens
            bfr_pct = None
        elif total_audited_tokens > 0:
            bfr_pct = round((total_byte_tokens / total_audited_tokens) * 100.0, 2)
        else:
            bfr_pct = None

        unk_rate_pct = round((total_unk_tokens / max(1, total_tokens)) * 100.0, 2)

        fertility = total_tokens / max(1, self.total_words)
        bytes_per_token = self.total_bytes / max(1, total_tokens)

        # Shannon Entropy (bits)
        entropy = 0.0
        for count in token_freq_counter.values():
            p = count / total_tokens
            if p > 0:
                entropy -= p * math.log2(p)

        vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else getattr(tokenizer, "vocab_size", len(token_freq_counter))

        # Eval Split Token Observations (sample coverage, NOT dead IDs)
        eval_active_tokens = len([tid for tid, cnt in token_freq_counter.items() if cnt > 0])
        eval_unobserved_tokens = max(0, vocab_size - eval_active_tokens)
        eval_unobserved_pct = round((eval_unobserved_tokens / max(1, vocab_size)) * 100.0, 1)

        # Training Split Audit: Measure genuine Dead IDs never emitted in training
        train_active_tokens = None
        train_dead_tokens = None
        train_dead_pct = None
        train_alphabet_total = None
        train_alphabet_observed = None
        train_alphabet_unobserved = None
        train_alphabet_utilization_pct = None

        if self.train_corpus:
            train_emitted_ids = set()
            for doc in self.train_corpus:
                if hasattr(tokenizer, "encode"):
                    try:
                        ids = tokenizer.encode(doc, add_special_tokens=False)
                    except TypeError:
                        ids = tokenizer.encode(doc)
                else:
                    ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
                if hasattr(ids, "ids"):
                    ids = ids.ids
                train_emitted_ids.update(ids)

            train_active_tokens = len(train_emitted_ids)
            train_dead_tokens = max(0, vocab_size - train_active_tokens)
            train_dead_pct = round((train_dead_tokens / max(1, vocab_size)) * 100.0, 1)

            # For proposed models, measure utilization of the seeded base akshara inventory
            if hasattr(tokenizer, "pua_mapper"):
                mapper = tokenizer.pua_mapper
                train_pua_chars = set()
                for doc in self.train_corpus:
                    train_pua_chars.update(mapper.text_to_pua(doc))
                pua_keys = set(mapper.pua2akshara.keys())
                observed_pua = train_pua_chars.intersection(pua_keys)
                train_alphabet_total = len(pua_keys)
                train_alphabet_observed = len(observed_pua)
                train_alphabet_unobserved = train_alphabet_total - train_alphabet_observed
                train_alphabet_utilization_pct = round(
                    (train_alphabet_observed / max(1, train_alphabet_total)) * 100.0, 1
                )

        return {
            "name": name,
            "vocab_size": vocab_size,
            "total_tokens": total_tokens,
            "fertility": round(fertility, 3),
            "bytes_per_token": round(bytes_per_token, 3),
            "entropy_bits": round(entropy, 3),
            "akshara_integrity_rate_pct": air_pct,
            "byte_fallback_rate_pct": bfr_pct,
            "unk_rate_pct": unk_rate_pct,
            "is_byte_level_bpe": is_byte_level_bpe,
            "train_active_tokens": train_active_tokens,
            "train_dead_tokens": train_dead_tokens,
            "train_dead_pct": train_dead_pct,
            "train_alphabet_total": train_alphabet_total,
            "train_alphabet_observed": train_alphabet_observed,
            "train_alphabet_unobserved": train_alphabet_unobserved,
            "train_alphabet_utilization_pct": train_alphabet_utilization_pct,
            "eval_active_tokens": eval_active_tokens,
            "eval_unobserved_tokens": eval_unobserved_tokens,
            "eval_unobserved_pct": eval_unobserved_pct,
        }

    def compute_slr_and_format_tables(
        self,
        results: List[Dict[str, Any]],
        anchor_name: str = "XLM-RoBERTa",
    ) -> Tuple[str, str]:
        """Computes Sequence Length Reduction (SLR %) against an explicit, consistent anchor."""
        if not results:
            return "", ""

        anchor_tokens = None
        anchor_label = ""
        for r in results:
            if anchor_name.lower() in r["name"].lower():
                anchor_tokens = r["total_tokens"]
                anchor_label = r["name"]
                break

        if anchor_tokens is None:
            for r in results:
                if "Proposed" not in r["name"] and "Ablation" not in r["name"]:
                    anchor_tokens = r["total_tokens"]
                    anchor_label = r["name"]
                    break

        if anchor_tokens is None:
            anchor_tokens = results[0]["total_tokens"]
            anchor_label = results[0]["name"]

        for r in results:
            slr = ((anchor_tokens - r["total_tokens"]) / anchor_tokens) * 100.0
            r["seq_length_reduction_pct"] = round(slr, 1)

        has_train_stats = any(r.get("train_dead_pct") is not None for r in results)

        # 1. Markdown Table
        if has_train_stats:
            header = f"| Tokenizer | Vocab Size | Fertility (Tok/Word) | Bytes/Token | SLR vs. {anchor_label[:12]} (%) | Entropy (bits) | AIR (%) | BFR (%) | UNK (%) | Train Dead (%) |"
            sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        else:
            header = f"| Tokenizer | Vocab Size | Fertility (Tok/Word) | Bytes/Token | SLR vs. {anchor_label[:12]} (%) | Entropy (bits) | AIR (%) | BFR (%) | UNK (%) |"
            sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"

        rows = [header, sep]
        for r in results:
            slr_str = f"{r['seq_length_reduction_pct']:+.1f}%" if r["name"] != anchor_label else "Anchor (0.0%)"
            air_str = f"{r['akshara_integrity_rate_pct']:.1f}%" if r['akshara_integrity_rate_pct'] is not None else "N/A"
            if r.get("is_byte_level_bpe"):
                bfr_str = "N/A* (Byte-Level)"
            elif r['byte_fallback_rate_pct'] is not None:
                bfr_str = f"{r['byte_fallback_rate_pct']:.2f}%"
            else:
                bfr_str = "N/A"
            unk_str = f"{r.get('unk_rate_pct', 0.0):.2f}%"
            if has_train_stats:
                dead_str = f"{r['train_dead_pct']:.1f}%" if r.get('train_dead_pct') is not None else "N/A"
                rows.append(
                    f"| **{r['name']}** | {r['vocab_size']:,} | {r['fertility']} | {r['bytes_per_token']} | {slr_str} | {r['entropy_bits']} | {air_str} | {bfr_str} | {unk_str} | {dead_str} |"
                )
            else:
                rows.append(
                    f"| **{r['name']}** | {r['vocab_size']:,} | {r['fertility']} | {r['bytes_per_token']} | {slr_str} | {r['entropy_bits']} | {air_str} | {bfr_str} | {unk_str} |"
                )

        note_lines = [
            "\n*Notes:",
            "- AIR (%) measures the empirical percentage of phonetic aksharas in the held-out split preserved intact without boundary cuts or byte splits.",
            "- Train Dead (%) reports vocabulary token IDs never emitted on the training split (e.g. unreached merge capacity or unobserved base inventory units).",
            "- Byte-Level BPE architectures (e.g. Qwen, LLaMA) map raw UTF-8 bytes to unicode base characters rather than emitting explicit `<0xXX>` byte fallback tokens.",
        ]
        rows.append("\n".join(note_lines))
        md_table = "\n".join(rows)

        # 2. LaTeX Table
        cols = "lccccccccc" if has_train_stats else "lcccccccc"
        latex_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Intrinsic, Akshara-syllabic, Byte-Fallback, and Unknown token evaluation of tokenizers on Sinhala. "
            f"Sequence Length Reduction (SLR) is anchored to {anchor_label}. "
            r"AIR (\%) measures the empirical percentage of phonetic aksharas in the held-out split preserved intact without boundary cuts or byte splits; "
            r"BFR (\%) denotes the Byte-Fallback Rate (share of emitted tokens that are raw \texttt{\textless 0xXX\textgreater} byte tokens); "
            r"UNK (\%) measures the rate of unknown tokens emitted; "
            + (r"Dead (\%) reports the share of vocabulary IDs never emitted on the training split; " if has_train_stats else "")
            + r"\textsuperscript{*}Byte-Level BPE models (e.g. Qwen, LLaMA) represent raw bytes as base vocabulary characters rather than \texttt{\textless 0xXX\textgreater} tokens.}",
            r"\label{tab:tokenizer_intrinsic_evaluation}",
            f"\\begin{{tabular}}{{{cols}}}",
            r"\toprule",
        ]
        if has_train_stats:
            latex_lines.append(
                r"\textbf{Tokenizer} & \textbf{Vocab} & \textbf{Fertility} $\downarrow$ & \textbf{BPT} $\uparrow$ & \textbf{SLR (\%)} $\uparrow$ & \textbf{Entropy} & \textbf{AIR (\%)} $\uparrow$ & \textbf{BFR (\%)} $\downarrow$ & \textbf{UNK (\%)} $\downarrow$ & \textbf{Dead (\%)} $\downarrow$ \\"
            )
        else:
            latex_lines.append(
                r"\textbf{Tokenizer} & \textbf{Vocab} & \textbf{Fertility} $\downarrow$ & \textbf{BPT} $\uparrow$ & \textbf{SLR (\%)} $\uparrow$ & \textbf{Entropy} & \textbf{AIR (\%)} $\uparrow$ & \textbf{BFR (\%)} $\downarrow$ & \textbf{UNK (\%)} $\downarrow$ \\"
            )
        latex_lines.append(r"\midrule")

        for r in results:
            slr_str = f"{r['seq_length_reduction_pct']:+.1f}\\%" if r["name"] != anchor_label else "Anchor"
            air_str = f"{r['akshara_integrity_rate_pct']:.1f}\\%" if r['akshara_integrity_rate_pct'] is not None else "N/A"
            if r.get("is_byte_level_bpe"):
                bfr_str = r"N/A\textsuperscript{*}"
            elif r['byte_fallback_rate_pct'] is not None:
                bfr_str = f"{r['byte_fallback_rate_pct']:.2f}\\%"
            else:
                bfr_str = "N/A"
            unk_str = f"{r.get('unk_rate_pct', 0.0):.2f}\\%"
            is_proposed = "Proposed" in r["name"]
            row_prefix = r"\textbf{" if is_proposed else ""
            row_suffix = r"}" if is_proposed else ""
            if has_train_stats:
                dead_str = f"{r['train_dead_pct']:.1f}\\%" if r.get('train_dead_pct') is not None else "N/A"
                latex_lines.append(
                    f"{row_prefix}{r['name']}{row_suffix} & {r['vocab_size']:,} & {r['fertility']:.2f} & {r['bytes_per_token']:.2f} & {slr_str} & {r['entropy_bits']:.2f} & {air_str} & {bfr_str} & {unk_str} & {dead_str} \\\\"
                )
            else:
                latex_lines.append(
                    f"{row_prefix}{r['name']}{row_suffix} & {r['vocab_size']:,} & {r['fertility']:.2f} & {r['bytes_per_token']:.2f} & {slr_str} & {r['entropy_bits']:.2f} & {air_str} & {bfr_str} & {unk_str} \\\\"
                )

        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        latex_table = "\n".join(latex_lines)

        return md_table, latex_table


# ==============================================================================
# 6. EXTRINSIC LM PRETRAINING EXPERIMENT: EXACT OFFSET-BASED BPC ON ALL EVAL CHUNKS
# ==============================================================================

class ExactOffsetTokenDataset(TorchDataset):
    """
    Chunked autoregressive dataset for causal language modeling with exact
    character and UTF-8 byte counting from original document offset mappings.
    Eliminates decoding whitespace stripping and split-byte artifacts.
    """

    def __init__(
        self,
        token_ids: List[int],
        token_metadata: Optional[List[Tuple[int, int, int]]] = None,
        docs: Optional[List[str]] = None,
        block_size: int = 128,
        tok: Optional[Any] = None,
    ):
        self.block_size = block_size
        total_chunks = len(token_ids) // block_size
        self.chunks = []
        self.chunk_chars = []
        self.chunk_bytes = []

        for k in range(total_chunks):
            chunk_slice = token_ids[k * block_size : (k + 1) * block_size]
            self.chunks.append(torch.tensor(chunk_slice, dtype=torch.long))

            # Target tokens evaluated by cross-entropy loss are chunk_slice[1:]
            target_indices = range(k * block_size + 1, (k + 1) * block_size)
            chars_count = 0
            bytes_count = 0

            if token_metadata is not None and docs is not None and len(token_metadata) >= (k + 1) * block_size:
                doc_spans: Dict[int, List[Tuple[int, int]]] = {}
                for idx in target_indices:
                    d_idx, s, e = token_metadata[idx]
                    if s < e:
                        if d_idx not in doc_spans:
                            doc_spans[d_idx] = []
                        doc_spans[d_idx].append((s, e))

                for d_idx, spans in doc_spans.items():
                    if 0 <= d_idx < len(docs):
                        min_s = min(s for s, e in spans)
                        max_e = max(e for s, e in spans)
                        doc_slice = docs[d_idx][min_s:max_e]
                        chars_count += len(doc_slice)
                        bytes_count += len(doc_slice.encode("utf-8"))

            if chars_count == 0 and tok is not None:
                # Robust fallback if offsets unavailable
                target_ids = chunk_slice[1:]
                decoded_str = tok.decode(target_ids, skip_special_tokens=True)
                chars_count = max(1, len(decoded_str))
                bytes_count = max(1, len(decoded_str.encode("utf-8")))

            self.chunk_chars.append(max(1, chars_count))
            self.chunk_bytes.append(max(1, bytes_count))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:], self.chunk_chars[idx], self.chunk_bytes[idx]


def run_extrinsic_lm_benchmark(
    tokenizers_dict: Dict[str, Any],
    train_texts: List[str],
    val_texts: List[str],
    max_steps: int = 500,
    eval_interval: int = 50,
    batch_size: int = 8,
    block_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Pretrains a compact Transformer (GPT-2 Mini) across each tokenizer.
    Evaluates cross-entropy loss converted to vocabulary-invariant:
    - Bits-Per-Character (BPC)
    - Bits-Per-Byte (BPB)
    Calculated across ALL validation chunks using exact source-text character offsets,
    eliminating decoder spacing rewrites, boundary space stripping, and split-byte artifacts.
    Logs characters seen alongside tokens seen in the training trajectory.
    """
    print(f"\n[*] Starting Extrinsic LM Experiment on {device.upper()} (Max Steps: {max_steps})...")
    print("    Metric: Vocabulary-invariant Bits-Per-Character (BPC) and Bits-Per-Byte (BPB) via exact document offsets.")

    results = {}

    for name, tok in tokenizers_dict.items():
        print(f"\n--- Training Mini-LM with {name} ---")

        train_ids = []
        for t in train_texts:
            if hasattr(tok, "encode"):
                try:
                    res = tok.encode(t, add_special_tokens=False)
                except TypeError:
                    res = tok.encode(t)
            else:
                res = tok(t, add_special_tokens=False)["input_ids"]
            if hasattr(res, "ids"):
                res = res.ids
            train_ids.extend(res)

        val_ids = []
        val_metadata = []
        for d_idx, doc in enumerate(val_texts):
            offsets = []
            res_ids = []
            if hasattr(tok, "encode_tokens_and_offsets"):
                try:
                    res_ids, _, offsets = tok.encode_tokens_and_offsets(doc, add_special_tokens=False)
                except Exception:
                    pass
            if not res_ids:
                if callable(tok):
                    try:
                        out = tok(doc, return_offsets_mapping=True, add_special_tokens=False)
                        res_ids = out["input_ids"]
                        offsets = out.get("offset_mapping", [])
                    except Exception:
                        pass
            if not res_ids:
                if hasattr(tok, "encode"):
                    try:
                        res = tok.encode(doc, add_special_tokens=False)
                    except TypeError:
                        res = tok.encode(doc)
                else:
                    res = tok(doc, add_special_tokens=False)["input_ids"]
                if hasattr(res, "ids"):
                    res = res.ids
                res_ids = list(res)

            val_ids.extend(res_ids)
            if offsets and len(offsets) == len(res_ids):
                for s, e in offsets:
                    val_metadata.append((d_idx, s, e))
            else:
                for _ in range(len(res_ids)):
                    val_metadata.append((d_idx, 0, 0))

        exact_vocab_size = len(tok) if hasattr(tok, "__len__") else getattr(tok, "vocab_size", 32000)
        padded_vocab_size = int(math.ceil(exact_vocab_size / 64.0) * 64)

        config = GPT2Config(
            vocab_size=padded_vocab_size,
            n_positions=block_size * 2,
            n_embd=256,
            n_layer=4,
            n_head=4,
            bos_token_id=getattr(tok, "bos_token_id", 0) or 0,
            eos_token_id=getattr(tok, "eos_token_id", 1) or 1,
        )
        model = GPT2LMHeadModel(config).to(device)

        train_ds = ExactOffsetTokenDataset(train_ids, block_size=block_size)
        val_ds = ExactOffsetTokenDataset(val_ids, token_metadata=val_metadata, docs=val_texts, block_size=block_size, tok=tok)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Training text consumption stats for trajectory plotting
        train_total_chars = sum(len(t) for t in train_texts)
        train_total_bytes = sum(len(t.encode("utf-8")) for t in train_texts)
        train_chars_per_token = train_total_chars / max(1, len(train_ids))
        train_bytes_per_token = train_total_bytes / max(1, len(train_ids))

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        step = 0
        start_time = time.time()
        step_trajectory = []

        def run_exact_val_evaluation() -> Tuple[float, float, float]:
            """
            Evaluates ALL validation chunks using exact source-text character offsets,
            eliminating decoder spacing rewrites and split-byte artifacts.
            """
            model.eval()
            total_nats = 0.0
            total_eval_chars = 0
            total_eval_bytes = 0
            total_eval_tokens = 0

            with torch.no_grad():
                for x, y, chunk_c, chunk_b in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x).logits
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

                    batch_tok_count = y.numel()
                    total_nats += loss.item() * batch_tok_count
                    total_eval_tokens += batch_tok_count
                    total_eval_chars += chunk_c.sum().item()
                    total_eval_bytes += chunk_b.sum().item()

            model.train()

            avg_loss = total_nats / max(1, total_eval_tokens)
            bpc = total_nats / (math.log(2.0) * max(1, total_eval_chars))
            bpb = total_nats / (math.log(2.0) * max(1, total_eval_bytes))
            return avg_loss, bpc, bpb

        for epoch in range(25):
            for x, y, _, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1

                if step % eval_interval == 0 or step == max_steps:
                    val_loss, bpc, bpb = run_exact_val_evaluation()
                    tokens_seen = step * batch_size * block_size
                    chars_seen = int(tokens_seen * train_chars_per_token)
                    bytes_seen = int(tokens_seen * train_bytes_per_token)

                    step_trajectory.append({
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "chars_seen": chars_seen,
                        "bytes_seen": bytes_seen,
                        "token_loss": round(val_loss, 3),
                        "val_bpc": round(bpc, 4),
                        "val_bpb": round(bpb, 4),
                    })
                    print(f"  Step {step:03d}/{max_steps} | Tokens: {tokens_seen:,} | Chars: {chars_seen:,} | Loss: {val_loss:.3f} | BPC: {bpc:.3f} | BPB: {bpb:.3f}")

                if step >= max_steps:
                    break
            if step >= max_steps:
                break

        elapsed = time.time() - start_time
        final_loss, final_bpc, final_bpb = run_exact_val_evaluation()
        tokens_per_sec = (step * batch_size * block_size) / max(0.1, elapsed)

        results[name] = {
            "model_vocab_size": padded_vocab_size,
            "final_val_token_loss": round(final_loss, 3),
            "final_val_bpc": round(final_bpc, 4),
            "final_val_bpb": round(final_bpb, 4),
            "training_time_sec": round(elapsed, 1),
            "tokens_per_sec": round(tokens_per_sec, 0),
            "trajectory": step_trajectory,
        }

    return results


# ==============================================================================
# 7. MAIN REPLICATION PIPELINE ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sinhala Tokenizer Research Pipeline for Q1 Paper")
    parser.add_argument("--dataset", type=str, default="Isuru0x01/sinhala_stories", help="Hugging Face repo or local path")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of documents to sample for training/eval")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Proposed tokenizer vocabulary size")
    parser.add_argument("--output_dir", type=str, default="./sinhala_tokenizer_output", help="Directory to save artifacts")
    parser.add_argument("--skip_lm_eval", action="store_true", help="Skip extrinsic mini-LM pretraining test")
    parser.add_argument("--lm_steps", type=int, default=500, help="Max steps for extrinsic LM pretraining experiment")
    args = parser.parse_args()

    if not HAS_CORE_DEPS:
        print(f"\n[ERROR] Missing required library: {IMPORT_ERR}")
        print("Please install dependencies using:")
        print("pip install tokenizers transformers datasets sentencepiece torch scipy tabulate\n")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    print("=" * 85)
    print("SINHALA TOKENIZER RESEARCH PIPELINE (Q1 REPLICATION SUITE)")
    print(f"Dataset: {args.dataset} | Sample Size: {args.sample_size:,} | Vocab: {args.vocab_size:,}")
    print("=" * 85)

    # ---------------------------------------------------------
    # Step 1: Ingest, Deduplicate & Split Corpus (Fails Loudly)
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Corpus, Deduplicating, and Normalizing...")
    raw_corpus = []

    if HAS_DATASETS:
        try:
            ds = load_dataset(args.dataset, split="train", token=hf_token)
            print(f"[+] Successfully loaded {len(ds):,} documents from Hugging Face: {args.dataset}")
            story_col = "story" if "story" in ds.column_names else ds.column_names[0]
            for row in ds.select(range(min(args.sample_size, len(ds)))):
                text = row.get(story_col, "")
                if text and len(text) > 30:
                    raw_corpus.append(normalize_sinhala_text(text))
        except Exception as e_hub:
            print(f"[-] Could not load from Hugging Face Hub: {e_hub}")
            local_parquet = "data/train-append.parquet"
            if os.path.exists(local_parquet):
                print(f"[+] Loading from local parquet: {local_parquet}")
                ds = load_dataset("parquet", data_files=local_parquet, split="train")
                for row in ds.select(range(min(args.sample_size, len(ds)))):
                    text = row.get("story", "")
                    if text:
                        raw_corpus.append(normalize_sinhala_text(text))
            else:
                raise RuntimeError(
                    f"\n[FATAL] Unable to load dataset '{args.dataset}' from Hub and no local file found at '{local_parquet}'.\n"
                    "Please verify network access, export HF_TOKEN if the dataset is private, or provide a local parquet path.\n"
                    "Silent synthetic fallback is disabled to guarantee scientific reproducibility."
                )
    else:
        local_parquet = "data/train-append.parquet"
        if os.path.exists(local_parquet):
            try:
                import pandas as pd
                print(f"[+] Loading from local parquet via pandas: {local_parquet}")
                df = pd.read_parquet(local_parquet)
                col = "story" if "story" in df.columns else df.columns[0]
                for text in df[col].dropna().head(args.sample_size):
                    if text and len(text) > 30:
                        raw_corpus.append(normalize_sinhala_text(text))
            except Exception as e_pd:
                raise RuntimeError(f"[FATAL] Failed reading local parquet: {e_pd}")
        else:
            raise RuntimeError(
                "\n[FATAL] The 'datasets' library is required to load from Hugging Face Hub.\n"
                "Please run: pip install datasets\n"
            )

    # Deduplicate documents to eliminate lexical leakage
    corpus = list(dict.fromkeys(raw_corpus))
    print(f"[✓] Deduplicated corpus: {len(raw_corpus):,} -> {len(corpus):,} unique documents.")

    # Deterministic randomized train/evaluation split (seed=42)
    rng = random.Random(42)
    rng.shuffle(corpus)

    val_split_idx = int(len(corpus) * 0.85)
    train_corpus = corpus[:val_split_idx]
    eval_corpus = corpus[val_split_idx:]
    print(f"[✓] Train split: {len(train_corpus):,} docs | Eval split: {len(eval_corpus):,} docs.")

    # ---------------------------------------------------------
    # Step 2: Build Akshara PUA Alphabet & Train Models
    # ---------------------------------------------------------
    print("\n[Step 2] Building Akshara PUA Alphabet & Training Tokenizers...")
    all_train_aksharas = []
    for doc in train_corpus:
        all_train_aksharas.extend(m.group(0) for m in SINHALA_AKSHARA_REGEX.finditer(doc))

    pua_mapper = AksharaPUAMapper(all_train_aksharas)
    trainer = SinhalaTokenizerTrainer(vocab_size=args.vocab_size)
    trainer.enforce_vocab_parity(pua_mapper)

    v_label = f"{trainer.vocab_size // 1000}k" if trainer.vocab_size >= 1000 else f"{trainer.vocab_size}v"
    proposed_bpe_name = f"Proposed Akshara-BPE ({v_label})"
    proposed_uni_name = f"Proposed Akshara-Unigram ({v_label})"
    bpe_ablation_name = f"Ablation: Plain BPE ({v_label})"
    unigram_ablation_name = f"Ablation: Plain Unigram ({v_label})"

    # 2a. Proposed Akshara-BPE
    proposed_bpe = trainer.train_proposed_bpe(train_corpus, pua_mapper, name=proposed_bpe_name)
    bpe_save_dir = os.path.join(args.output_dir, "proposed_sinhala_bpe")
    proposed_bpe.save_pretrained(bpe_save_dir)
    print(f"[✓] Saved Proposed Akshara-BPE to: {bpe_save_dir}")

    # 2b. Proposed Akshara-Unigram
    proposed_unigram = trainer.train_proposed_unigram(train_corpus, pua_mapper, name=proposed_uni_name)
    unigram_save_dir = os.path.join(args.output_dir, "proposed_sinhala_unigram")
    proposed_unigram.save_pretrained(unigram_save_dir)
    print(f"[✓] Saved Proposed Akshara-Unigram to: {unigram_save_dir}")

    # 2c. In-Domain Ablation Controls (Same corpus & vocab, without Akshara mapping)
    ablation_plain_bpe = trainer.train_ablation_plain_bpe(train_corpus, name=bpe_ablation_name)
    ablation_plain_unigram = trainer.train_ablation_plain_unigram(train_corpus, name=unigram_ablation_name)

    # 2d. Vocabulary Parity Verification between Proposed and Ablation Controls
    proposed_bpe_v = proposed_bpe.vocab_size
    ablation_bpe_v = getattr(ablation_plain_bpe, "vocab_size", len(ablation_plain_bpe))
    proposed_uni_v = proposed_unigram.vocab_size
    ablation_uni_v = getattr(ablation_plain_unigram, "vocab_size", len(ablation_plain_unigram))

    bpe_mismatch = abs(proposed_bpe_v - ablation_bpe_v) > max(50, int(0.02 * trainer.vocab_size))
    uni_mismatch = abs(proposed_uni_v - ablation_uni_v) > max(50, int(0.02 * trainer.vocab_size))

    print("\n" + "=" * 85)
    print("[Step 2d] Target and Trained Vocabulary Parity Verification")
    print("=" * 85)
    print(f"Enforced Nominal Target Vocab: {trainer.vocab_size:,} (Requested: {args.vocab_size:,})")
    print(f"  - Proposed Akshara-BPE Vocab:     {proposed_bpe_v:,}")
    print(f"  - Proposed Akshara-Unigram Vocab: {proposed_uni_v:,}")
    print(f"  - Ablation Plain BPE Vocab:       {ablation_bpe_v:,}")
    print(f"  - Ablation Plain Unigram Vocab:   {ablation_uni_v:,}")

    if bpe_mismatch or uni_mismatch:
        print("\n[!] TOY / SMALL CORPUS MERGE SATURATION NOTICE:")
        if bpe_mismatch:
            print(f"    BPE: Proposed ({proposed_bpe_v:,}) vs Ablation Plain BPE ({ablation_bpe_v:,}) -> diff: {abs(proposed_bpe_v - ablation_bpe_v):,}")
        if uni_mismatch:
            print(f"    Unigram: Proposed ({proposed_uni_v:,}) vs Ablation Plain Unigram ({ablation_uni_v:,}) -> diff: {abs(proposed_uni_v - ablation_uni_v):,}")
        print("    * Note: Both models were trained with the exact same target vocabulary size.")
        print(f"      However, on small corpora (e.g. toy runs with few sentences), plain BPE/Unigram")
        print(f"      exhausts all available character/subword n-grams and saturates early, whereas")
        print(f"      Proposed models retain the full {len(pua_mapper.pua2akshara):,} seeded base alphabet.")
        print("      For publication benchmarks, ensure full corpus coverage (--sample_size >= 10,000).")
    else:
        print("\n[✓] Strict vocabulary parity confirmed between Proposed and Ablation models.")
    print("=" * 85)

    # ---------------------------------------------------------
    # Step 3: Benchmarking Against Ablations & Baselines
    # ---------------------------------------------------------
    print("\n[Step 3] Loading Verified Baselines and Running Intrinsic Evaluation...")
    tokenizers_to_compare = {
        proposed_bpe.name: proposed_bpe,
        proposed_unigram.name: proposed_unigram,
        bpe_ablation_name: ablation_plain_bpe,
        unigram_ablation_name: ablation_plain_unigram,
    }

    verified_baselines = load_verified_baseline_tokenizers(hf_token=hf_token)
    tokenizers_to_compare.update(verified_baselines)

    evaluator = TokenizerEvaluator(eval_corpus, train_corpus=train_corpus)
    intrinsic_results = []

    for name, tok in tokenizers_to_compare.items():
        print(f"[*] Evaluating Intrinsic & Akshara Metrics: {name}...")
        res = evaluator.evaluate_tokenizer(name, tok)
        dead_info = f" | Train Dead IDs: {res['train_dead_tokens']:,}/{res['vocab_size']:,} ({res['train_dead_pct']}%)" if res.get('train_dead_tokens') is not None else ""
        eval_active_info = f" | Eval Active: {res['eval_active_tokens']:,}/{res['vocab_size']:,} ({res['eval_unobserved_pct']}% unobserved in eval)"
        print(f"    Emitted tokens: {res['total_tokens']:,}{dead_info}{eval_active_info}")
        if res.get("train_alphabet_total") is not None:
            print(f"    -> Seeded Akshara Alphabet in Train: {res['train_alphabet_observed']:,}/{res['train_alphabet_total']:,} ({res['train_alphabet_utilization_pct']}% observed, {res['train_alphabet_unobserved']:,} rare forms unobserved in train)")
        intrinsic_results.append(res)

    print("\n" + "=" * 85)
    print("INTRINSIC EVALUATION RESULTS TABLE")
    print("=" * 85)
    md_table, latex_table = evaluator.compute_slr_and_format_tables(
        intrinsic_results,
        anchor_name="XLM-RoBERTa",
    )
    print(md_table)

    tex_path = os.path.join(args.output_dir, "table_intrinsic_metrics.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\n[+] Exported publication LaTeX table to: {tex_path}")

    intrinsic_json_path = os.path.join(args.output_dir, "intrinsic_evaluation_results.json")
    with open(intrinsic_json_path, "w", encoding="utf-8") as f:
        json.dump(intrinsic_results, f, indent=2)
    print(f"[+] Saved full intrinsic metrics (including train dead IDs & eval coverage) to: {intrinsic_json_path}")

    # ---------------------------------------------------------
    # Step 4: Extrinsic LM Pretraining Experiment (Exact BPC)
    # ---------------------------------------------------------
    if not args.skip_lm_eval and len(eval_corpus) >= 10:
        print("\n" + "=" * 85)
        print("[Step 4] Extrinsic LM Pretraining Validation (Exact Offset-Based BPC on All Val Chunks)")
        print("=" * 85)

        lm_results = run_extrinsic_lm_benchmark(
            tokenizers_dict=tokenizers_to_compare,
            train_texts=train_corpus[:min(1000, len(train_corpus))],
            val_texts=eval_corpus[:min(200, len(eval_corpus))],
            max_steps=args.lm_steps,
            eval_interval=50,
        )

        lm_json_path = os.path.join(args.output_dir, "lm_benchmark_results.json")
        with open(lm_json_path, "w", encoding="utf-8") as f:
            json.dump(lm_results, f, indent=2)
        print(f"[+] LM Benchmark metrics & step trajectories saved to: {lm_json_path}")

        print("\n" + "=" * 85)
        print("EXTRINSIC LM BENCHMARK SUMMARY (LOWER BPC/BPB IS BETTER)")
        print("=" * 85)
        print("| Tokenizer | Model Vocab | Val Token Loss | Val BPC (Bits/Char) | Val BPB (Bits/Byte) | Throughput (tok/s) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for name, data in lm_results.items():
            print(f"| **{name}** | {data['model_vocab_size']:,} | {data['final_val_token_loss']} | {data['final_val_bpc']} | {data['final_val_bpb']} | {data['tokens_per_sec']:.0f} |")

    print("\n" + "=" * 85)
    print("ALL PHASES COMPLETED ACCURATELY.")
    print(f"Artifacts and LaTeX exports saved in: {args.output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
