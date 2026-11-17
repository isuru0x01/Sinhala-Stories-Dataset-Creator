"""
validator.py
Decoupled text validation and quality heuristics for crowdsourced NLP corpora.
Extracts pure functional logic to enable independent automated unit testing.
"""

from __future__ import annotations
import os
import re
import math
import struct
import hashlib
import unicodedata
from typing import Dict, Any, Tuple, List, Set, Optional, Iterable

import langid
import numpy as np
import yaml

# Default fallback configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "Sinhala Stories Dataset Creator",
        "app_version": "2.0",
        "validation_version": "1.2",
        "hf_repo_id": "Isuru0x01/sinhala_stories",
        "id_prefix": "SLS",
    },
    "validation": {
        "min_length": 50,
        "max_length": 50000,
        "target_language_iso": "si",
        "min_language_confidence": 0.90,
        "unicode_script_name": "Sinhala",
        "unicode_range_start": 0x0D80,
        "unicode_range_end": 0x0DFF,
        "min_script_ratio": 0.90,
        "min_spam_score": 0.20,
        "jaccard_similarity_threshold": 0.85,
    },
    "storage": {
        "pending_dir": "pending",
        "local_queue_dir": "local_queue",
        "append_parquet_path": "data/train-append.parquet",
        "hashes_path": "hashes.txt",
        "bloom_path": "hashes.bloom",
        "stats_path": "dataset_stats.json",
    },
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file with fallback to built-in defaults."""
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    for section, values in loaded.items():
                        if isinstance(values, dict) and section in config:
                            config[section].update(values)
                        else:
                            config[section] = values
        except Exception:
            pass  # Fall back to default config if file is malformed
    return config


def normalize_text(text: str) -> str:
    """
    Performs Unicode NFC normalization, collapses multiple consecutive dots,
    trims whitespace, and collapses multiple spaces into single space.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    # Collapse multiple consecutive dots into a single dot
    normalized = re.sub(r"\.{2,}", ".", normalized)
    # Strip and collapse multiple spaces/newlines
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


# Alias for backward compatibility with app.py
normalize_story = normalize_text


def compute_sha256(text: str) -> str:
    """Computes the SHA-256 cryptographic digest of the input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_script_ratio(
    text: str,
    start_hex: int = 0x0D80,
    end_hex: int = 0x0DFF,
) -> float:
    """
    Calculates the ratio of characters falling within [start_hex, end_hex]
    over all alphabetic characters in the text.
    """
    alphabetic_chars = [c for c in text if c.isalpha()]
    if not alphabetic_chars:
        return 0.0
    script_chars = [c for c in alphabetic_chars if start_hex <= ord(c) <= end_hex]
    return len(script_chars) / len(alphabetic_chars)


# Alias for backward compatibility
calculate_sinhala_percentage = calculate_script_ratio


def detect_language(text: str, target_lang: str = "si") -> Tuple[str, float]:
    """
    Identifies the language of the text using langid.
    Computes a softmax-normalized probability over the top 5 candidates.
    Returns (predicted_iso_code, probability).
    """
    try:
        ranks = langid.rank(text)
        if not ranks:
            return "unknown", 0.0
        top_ranks = ranks[:5]
        max_score = top_ranks[0][1]
        scores = np.array([r[1] for r in top_ranks])
        exp_scores = np.exp(scores - max_score)
        probs = exp_scores / np.sum(exp_scores)
        return top_ranks[0][0], float(probs[0])
    except Exception:
        return "unknown", 0.0


# Alias for backward compatibility
detect_sinhala_language = detect_language


def get_words(text: str) -> Set[str]:
    """Extracts lowercase word tokens using Unicode-aware regex matching."""
    return set(re.findall(r"\w+", text.lower()))


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Computes the Jaccard similarity coefficient between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Computes word-level Jaccard similarity between two texts."""
    words_a = get_words(text_a)
    words_b = get_words(text_b)
    return jaccard_similarity(words_a, words_b)


def compute_spam_score_detailed(text: str) -> Tuple[float, List[str]]:
    """
    Combines lightweight quality heuristics into a normalized score between 0.0 and 1.0.
    1.0 represents a clean, high-diversity text. Lower scores indicate repetition/spam.
    Returns (score, list_of_reasons).
    """
    reasons: List[str] = []
    score = 1.0

    # 1. Repeated characters (5+ consecutive identical characters)
    for match in re.finditer(r"(.)\1{4,}", text):
        match_str = match.group(0)
        repetitions = len(match_str)
        penalty = 0.1 * repetitions
        score -= penalty
        reasons.append(f"Character '{match.group(1)}' repeated {repetitions} times (-{penalty:.2f})")

    # 2. Consecutive repeated words
    words = text.split()
    if words:
        repeated_words_count = 0
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                repeated_words_count += 1
        if repeated_words_count > 0:
            penalty = 0.1 * repeated_words_count
            score -= penalty
            reasons.append(f"Consecutive repeated words: {repeated_words_count} times (-{penalty:.2f})")

        # 3. Overall vocabulary duplication ratio
        unique_words = set(words)
        duplication_ratio = 1.0 - (len(unique_words) / len(words))
        if duplication_ratio > 0.2:
            penalty = 1.5 * duplication_ratio
            score -= penalty
            reasons.append(f"High word duplication ratio: {duplication_ratio:.1%} (-{penalty:.2f})")

    # 4. Repeated sentences
    sentences = [s.strip() for s in re.split(r"[.?!\u0964]", text) if s.strip()]
    if sentences:
        unique_sentences = set(sentences)
        dupes_count = len(sentences) - len(unique_sentences)
        if dupes_count > 0:
            ratio = dupes_count / len(sentences)
            if ratio > 0.2:
                penalty = ratio * 1.5
                score -= penalty
                reasons.append(f"High sentence repetition: {ratio:.1%} (-{penalty:.2f})")

    # 5. Excessive URLs
    urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", text)
    if urls:
        penalty = 0.5 * len(urls)
        score -= penalty
        reasons.append(f"URLs found: {len(urls)} (-{penalty:.2f})")

    # 6. Excessive emoji ratio
    emoji_count = len(re.findall(r"[\U00010000-\U0010ffff]", text))
    if len(text) > 0:
        emoji_ratio = emoji_count / len(text)
        if emoji_ratio > 0.1:
            penalty = emoji_ratio * 2.0
            score -= penalty
            reasons.append(f"High emoji ratio: {emoji_ratio:.1%} (-{penalty:.2f})")

    # 7. Abnormal punctuation ratio
    punctuations = re.findall(r"[!,.:;?\"\'()\-—\[\]{}「」\u0964]", text)
    if len(text) > 0:
        punc_ratio = len(punctuations) / len(text)
        if punc_ratio > 0.25:
            penalty = (punc_ratio - 0.25) * 3.0
            score -= penalty
            reasons.append(f"High punctuation ratio: {punc_ratio:.1%} (-{penalty:.2f})")

    # 8. Shannon character entropy gating (catches keyboard-mashing and synthesized loops)
    if len(text) >= 40:
        entropy = compute_shannon_entropy(text)
        if entropy < 2.5:
            penalty = (2.5 - entropy) * 0.6
            if entropy < 1.8:
                penalty += 0.4
            score -= penalty
            reasons.append(f"Low character entropy: {entropy:.2f} bits/char (-{penalty:.2f})")

    # 9. Bigram repetition ratio (catches repeating phrase loops)
    if words and len(words) >= 6:
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        bigram_dupe_ratio = 1.0 - (len(unique_bigrams) / len(bigrams))
        if bigram_dupe_ratio > 0.40:
            penalty = (bigram_dupe_ratio - 0.40) * 1.5
            score -= penalty
            reasons.append(f"High bigram repetition: {bigram_dupe_ratio:.1%} (-{penalty:.2f})")

    # Clamp score to [0.0, 1.0]
    final_score = max(0.0, min(1.0, score))
    return final_score, reasons


def compute_shannon_entropy(text: str) -> float:
    """
    Computes Shannon character entropy in bits per character:
    H(X) = -sum(P(c) * log2(P(c)))
    Natural Sinhala prose typically has H >= 3.2 bits/char.
    Low entropy (< 2.5) indicates repetitive character loops or keyboard mashing.
    """
    if not text:
        return 0.0
    from collections import Counter
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def compute_spam_score(text: str) -> float:
    """Computes quality score where 1.0 is highest quality and <0.20 is spam."""
    score, _ = compute_spam_score_detailed(text)
    return score


def validate_text_entry(text: str, config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Validates a text submission against length, script, language, and spam criteria.
    Returns a list of error messages (empty list if submission passes all gates).
    """
    if config is None:
        config = load_config()

    v_cfg = config.get("validation", DEFAULT_CONFIG["validation"])
    errors: List[str] = []

    normalized = normalize_text(text)
    if not normalized:
        errors.append("Story cannot be empty.")
        return errors

    min_len = v_cfg.get("min_length", 50)
    max_len = v_cfg.get("max_length", 50000)
    curr_len = len(normalized)

    if curr_len < min_len:
        errors.append(f"Must be at least {min_len} characters (Current: {curr_len}).")
    elif curr_len > max_len:
        errors.append(f"Exceeds maximum {max_len} characters (Current: {curr_len}).")

    start_hex = v_cfg.get("unicode_range_start", 0x0D80)
    end_hex = v_cfg.get("unicode_range_end", 0x0DFF)
    script_name = v_cfg.get("unicode_script_name", "Target Script")

    if not any(start_hex <= ord(c) <= end_hex for c in normalized):
        errors.append(f"No {script_name} characters detected.")

    script_ratio = calculate_script_ratio(normalized, start_hex, end_hex)
    min_ratio = v_cfg.get("min_script_ratio", 0.90)
    if script_ratio < min_ratio and any(c.isalpha() for c in normalized):
        errors.append(
            f"{script_name} character ratio ({script_ratio:.1%}) is below the {min_ratio:.1%} threshold."
        )

    target_lang = v_cfg.get("target_language_iso", "si")
    min_lang_conf = v_cfg.get("min_language_confidence", 0.90)
    lang, prob = detect_language(normalized, target_lang)
    if lang != target_lang or prob < min_lang_conf:
        errors.append(
            f"Language identification failed. Detected '{lang}' with {prob:.1%} probability. "
            f"Expected '{target_lang}' with at least {min_lang_conf:.1%} confidence."
        )

    min_spam = v_cfg.get("min_spam_score", 0.20)
    spam_score, reasons = compute_spam_score_detailed(normalized)
    if spam_score < min_spam:
        reason_str = "; ".join(reasons) if reasons else "High repetition or low vocabulary diversity"
        errors.append(f"Rejected as potential spam (Quality score: {spam_score:.2f} < {min_spam}). Reasons: {reason_str}")

    return errors


class BloomFilter:
    """
    Compact, high-performance binary Bloom Filter for O(1) deduplication.
    Uses double hashing (Kirsch-Mitzenmacher optimization) over SHA-256 digests.
    Requires zero external C-dependencies (pure standard library).
    """
    MAGIC = b"BLM1"

    def __init__(self, capacity: int = 100000, error_rate: float = 0.001):
        self.capacity = max(1, capacity)
        self.error_rate = min(max(error_rate, 1e-6), 0.1)
        # Optimal bit size: m = - (n * ln(p)) / (ln(2)^2)
        num_bits = int(- (self.capacity * math.log(self.error_rate)) / (math.log(2) ** 2))
        self.bit_size = max(64, num_bits)
        # Optimal number of hash functions: k = (m / n) * ln(2)
        self.num_hashes = max(1, round((self.bit_size / self.capacity) * math.log(2)))
        self.byte_count = (self.bit_size + 7) // 8
        self.bit_array = bytearray(self.byte_count)
        self.count = 0

    def _get_hashes(self, item: str):
        # 32-byte SHA256 digest provides two independent 64-bit integer seeds (h1, h2)
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = struct.unpack(">Q", digest[:8])[0]
        h2 = struct.unpack(">Q", digest[8:16])[0]
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.bit_size

    def add(self, item: str) -> None:
        """Adds an item (e.g. SHA-256 hex string or text) to the filter."""
        for bit in self._get_hashes(item):
            self.bit_array[bit // 8] |= (1 << (bit % 8))
        self.count += 1

    def __contains__(self, item: str) -> bool:
        """Returns True if the item might be in the set, False if definitely not."""
        for bit in self._get_hashes(item):
            if not (self.bit_array[bit // 8] & (1 << (bit % 8))):
                return False
        return True

    def to_bytes(self) -> bytes:
        """Serializes the filter into a compact binary representation."""
        header = struct.pack(">4sIIH", self.MAGIC, self.bit_size, self.num_hashes, self.count)
        return header + bytes(self.bit_array)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Deserializes a BloomFilter instance from binary data."""
        if len(data) < 14:
            raise ValueError("Invalid bloom filter binary data (too short)")
        magic, bit_size, num_hashes, count = struct.unpack(">4sIIH", data[:14])
        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        bf = cls.__new__(cls)
        bf.bit_size = bit_size
        bf.num_hashes = num_hashes
        bf.count = count
        bf.byte_count = (bit_size + 7) // 8
        expected_len = 14 + bf.byte_count
        if len(data) < expected_len:
            raise ValueError(f"Truncated bloom filter data: expected {expected_len} bytes, got {len(data)}")
        bf.bit_array = bytearray(data[14:expected_len])
        bf.capacity = max(count * 2, 100000)
        bf.error_rate = 0.001
        return bf

    def save(self, filepath: str) -> None:
        """Writes the binary filter to disk."""
        with open(filepath, "wb") as f:
            f.write(self.to_bytes())

    @classmethod
    def load(cls, filepath: str) -> "BloomFilter":
        """Loads a BloomFilter instance from a file."""
        with open(filepath, "rb") as f:
            return cls.from_bytes(f.read())


def build_bloom_filter(hashes: Iterable[str], capacity: int = 100000, error_rate: float = 0.001) -> BloomFilter:
    """Builds and populates a BloomFilter from an iterable of hash strings."""
    hash_list = list(hashes)
    cap = max(len(hash_list) * 2, capacity)
    bf = BloomFilter(capacity=cap, error_rate=error_rate)
    for h in hash_list:
        clean_h = h.strip()
        if clean_h:
            bf.add(clean_h)
    return bf
