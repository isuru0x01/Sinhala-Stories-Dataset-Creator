import pytest
import os
import tempfile
from validator import (
    normalize_text,
    calculate_script_ratio,
    compute_spam_score,
    compute_jaccard_similarity,
    compute_sha256,
    load_config,
    validate_text_entry,
    BloomFilter,
    build_bloom_filter,
    compute_shannon_entropy,
)


def test_unicode_nfc_and_whitespace_normalization():
    raw_text = "  මෙය   සිංහල   කතාවකි....   අත්හදා බැලීමක්.  "
    normalized = normalize_text(raw_text)
    assert normalized == "මෙය සිංහල කතාවකි. අත්හදා බැලීමක්."
    assert "  " not in normalized
    assert "...." not in normalized


def test_sinhala_script_ratio_valid():
    sinhala_text = "ලංකාවේ සිංහල භාෂාව සඳහා විශාල දත්ත එකතුවක් නිර්මාණය කිරීම."
    ratio = calculate_script_ratio(sinhala_text, 0x0D80, 0x0DFF)
    assert ratio >= 0.90


def test_sinhala_script_ratio_rejects_english():
    mixed_text = "This is mostly English text with one word සිංහල."
    ratio = calculate_script_ratio(mixed_text, 0x0D80, 0x0DFF)
    assert ratio < 0.90


def test_jaccard_near_duplicate_detection():
    story_1 = "අද උදෑසන මම කොළඹ නගරයට ගියෙමි එහිදී මගේ මිතුරෙකු හමුවිය"
    story_2 = "අද උදෑසන මම කොළඹ නගරයට ගියෙමි එහිදී මගේ හොඳම මිතුරෙකු හමුවිය"
    story_different = "වැසි සහිත කාලගුණය නිසා ගොවීන්ට විශාල ගැටලුවකට මුහුණ දීමට සිදුවිය"

    assert compute_jaccard_similarity(story_1, story_2) >= 0.85
    assert compute_jaccard_similarity(story_1, story_different) < 0.30


def test_spam_heuristic_scoring():
    clean_story = (
        "ගමේ පන්සලේ වෙසක් උත්සවය ඉතාමත් අලංකාර ලෙස පැවැත්විණි. "
        "බොහෝ දෙනෙක් පහන් කූඩු සහ තොරණ නැරඹීමට පැමිණ සිටියහ."
    )
    spam_story = "සිංහල සිංහල සිංහල සිංහල සිංහල සිංහල සිංහල සිංහල සිංහල සිංහල"

    assert compute_spam_score(clean_story) >= 0.20
    assert compute_spam_score(spam_story) < 0.20


def test_sha256_deterministic_hashing():
    text = "සිංහල භාෂා පර්යේෂණ දත්ත පද්ධතිය"
    hash1 = compute_sha256(normalize_text(text))
    hash2 = compute_sha256(normalize_text(text + "   "))
    assert hash1 == hash2
    assert len(hash1) == 64


def test_config_loading():
    cfg = load_config("config.yaml")
    assert "project" in cfg
    assert "validation" in cfg
    assert cfg["validation"]["unicode_range_start"] == 0x0D80
    assert cfg["validation"]["unicode_range_end"] == 0x0DFF
    assert cfg["validation"]["min_script_ratio"] == 0.90


def test_validate_text_entry_flow():
    # Story too short
    errors = validate_text_entry("කෙටි")
    assert any("at least 50 characters" in e for e in errors)

    # Valid story
    valid_story = (
        "පුරාණ ලංකාවේ රජවරුන් විසින් ඉදිකරන ලද මහා වැව් සහ දාගැබ් අදටත් අපගේ අභිමානය විදහා දක්වයි. "
        "එම නිර්මාණ තුළින් අපේ මුතුන්මිත්තන්ගේ දැනුම සහ තාක්ෂණික කුසලතාව ඉතා හොඳින් පැහැදිලි වේ."
    )
    errors = validate_text_entry(valid_story)
    assert len(errors) == 0


def test_bloom_filter_deduplication():
    # 1. Test insertion and membership
    sample_hashes = [
        compute_sha256(f"කතාව {i}") for i in range(100)
    ]
    bf = build_bloom_filter(sample_hashes, capacity=1000, error_rate=0.001)

    for h in sample_hashes:
        assert h in bf

    # 2. Test non-inserted item
    unseen_hash = compute_sha256("සම්පූර්ණයෙන්ම වෙනස් අලුත් කතාවක්")
    assert unseen_hash not in bf

    # 3. Test binary serialization roundtrip
    data = bf.to_bytes()
    assert isinstance(data, bytes)
    assert data[:4] == BloomFilter.MAGIC

    bf_restored = BloomFilter.from_bytes(data)
    for h in sample_hashes:
        assert h in bf_restored
    assert unseen_hash not in bf_restored

    # 4. Test file save and load roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        bloom_file = os.path.join(tmpdir, "hashes.bloom")
        bf.save(bloom_file)
        assert os.path.exists(bloom_file)

        bf_file_loaded = BloomFilter.load(bloom_file)
        assert len(bf_file_loaded.bit_array) == len(bf.bit_array)
        for h in sample_hashes:
            assert h in bf_file_loaded


def test_shannon_entropy_and_ngram_repetition_gating():
    # 1. Natural Sinhala prose has healthy character entropy (> 3.5 bits/char)
    natural_story = (
        "පුරාණ ලංකාවේ රජවරුන් විසින් ඉදිකරන ලද මහා වැව් සහ දාගැබ් අදටත් අපගේ අභිමානය විදහා දක්වයි. "
        "එම නිර්මාණ තුළින් අපේ මුතුන්මිත්තන්ගේ දැනුම සහ තාක්ෂණික කුසලතාව ඉතා හොඳින් පැහැදිලි වේ."
    )
    entropy_natural = compute_shannon_entropy(natural_story)
    assert entropy_natural >= 3.5

    # 2. Keyboard-smashing or character loops have artificially low entropy (< 2.0)
    mash_spam = "අඅඅබබබඅඅඅබබබඅඅඅබබබඅඅඅබබබඅඅඅබබබඅඅඅබබබඅඅඅබබබඅඅඅබබබඅඅඅබබබඅඅඅබබබ"
    entropy_spam = compute_shannon_entropy(mash_spam)
    assert entropy_spam < 2.0
    assert compute_spam_score(mash_spam) < 0.20

    # 3. Repeating phrase loops trigger bigram repetition penalty
    phrase_loop = "මම ගියා මම ගියා මම ගියා මම ගියා මම ගියා මම ගියා මම ගියා මම ගියා මම ගියා"
    score_phrase_loop = compute_spam_score(phrase_loop)
    assert score_phrase_loop < 0.20
