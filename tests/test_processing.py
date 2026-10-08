from dataclasses import FrozenInstanceError, replace

import cv2
import numpy as np
import pytest

from arducam_photo import IspError, ProcessingRecipe, process_image
from arducam_photo.isp import process_raw


def test_recipe_immutable():
    with pytest.raises(FrozenInstanceError):
        ProcessingRecipe().black_level = 0


@pytest.mark.parametrize("name,black", [("minimal", 0), ("black_level", 25), ("reference", 25)])
def test_raw_recipe_black_level_channel_order_nonmutation(name, black):
    raw = np.zeros((8, 8), np.uint8)
    raw[0::2, 1::2] = 216
    before = raw.copy()
    recipe = ProcessingRecipe(name=name, black_level=25, apply_ccm=False)
    image = process_image(raw, recipe, raw=True)
    assert image[4, 4].tolist() == [0, 0, 216 - black]
    assert image.dtype == np.uint8 and np.array_equal(raw, before)
    assert np.array_equal(image, process_image(raw, recipe, raw=True))


@pytest.mark.parametrize("name", ["minimal", "black_level"])
def test_no_ccm_nonreference(name):
    image = process_image(np.full((8, 8), 100, np.uint8), ProcessingRecipe(name=name),
                          raw=True, ccm_path="nonexistent.json")
    assert image.shape == (8, 8, 3)


def test_reference_ccm_temperature(tmp_path):
    import json
    path = tmp_path / "ccm.json"
    identity = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    swap = [0, 0, 1, 0, 1, 0, 1, 0, 0]
    path.write_text(json.dumps({"ccms": [{"ct": 3000, "ccm": identity}, {"ct": 5000, "ccm": swap}]}))
    raw = np.zeros((8, 8), np.uint8)
    raw[0::2, 1::2] = 216
    low = process_image(raw, ProcessingRecipe(temperature=3000), raw=True, ccm_path=path)
    high = process_image(raw, ProcessingRecipe(temperature=5000), raw=True, ccm_path=path)
    middle = process_image(raw, ProcessingRecipe(temperature=4000), raw=True, ccm_path=path)
    assert low[4, 4, 2] >= 190 and low[4, 4, 0] < 10
    assert high[4, 4, 0] >= 190 and high[4, 4, 2] < 10
    assert 90 <= middle[4, 4, 0] <= 110 and 90 <= middle[4, 4, 2] <= 110
    assert raw[0, 1] == 216


def test_colour_default_identity_and_bgr_grayscale():
    original = np.full((10, 20, 3), [200, 50, 10], np.uint8)
    image = process_image(original, ProcessingRecipe(), ccm_path="missing.json")
    assert np.array_equal(image, original) and not np.shares_memory(image, original)
    gray = process_image(original, ProcessingRecipe(grayscale=True))
    assert np.array_equal(gray, cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))


@pytest.mark.parametrize("grayscale,threshold", [(False, "none"), (True, "none"), (False, "otsu"),
                                               (False, "adaptive")])
def test_optional_chain_reproducible_nonmutating(grayscale, threshold):
    original = np.random.default_rng(1).integers(0, 256, (40, 80, 3), dtype=np.uint8)
    before = original.copy()
    recipe = ProcessingRecipe(grayscale=grayscale, threshold=threshold, clahe_clip=2, max_dimension=30)
    a = process_image(original, recipe)
    b = process_image(original, recipe)
    assert np.array_equal(a, b) and np.array_equal(original, before)
    assert a.shape[:2] == (15, 30) and a.dtype == np.uint8
    assert a.ndim == (3 if not grayscale and threshold == "none" else 2)
    if threshold != "none":
        assert set(np.unique(a)).issubset({0, 255})


def test_reduction_never_enlarges():
    original = np.zeros((4, 8, 3), np.uint8)
    assert process_image(original, ProcessingRecipe(max_dimension=20)).shape == original.shape
    assert process_image(original, ProcessingRecipe(max_dimension=1)).shape == (1, 1, 3)


@pytest.mark.parametrize("change", [
    {"black_level": -1}, {"black_level": 256}, {"black_level": 1.5}, {"black_level": True},
    {"temperature": 0}, {"temperature": float("nan")}, {"temperature": True},
    {"clahe_clip": -1}, {"clahe_clip": float("inf")}, {"clahe_grid": 0}, {"clahe_grid": 2.5},
    {"clahe_grid": 65}, {"adaptive_block": 257},
    {"adaptive_block": 2}, {"adaptive_block": 4}, {"adaptive_c": float("nan")},
    {"threshold": "invalid"}, {"max_dimension": 0}, {"max_dimension": True},
    {"apply_ccm": 1}, {"grayscale": 1}, {"version": 2}, {"version": True}, {"name": "unknown"},
])
def test_invalid_recipe(change):
    with pytest.raises(IspError):
        process_image(np.zeros((8, 8, 3), np.uint8), replace(ProcessingRecipe(), **change))


@pytest.mark.parametrize("original,raw", [
    (None, False), (np.zeros((8, 8, 3), np.uint16), False),
    (np.zeros((8, 8), np.uint8), False), (np.zeros((8, 8, 4), np.uint8), False),
    (np.zeros((0, 8, 3), np.uint8), False), (np.zeros((8, 8, 3), np.uint8), True),
    (np.zeros((2, 8), np.uint8), True),
])
def test_invalid_original(original, raw):
    with pytest.raises(IspError):
        process_image(original, ProcessingRecipe(apply_ccm=False), raw=raw)


def test_process_raw_keyword_controls():
    raw = np.full((8, 8), 20, np.uint8)
    assert process_raw(raw).max() == 4
    assert process_raw(raw, black_level=0).max() == 20
    assert process_raw(raw, black_level=25).max() == 0
    assert np.all(raw == 20)
