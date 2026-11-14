"""Unit tests for the rule-based parts of NavAssist (no models needed)."""
import numpy as np
from navassist import preprocess
from navassist.reasoning import zone_of, priority_of, analyse, decide, zone_occupancy, free_direction
from navassist.speech import Speaker
from navassist.depth import distance_band

W, H = 900, 600


def cmap_with(box=None, value=0.9, background=0.1):
    m = np.full((H, W), background, dtype=np.float32)
    if box:
        x1, y1, x2, y2 = box
        m[y1:y2, x1:x2] = value
    return m


def det(label, box, conf=0.9):
    return {"label": label, "conf": conf, "box": box}


# --- zones -------------------------------------------------------------
def test_zone_left_centre_right():
    assert zone_of((10, 0, 200, 100), W) == "left"
    assert zone_of((400, 0, 500, 100), W) == "centre"
    assert zone_of((700, 0, 890, 100), W) == "right"


def test_wide_object_counts_as_centre():
    assert zone_of((100, 0, 450, 100), W) == "centre"


# --- distance bands and priority ----------------------------------------
def test_distance_bands():
    assert distance_band(0.85) == "near"
    assert distance_band(0.55) == "mid"
    assert distance_band(0.10) == "far"


def test_priority_table():
    assert priority_of("centre", "near") == "HIGH"
    assert priority_of("centre", "mid") == "MEDIUM"
    assert priority_of("left", "near") == "MEDIUM"
    assert priority_of("right", "far") == "LOW"
    assert priority_of("centre", "far") == "LOW"


# --- full decisions ---------------------------------------------------------
def test_near_chair_in_centre_gives_stop_and_free_side():
    box = (380, 250, 520, 520)
    m = cmap_with(box, 0.9)
    m[200:, 620:] = 0.6                      # right side more cluttered than left
    hz = analyse([det("chair", box)], m)
    d = decide(hz, zone_occupancy(m), [])
    assert d["level"] == "HIGH"
    assert d["message"] == "Stop. Chair ahead. Move left."


def test_far_object_is_not_announced():
    box = (420, 250, 470, 300)
    m = cmap_with(box, 0.2)
    hz = analyse([det("person", box)], m)
    assert hz[0]["priority"] == "LOW"
    assert decide(hz, zone_occupancy(m), [])["message"] is None


def test_side_obstacle_gives_side_warning():
    box = (650, 200, 880, 580)
    m = cmap_with(box, 0.9)
    d = decide(analyse([det("bench", box)], m), zone_occupancy(m), [])
    assert d["level"] == "MEDIUM" and d["message"] == "Bench on your right."


def test_blocked_on_all_sides():
    m = cmap_with(None, background=0.9)
    hz = analyse([det("person", (350, 100, 550, 590)), det("chair", (20, 100, 280, 590)),
                  det("bench", (640, 100, 890, 590))], m)
    d = decide(hz, zone_occupancy(m), [])
    assert d["message"] == "Stop. Person ahead. Path blocked."


def test_wide_obstacle_blocks_all_zones_it_covers():
    m = cmap_with(None, background=0.9)
    hz = analyse([det("bus", (20, 100, 880, 590))], m)
    assert free_direction(hz, zone_occupancy(m)) is None


def test_text_added_when_no_high_hazard():
    m = cmap_with()
    texts = [{"text": "EXIT", "conf": 0.95, "box": (700, 50, 800, 100)}]
    assert decide([], zone_occupancy(m), texts)["message"] == "Sign reads EXIT."


def test_text_suppressed_during_high_alert():
    box = (380, 250, 520, 520)
    m = cmap_with(box, 0.9)
    texts = [{"text": "EXIT", "conf": 0.95, "box": (700, 50, 800, 100)}]
    msg = decide(analyse([det("chair", box)], m), zone_occupancy(m), texts)["message"]
    assert "EXIT" not in msg


def test_hazards_sorted_by_urgency():
    m = cmap_with((380, 250, 520, 520), 0.9)
    m[250:520, 650:800] = 0.5
    hz = analyse([det("person", (650, 250, 800, 520)), det("chair", (380, 250, 520, 520))], m)
    assert hz[0]["label"] == "chair"


# --- speech repeat suppression ---------------------------------------------------
def test_repeat_suppression():
    s = Speaker(enabled=False, cooldown=3.0)
    assert s.say("Chair ahead.", now=0.0)
    assert not s.say("Chair ahead.", now=1.5)
    assert s.say("Chair ahead.", now=3.5)
    assert s.say("Person on your left.", now=1.6)


def test_empty_message_not_spoken():
    assert not Speaker(enabled=False).say(None, now=0.0)


# --- preprocessing quality checks --------------------------------------------------
def test_blurred_and_dark_frames_rejected():
    rng = np.random.default_rng(0)
    sharp = (rng.random((300, 300, 3)) * 255).astype(np.uint8)
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    dark = (rng.random((300, 300, 3)) * 20).astype(np.uint8)
    assert preprocess.check_quality(sharp)["usable"]
    assert preprocess.check_quality(flat)["reason"] == "blurred"
    assert preprocess.check_quality(dark)["usable"] is False


def test_resize_limits_long_side():
    big = np.zeros((2000, 3000, 3), dtype=np.uint8)
    assert max(preprocess.resize_frame(big).shape[:2]) == 960


def test_hand_from_bottom_edge_is_ignored():
    m = cmap_with((100, 300, 500, 600), 0.9)
    assert analyse([det("person", (100, 300, 500, 599))], m) == []


def test_pedestrian_with_upper_body_is_kept():
    m = cmap_with((380, 60, 520, 590), 0.9)
    hz = analyse([det("person", (380, 60, 520, 599))], m)
    assert hz and hz[0]["priority"] == "HIGH"
