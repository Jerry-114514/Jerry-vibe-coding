"""Frozen phase-A metrics shared by baselines and subsequent model inference."""
import math


def box_iou(a, b):
    if len(a) != 4 or len(b) != 4 or not all(math.isfinite(float(x)) for x in [*a, *b]):
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_box(width, height, area):
    """Centered rectangle, image aspect ratio, exact requested area when feasible."""
    if min(width, height, area) <= 0:
        raise ValueError("Width, height and area must be positive")
    effective_area = min(area, width * height)
    box_w = math.sqrt(effective_area * width / height)
    box_h = math.sqrt(effective_area * height / width)
    return [(width - box_w) / 2, (height - box_h) / 2,
            (width + box_w) / 2, (height + box_h) / 2]


def consistency_bit(bits, expected=5):
    if len(bits) != expected:
        raise ValueError(f"Expected {expected} paraphrases, got {len(bits)}")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Correctness bits must be binary")
    return int(all(bits))
