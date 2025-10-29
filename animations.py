from typing import Tuple

# Easing -------------------------------------------------------------
def progress(t: int, T_minus1: int, ease: str) -> float:
    if T_minus1 <= 0:
        return 1.0
    x = float(t) / float(T_minus1)
    if ease == "linear":
        return x
    if ease == "ease_in":
        return x * x
    if ease == "ease_out":
        return 1 - (1 - x) * (1 - x)
    if ease == "ease_in_out":
        if x < 0.5:
            return 2 * x * x
        return 1 - (-2 * x + 2) ** 2 / 2
    return x

# Opacity ------------------------------------------------------------
def compute_opacity(kind: str, p: float, target: float) -> float:
    """
    kind: 'fade_in' | 'fade_out' | others
    p in [0,1], target in [0,1]
    """
    target = max(0.0, min(1.0, float(target)))
    if kind == "fade_out":
        return target * (1.0 - p)
    if kind == "fade_in":
        return target * p
    return target

# Offsets ------------------------------------------------------------
def compute_offsets(kind: str, p: float, img_w: int, img_h: int) -> Tuple[int, int]:
    """
    Move from off-screen-ish towards the final position.
    p=0 => max offset, p=1 => 0 offset.
    """
    dx = dy = 0.0
    if kind == "move_from_left":
        dx = -(img_w * (1.0 - p) * 0.5)
    elif kind == "move_from_right":
        dx = (img_w * (1.0 - p) * 0.5)
    elif kind == "move_from_top":
        dy = -(img_h * (1.0 - p) * 0.5)
    elif kind == "move_from_bottom":
        dy = (img_h * (1.0 - p) * 0.5)
    return int(round(dx)), int(round(dy))
