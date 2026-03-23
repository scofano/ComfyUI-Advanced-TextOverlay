import os
import re
import subprocess
import numpy as np
import torch
from html.parser import HTMLParser
from PIL import Image, ImageDraw, ImageFont

import imageio.v2 as imageio

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm isn't installed, fall back to a no-op wrapper
    def tqdm(x, **kwargs):
        return x

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

# Relative import so it works as a package module in ComfyUI
from . import animations
from .font_utils import get_available_fonts, get_font_variant_path


class InlineRichTextParser(HTMLParser):
    """Parses a small HTML-like subset into styled text runs."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.runs = []
        self._style_stack = [{"bold": False, "italic": False, "fill": None, "bg": None}]

    def handle_starttag(self, tag, attrs):
        tag = (tag or "").lower()
        attrs = dict(attrs or [])

        if tag == "br":
            self.runs.append({"text": "\n", "style": self._style_stack[-1].copy()})
            return

        new_style = self._style_stack[-1].copy()

        if tag == "b":
            new_style["bold"] = True
        elif tag == "i":
            new_style["italic"] = True
        elif tag == "span":
            span_style = self._parse_span_attrs(attrs)
            for key, value in span_style.items():
                if value is not None:
                    new_style[key] = value

        self._style_stack.append(new_style)

    def handle_startendtag(self, tag, attrs):
        tag = (tag or "").lower()
        self.handle_starttag(tag, attrs)
        if tag != "br" and len(self._style_stack) > 1:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if len(self._style_stack) > 1:
            self._style_stack.pop()

    def handle_data(self, data):
        if data:
            self.runs.append({"text": data, "style": self._style_stack[-1].copy()})

    def _parse_span_attrs(self, attrs):
        parsed = {"fill": None, "bg": None}

        if attrs.get("color"):
            parsed["fill"] = attrs.get("color")
        if attrs.get("fill"):
            parsed["fill"] = attrs.get("fill")
        if attrs.get("fg"):
            parsed["fill"] = attrs.get("fg")

        for key in ("bg", "background", "background-color"):
            if attrs.get(key):
                parsed["bg"] = attrs.get(key)

        style_text = attrs.get("style", "") or ""
        for part in style_text.split(";"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "color":
                parsed["fill"] = value
            elif key in ("background", "background-color"):
                parsed["bg"] = value

        return parsed

class TextOverlay:
    """
    Text overlay node with:
      - Fill/stroke alpha
      - Shadow and background box (both animate with opacity)
      - Pixel-perfect stroke alignment for MULTILINE text
      - Even stroke width for crisper edges
      - Default vertical_alignment = 'middle'
      - Batch animation: uses the first `animation_frames` frames for the animation,
        then holds the final pose for the rest of the video.
      - Mask animations removed. Renamed kinds: fade_in/fade_out, move_from_*.
    """

    _horizontal_alignments = ["left", "center", "right"]
    _vertical_alignments = ["top", "middle", "bottom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                # UI order (top → bottom)
                "text": ("STRING", {"multiline": True, "default": "the quick brown fox\njumps over the lazy dog"}),
                "all_caps": ("BOOLEAN", {"default": False}),

                # font, font-size, font color, font alpha
                "font": (get_available_fonts(), {"default": get_available_fonts()[0] if get_available_fonts() else "Arial"}),
                "font_size": ("INT", {"default": 32, "min": 1, "max": 9999, "step": 1}),
                "letter_spacing": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 50.0, "step": 0.5}),
                "font_alignment": (cls._horizontal_alignments, {"default": "center"}),
                "fill_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "fill_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # padding, alignment, offsets
                "padding": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 1}),
                "vertical_alignment": (cls._vertical_alignments, {"default": "middle"}),
                "y_shift": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "horizontal_alignment": (cls._horizontal_alignments, {"default": "center"}),
                "x_shift": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "line_spacing": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5}),

                # strokes
                "stroke_enable": ("BOOLEAN", {"default": True}),
                "stroke_color_hex": ("STRING", {"default": "#000000"}),
                "stroke_thickness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stroke_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # background box
                "bg_enable": ("BOOLEAN", {"default": False}),
                "bg_padding": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "bg_radius": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "bg_color_hex": ("STRING", {"default": "#000000"}),
                "bg_alpha": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # shadow
                "shadow_enable": ("BOOLEAN", {"default": False}),
                "shadow_distance": ("INT", {"default": 3, "min": -50, "max": 50, "step": 1}),
                "shadow_color_hex": ("STRING", {"default": "#000000"}),
                "shadow_alpha": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # --- animation controls ---
                "animate": ("BOOLEAN", {"default": False}),
                "animation_kind": ([
                    "fade_in", "fade_out",
                    "move_from_top", "move_from_bottom", "move_from_left", "move_from_right",
                ], {"default": "fade_in"}),
                "animation_frames": ("INT", {"default": 32, "min": 1, "max": 1000, "step": 1}),
                "animation_ease": (["linear","ease_in","ease_out","ease_in_out"], {"default": "ease_in_out"}),
                "animation_opacity_target": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # how long to wait before starting overlay (frames for this node, seconds for the Video node)
                "pause_frames_before_start": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_process"
    CATEGORY = "Advanced Text Overlay"

    # ---------------- helpers ----------------

    def hex_to_rgb(self, hex_color: str, fallback=(255, 255, 255)):
        try:
            hex_color = (hex_color or "").strip().lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join(ch * 2 for ch in hex_color)
            if len(hex_color) != 6:
                return fallback
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return fallback

    def _normalize_text(self, text: str) -> str:
        return (text or "").replace("\\n", "\n").replace("\\N", "\n")

    def _style_key(self, style):
        return (
            bool(style.get("bold")),
            bool(style.get("italic")),
            style.get("fill"),
            style.get("bg"),
        )

    def _font_signature(self, font_obj):
        try:
            return (font_obj.getname(), getattr(font_obj, "path", None), getattr(font_obj, "size", None))
        except Exception:
            return (None, getattr(font_obj, "path", None), getattr(font_obj, "size", None))

    def _load_font(self, font, font_size, bold=False, italic=False):
        if not hasattr(self, "_font_object_cache"):
            self._font_object_cache = {}

        cache_key = (font, font_size, bool(bold), bool(italic))
        if cache_key in self._font_object_cache:
            return self._font_object_cache[cache_key]

        font_path = get_font_variant_path(font, bold=bold, italic=italic)

        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        local_font_path = os.path.join(fonts_dir, font)
        if not os.path.exists(font_path) and os.path.exists(local_font_path):
            font_path = local_font_path

        try:
            loaded = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font: {e} — using default font")
            loaded = ImageFont.load_default()

        self._font_object_cache[cache_key] = loaded
        return loaded

    def _parse_rich_text(self, text, all_caps):
        normalized = self._normalize_text(text)
        parser = InlineRichTextParser()
        default_style = {"bold": False, "italic": False, "fill": None, "bg": None}

        try:
            parser.feed(normalized)
            parser.close()
            parsed_runs = parser.runs or [{"text": normalized, "style": default_style.copy()}]
        except Exception:
            parsed_runs = [{"text": normalized, "style": default_style.copy()}]

        merged_runs = []
        for run in parsed_runs:
            chunk = run.get("text", "")
            if all_caps:
                chunk = chunk.upper()
            if chunk == "":
                continue

            style = run.get("style", default_style).copy()
            if merged_runs and self._style_key(merged_runs[-1]["style"]) == self._style_key(style):
                merged_runs[-1]["text"] += chunk
            else:
                merged_runs.append({"text": chunk, "style": style})

        return merged_runs

    def _tokenize_runs(self, runs, font_name, font_size):
        tokens = []
        for run in runs:
            style = run["style"].copy()
            font_obj = self._load_font(
                font_name,
                font_size,
                bold=style.get("bold", False),
                italic=style.get("italic", False),
            )

            for part in re.findall(r"\n|[^\S\n]+|\S+", run["text"]):
                if part == "\n":
                    tokens.append({"text": "\n", "style": style.copy(), "font": font_obj, "newline": True})
                elif part:
                    tokens.append({"text": part, "style": style.copy(), "font": font_obj, "newline": False})
        return tokens

    def _merge_line_segments(self, tokens):
        merged = []
        for token in tokens:
            if token.get("newline"):
                continue

            item = {
                "text": token.get("text", ""),
                "style": token.get("style", {}).copy(),
                "font": token.get("font"),
            }
            if not item["text"]:
                continue

            if merged and self._style_key(merged[-1]["style"]) == self._style_key(item["style"]):
                merged[-1]["text"] += item["text"]
            else:
                merged.append(item)

        return merged

    def _measure_text_advance(self, draw, text, font, letter_spacing):
        if not text:
            return 0.0

        if not hasattr(self, "_measure_cache"):
            self._measure_cache = {}

        cache_key = (self._font_signature(font), text, float(letter_spacing))
        if cache_key in self._measure_cache:
            return self._measure_cache[cache_key]

        total = 0.0
        for i, ch in enumerate(text):
            total += draw.textlength(ch, font=font)
            if i < len(text) - 1:
                total += letter_spacing

        self._measure_cache[cache_key] = total
        return total

    def _measure_line_width(self, draw, segments, letter_spacing):
        nonempty = [seg for seg in segments if seg.get("text")]
        if not nonempty:
            return 0.0

        total = sum(self._measure_text_advance(draw, seg["text"], seg["font"], letter_spacing) for seg in nonempty)
        if len(nonempty) > 1:
            total += letter_spacing * (len(nonempty) - 1)
        return total

    def _compute_line_metrics(self, draw, segments, stroke_width, default_font):
        if not hasattr(self, "_line_metric_cache"):
            self._line_metric_cache = {}

        fonts = [seg["font"] for seg in segments if seg.get("font") is not None]
        if not fonts:
            fonts = [default_font]

        tops = []
        bottoms = []
        seen = set()
        for font in fonts:
            sig = (self._font_signature(font), int(stroke_width))
            if sig in seen:
                continue
            seen.add(sig)

            if sig not in self._line_metric_cache:
                bbox = draw.textbbox((0, 0), "Ag", font=font, stroke_width=stroke_width)
                self._line_metric_cache[sig] = (bbox[1], bbox[3], bbox[3] - bbox[1])

            top, bottom, height = self._line_metric_cache[sig]
            tops.append(top)
            bottoms.append(bottom)

        line_top = min(tops) if tops else 0
        line_bottom = max(bottoms) if bottoms else 0
        return line_top, line_bottom, line_bottom - line_top

    def _split_token_to_fit(self, draw, token, max_width, letter_spacing):
        text = token.get("text", "")
        if not text:
            return None, None

        if text.isspace():
            return None, None

        split_at = 0
        for i in range(1, len(text) + 1):
            candidate = text[:i]
            width = self._measure_text_advance(draw, candidate, token["font"], letter_spacing)
            if width <= max_width or i == 1:
                split_at = i
            else:
                break

        split_at = max(1, split_at)
        head_text = text[:split_at]
        tail_text = text[split_at:]

        head = token.copy()
        head["text"] = head_text

        tail = None
        if tail_text:
            tail = token.copy()
            tail["text"] = tail_text

        return head, tail

    def _wrap_styled_lines(self, draw, text, all_caps, font_name, font_size, max_width, letter_spacing):
        runs = self._parse_rich_text(text, all_caps)
        tokens = self._tokenize_runs(runs, font_name, font_size)

        if not tokens:
            return [[]]

        lines = []
        current = []
        idx = 0
        ended_with_newline = False
        max_width = max(1, int(round(max_width)))

        while idx < len(tokens):
            token = tokens[idx]

            if token.get("newline"):
                lines.append(self._merge_line_segments(current))
                current = []
                ended_with_newline = True
                idx += 1
                continue

            ended_with_newline = False

            if not current and token.get("text", "").isspace():
                idx += 1
                continue

            candidate = current + [token]
            candidate_width = self._measure_line_width(draw, self._merge_line_segments(candidate), letter_spacing)

            if not current and candidate_width > max_width:
                head, tail = self._split_token_to_fit(draw, token, max_width, letter_spacing)
                if head is not None:
                    current.append(head)
                    lines.append(self._merge_line_segments(current))
                    current = []
                idx += 1
                if tail is not None and tail.get("text"):
                    tokens.insert(idx, tail)
                continue

            if candidate_width <= max_width or not current:
                current.append(token)
                idx += 1
                continue

            lines.append(self._merge_line_segments(current))
            current = []

            if token.get("text", "").isspace():
                idx += 1

        if current or not lines or ended_with_newline:
            lines.append(self._merge_line_segments(current))

        return lines

    # ---------------- core drawing ----------------

    def _compute_layout(self, img_w, img_h, draw, text, all_caps, font_name, stroke_width, padding,
                        h_align, v_align, x_shift, y_shift, line_spacing, letter_spacing, font_size, use_cache):
        cache_key = (
            img_w, img_h, text, bool(all_caps), font_name, int(font_size), int(stroke_width),
            int(padding), float(line_spacing), float(letter_spacing)
        )

        need_recompute = not (hasattr(self, "_cached") and self._cached is not None and use_cache and self._cached.get("key") == cache_key)

        if need_recompute:
            default_font = self._load_font(font_name, font_size)
            lines = self._wrap_styled_lines(draw, text, all_caps, font_name, font_size, img_w - 2 * padding, letter_spacing)
            widths, tops, heights = [], [], []

            for line in lines:
                widths.append(self._measure_line_width(draw, line, letter_spacing))
                line_top, _line_bottom, line_height = self._compute_line_metrics(draw, line, stroke_width, default_font)
                tops.append(line_top)
                heights.append(line_height)

            block_w = max(widths) if widths else 0
            block_h = (sum(heights) + (len(heights) - 1) * line_spacing) if heights else 0

            self._cached = {
                "key": cache_key,
                "lines": lines,
                "widths": widths,
                "tops": tops,
                "heights": heights,
                "block_w": block_w,
                "block_h": block_h,
            }

        lines = self._cached["lines"]
        widths = self._cached["widths"]
        tops = self._cached["tops"]
        heights = self._cached["heights"]
        block_w = self._cached["block_w"]
        block_h = self._cached["block_h"]

        if h_align == "left":
            x0 = padding
        elif h_align == "center":
            x0 = (img_w - block_w) / 2
        else:
            x0 = img_w - block_w - padding

        if v_align == "top":
            visual_top_y = padding
        elif v_align == "middle":
            visual_top_y = (img_h - block_h) / 2
        else:
            visual_top_y = img_h - block_h - padding

        x0 = int(round(x0 + x_shift))
        visual_top_y = int(round(visual_top_y + y_shift))

        return lines, widths, heights, tops, block_w, block_h, x0, visual_top_y

    def draw_text(
        self,
        image,
        text,
        all_caps,
        font_size,
        letter_spacing,
        font,
        fill_color_hex,
        fill_alpha,
        stroke_enable,
        stroke_color_hex,
        stroke_alpha,
        stroke_thickness,
        padding,
        horizontal_alignment,
        vertical_alignment,
        x_shift,
        y_shift,
        line_spacing,
        bg_enable,
        bg_color_hex,
        bg_alpha,
        bg_padding,
        bg_radius,
        shadow_enable,
        shadow_color_hex,
        shadow_alpha,
        shadow_distance,
        font_alignment,
        use_cache=False,
        opacity_scale=1.0,
        dx=0,
        dy=0
    ):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        loaded_font = self._load_font(font, font_size)
        draw = ImageDraw.Draw(image, "RGBA")

        opacity_scale = max(0.0, min(1.0, float(opacity_scale)))
        fill_alpha = max(0.0, min(1.0, float(fill_alpha) * opacity_scale))
        stroke_alpha = max(0.0, min(1.0, float(stroke_alpha) * opacity_scale))
        x_shift = int(round(x_shift + dx))
        y_shift = int(round(y_shift + dy))

        sw = int(round(font_size * stroke_thickness * 0.5)) if stroke_enable else 0
        if sw % 2 == 1 and sw > 0:
            sw += 1

        (lines, widths, heights, tops, block_w, block_h,
         x0, visual_top_y) = self._compute_layout(
            image.width, image.height, draw, text, all_caps, font, sw,
            padding, horizontal_alignment, vertical_alignment, x_shift, y_shift,
            line_spacing, letter_spacing, font_size, use_cache
        )

        def _line_offset(i):
            if font_alignment == "left":
                return 0
            elif font_alignment == "center":
                return int(round((block_w - widths[i]) / 2))
            else:
                return int(round(block_w - widths[i]))

        def _positioned_segments(line_segments, x_start):
            positioned = []
            nonempty = [seg for seg in line_segments if seg.get("text")]
            xx = x_start
            for idx, seg in enumerate(nonempty):
                seg_w = self._measure_text_advance(draw, seg["text"], seg["font"], letter_spacing)
                positioned.append((seg, xx, seg_w))
                xx += seg_w
                if idx < len(nonempty) - 1:
                    xx += letter_spacing
            return positioned

        def _draw_segment_chars(draw_ctx, seg, start_x, baseline_y, color_rgba, stroke_rgba=None, stroke_width=0, dx_extra=0, dy_extra=0):
            text_value = seg.get("text", "")
            if not text_value:
                return

            xx = start_x
            for ch_idx, ch in enumerate(text_value):
                kwargs = {"font": seg["font"]}
                if stroke_rgba is not None and stroke_width > 0:
                    kwargs["stroke_width"] = stroke_width
                    kwargs["stroke_fill"] = stroke_rgba

                draw_ctx.text((xx + dx_extra, baseline_y + dy_extra), ch, fill=color_rgba, **kwargs)

                char_w = draw.textlength(ch, font=seg["font"])
                if ch_idx < len(text_value) - 1:
                    xx += char_w + letter_spacing
                else:
                    xx += char_w

        # Background (animated alpha)
        if bg_enable and block_w > 0 and block_h > 0:
            br, bgc, bb = self.hex_to_rgb(bg_color_hex)
            ba = int(max(0.0, min(1.0, float(bg_alpha) * opacity_scale)) * 255)
            rect = [x0 - bg_padding, visual_top_y - bg_padding,
                    x0 + block_w + bg_padding, visual_top_y + block_h + bg_padding]
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay, "RGBA")
            try:
                od.rounded_rectangle(rect, radius=max(0, int(bg_radius)), fill=(br, bgc, bb, ba))
            except Exception:
                od.rectangle(rect, fill=(br, bgc, bb, ba))
            image = Image.alpha_composite(image, overlay)

        inline_bg_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        inline_bg_draw = ImageDraw.Draw(inline_bg_overlay, "RGBA")

        inline_bg_alpha = int(max(0.0, min(1.0, float(bg_alpha) * opacity_scale)) * 255)
        inline_bg_pad_x = max(1, int(round(font_size * 0.10)))
        inline_bg_pad_y = max(1, int(round(font_size * 0.06)))
        inline_bg_radius = max(0, int(round(min(bg_radius, font_size * 0.25))))

        yy_top = visual_top_y
        for i, line in enumerate(lines):
            baseline_y = yy_top - tops[i]
            x_line = x0 + _line_offset(i)
            positioned = _positioned_segments(line, x_line)
            for seg, seg_x, _seg_w in positioned:
                bg_hex = seg["style"].get("bg")
                if not bg_hex:
                    continue
                br, bgc, bb = self.hex_to_rgb(bg_hex, fallback=self.hex_to_rgb(bg_color_hex))
                bbox = draw.textbbox((seg_x, baseline_y), seg["text"], font=seg["font"], stroke_width=sw)
                rect = [
                    bbox[0] - inline_bg_pad_x,
                    bbox[1] - inline_bg_pad_y,
                    bbox[2] + inline_bg_pad_x,
                    bbox[3] + inline_bg_pad_y,
                ]
                try:
                    inline_bg_draw.rounded_rectangle(rect, radius=inline_bg_radius, fill=(br, bgc, bb, inline_bg_alpha))
                except Exception:
                    inline_bg_draw.rectangle(rect, fill=(br, bgc, bb, inline_bg_alpha))
            yy_top += int(round(heights[i] + line_spacing))

        image = Image.alpha_composite(image, inline_bg_overlay)

        # Shadow (animated alpha)
        if shadow_enable and block_w > 0 and block_h > 0:
            sh_r, sh_g, sh_b = self.hex_to_rgb(shadow_color_hex)
            sh_a = int(max(0.0, min(1.0, float(shadow_alpha) * opacity_scale)) * 255)
            sdx = sdy = int(shadow_distance)
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay, "RGBA")

            yy_top = visual_top_y
            for i, line in enumerate(lines):
                baseline_y = yy_top - tops[i]
                x_line = x0 + _line_offset(i)
                for seg, seg_x, _seg_w in _positioned_segments(line, x_line):
                    _draw_segment_chars(
                        od,
                        seg,
                        seg_x,
                        baseline_y,
                        (sh_r, sh_g, sh_b, sh_a),
                        dx_extra=sdx,
                        dy_extra=sdy,
                    )
                yy_top += int(round(heights[i] + line_spacing))

            image = Image.alpha_composite(image, overlay)

        # Stroke + Fill
        fr, fg, fb = self.hex_to_rgb(fill_color_hex)
        fa = int(max(0.0, min(1.0, fill_alpha)) * 255)
        sr, sg, sb = self.hex_to_rgb(stroke_color_hex)
        sa = int(max(0.0, min(1.0, stroke_alpha)) * 255)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay, "RGBA")

        yy_top = visual_top_y
        for i, line in enumerate(lines):
            baseline_y = yy_top - tops[i]
            x_line = x0 + _line_offset(i)

            for seg, seg_x, _seg_w in _positioned_segments(line, x_line):
                seg_r, seg_g, seg_b = self.hex_to_rgb(seg["style"].get("fill"), fallback=(fr, fg, fb))

                if sw > 0 and sa > 0:
                    _draw_segment_chars(
                        od,
                        seg,
                        seg_x,
                        baseline_y,
                        (seg_r, seg_g, seg_b, sa),
                        stroke_rgba=(sr, sg, sb, sa),
                        stroke_width=sw,
                    )

                if fa > 0:
                    _draw_segment_chars(
                        od,
                        seg,
                        seg_x,
                        baseline_y,
                        (seg_r, seg_g, seg_b, fa),
                    )

            yy_top += int(round(heights[i] + line_spacing))

        image = Image.alpha_composite(image, overlay)

        return image.convert("RGB")

    # ---------------- Comfy entrypoint ----------------

    def batch_process(
        self,
        image,
        text,
        all_caps,
        font_size,
        letter_spacing,
        font,
        fill_color_hex,
        fill_alpha,
        stroke_enable,
        stroke_color_hex,
        stroke_alpha,
        stroke_thickness,
        padding,
        horizontal_alignment,
        vertical_alignment,
        x_shift,
        y_shift,
        line_spacing,
        bg_enable,
        bg_color_hex,
        bg_alpha,
        bg_padding,
        bg_radius,
        shadow_enable,
        shadow_color_hex,
        shadow_alpha,
        shadow_distance,
        font_alignment,
        animate=False,
        animation_kind='fade_in',
        animation_frames=24,
        animation_ease='ease_out',
        animation_opacity_target=1.0,
        pause_frames_before_start=0,
    ):
        """
        Single image (H,W,C):
          - animate=False  -> one output image
          - animate=True   -> returns exactly `animation_frames` frames

        Batch (B,H,W,C) video:
          - animate=False  -> draw per-frame with no animation changes
          - animate=True   -> animation starts after `pause_before_start` frames,
                              then runs for up to `animation_frames` frames.
                              Frames after that hold the final pose.

        NOTE: This node has no FPS info; here `pause_before_start` is effectively
        in frames (not seconds). For 30fps video and a 1s pause, use 30.
        """

        pause_frames = max(0, int(pause_frames_before_start))

        # Single image (H, W, C)
        if len(image.shape) == 3:
            np_img = image.cpu().numpy()
            pil_img = Image.fromarray((np_img * 255).astype(np.uint8))

            if not animate:
                # No timeline here → pause_before_start is ignored
                out_img = self.draw_text(
                    pil_img, text, all_caps,
                    font_size, letter_spacing, font,
                    fill_color_hex, fill_alpha,
                    stroke_enable,
                    stroke_color_hex, stroke_alpha, stroke_thickness,
                    padding, horizontal_alignment, vertical_alignment,
                    x_shift, y_shift, line_spacing,
                    bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
                    shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance, font_alignment,
                    use_cache=False,
                )
                out = np.array(out_img).astype(np.float32) / 255.0
                return (torch.tensor(out),)

            T = max(1, int(animation_frames))
            outs = []

            # Prime layout cache (fills self._cached)
            _ = self.draw_text(
                pil_img, text, all_caps,
                font_size, letter_spacing, font,
                fill_color_hex, 1.0,
                stroke_enable,
                stroke_color_hex, 1.0, stroke_thickness,
                padding, horizontal_alignment, vertical_alignment,
                x_shift, y_shift, line_spacing,
                bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
                shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance, font_alignment,
                use_cache=False,
            )
            use_cache = True

            for t_idx in range(T):
                if t_idx < pause_frames:
                    # Before the pause ends: no text overlay at all
                    out_img = pil_img.copy()
                else:
                    active_frames = max(1, T - pause_frames)
                    local_idx = t_idx - pause_frames
                    eff_local = min(local_idx, active_frames - 1)

                    p = animations.progress(eff_local, max(1, active_frames - 1), animation_ease)
                    op = animations.compute_opacity(animation_kind, p, float(animation_opacity_target))
                    dx, dy = animations.compute_offsets(animation_kind, p, pil_img.width, pil_img.height)

                    out_img = self.draw_text(
                        pil_img, text, all_caps,
                        font_size, letter_spacing, font,
                        fill_color_hex, fill_alpha,
                        stroke_enable,
                        stroke_color_hex, stroke_alpha, stroke_thickness,
                        padding, horizontal_alignment, vertical_alignment,
                        x_shift, y_shift, line_spacing,
                        bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
                        shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance, font_alignment,
                        use_cache=use_cache,
                        opacity_scale=op,
                        dx=dx,
                        dy=dy,
                    )

                outs.append(np.array(out_img).astype(np.float32) / 255.0)

            return (torch.tensor(np.stack(outs)),)

        # Batch (B, H, W, C)
        if not (hasattr(image, "shape") and len(image.shape) == 4):
            raise ValueError("Unsupported image tensor shape")

        B, H, W, C = image.shape

        # Non-animated batch
        if not animate:
            out_list = []
            for i in range(B):
                np_img = image[i].cpu().numpy()
                pil_img = Image.fromarray((np_img * 255).astype(np.uint8))

                if i < pause_frames:
                    # Pass-through until pause is over
                    out_img = pil_img
                else:
                    out_img = self.draw_text(
                        pil_img, text, all_caps,
                        font_size, letter_spacing, font,
                        fill_color_hex, fill_alpha,
                        stroke_enable,
                        stroke_color_hex, stroke_alpha, stroke_thickness,
                        padding, horizontal_alignment, vertical_alignment,
                        x_shift, y_shift, line_spacing,
                        bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
                        shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance, font_alignment,
                        use_cache=False,
                    )

                out_list.append(np.array(out_img).astype(np.float32) / 255.0)
            return (torch.tensor(np.stack(out_list)),)

        # Animated batch: animate on frames [pause_frames .. pause_frames+T-1],
        # then hold on frames after that
        T = max(1, int(animation_frames))

        # Prime cache once using first frame
        np_img0 = image[0].cpu().numpy()
        pil_img0 = Image.fromarray((np_img0 * 255).astype(np.uint8))
        _ = self.draw_text(
            pil_img0, text, all_caps,
            font_size, letter_spacing, font,
            fill_color_hex, 1.0,
            stroke_enable,
            stroke_color_hex, 1.0, stroke_thickness,
            padding, horizontal_alignment, vertical_alignment,
            x_shift, y_shift, line_spacing,
            bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
            shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance, font_alignment,
            use_cache=False,
        )
        use_cache = True

        out_list = []
        for i in range(B):
            np_img = image[i].cpu().numpy()
            pil_img = Image.fromarray((np_img * 255).astype(np.uint8))

            if i < pause_frames:
                # No overlay yet
                out_img = pil_img
            else:
                eff_idx = i - pause_frames
                eff_t = min(eff_idx, T - 1)  # frames beyond animation hold the last pose

                p = animations.progress(eff_t, max(1, T - 1), animation_ease)
                op = animations.compute_opacity(animation_kind, p, float(animation_opacity_target))
                dx, dy = animations.compute_offsets(animation_kind, p, pil_img.width, pil_img.height)

                out_img = self.draw_text(
                    pil_img, text, all_caps,
                    font_size, letter_spacing, font,
                    fill_color_hex, fill_alpha,
                    stroke_enable,
                    stroke_color_hex, stroke_alpha, stroke_thickness,
                    padding, horizontal_alignment, vertical_alignment,
                    x_shift, y_shift, line_spacing,
                    bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
                    shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance, font_alignment,
                    use_cache=use_cache,
                    opacity_scale=op,
                    dx=dx,
                    dy=dy,
                )

            out_list.append(np.array(out_img).astype(np.float32) / 255.0)

        return (torch.tensor(np.stack(out_list)),)

class TextOverlayVideo:
    """
    Video version of Advanced Text Overlay.

    - Input: full video file path (STRING).
    - Output: STRING with full path of processed video in ComfyUI's output folder.
    - Uses the same text / animation controls as TextOverlay, and animates
      over the first `animation_frames` frames, then holds the final pose.
    """

    _horizontal_alignments = TextOverlay._horizontal_alignments
    _vertical_alignments = TextOverlay._vertical_alignments

    # Make this an output node so the prompt has outputs
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        base = TextOverlay.INPUT_TYPES()["required"].copy()
        base.pop("image")

        # 🔧 rename the pause key for the video node
        base["pause_seconds_before_start"] = base.pop("pause_frames_before_start")

        required = {
            "video_path": ("STRING", {"multiline": False, "default": ""}),
            "filename_prefix": ("STRING", {"default": "TxtOver"}),
            "delete_original": ("BOOLEAN", {"default": False}),
        }
        required.update(base)

        return {"required": required}


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "process_video"
    CATEGORY = "Advanced Text Overlay"

    def _get_output_dir(self):
        # Default ComfyUI output folder if available
        try:
            import folder_paths
            return folder_paths.get_output_directory()
        except Exception:
            # Fallback if run standalone
            out_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(out_dir, exist_ok=True)
            return out_dir

    def _make_unique_path(self, out_dir, filename_prefix, src_path):
        src_base = os.path.splitext(os.path.basename(src_path))[0]
        base_name = f"{filename_prefix}_{src_base}.mp4"
        out_full = os.path.join(out_dir, base_name)
        idx = 1
        while os.path.exists(out_full):
            base_name = f"{filename_prefix}_{src_base}_{idx}.mp4"
            out_full = os.path.join(out_dir, base_name)
            idx += 1
        return out_full

    def process_video(
    self,
    video_path,
    filename_prefix,
    delete_original,
    text,
    all_caps,
    font,
    font_size,
    letter_spacing,
    font_alignment,
    fill_color_hex,
    fill_alpha,
    padding,
    vertical_alignment,
    y_shift,
    horizontal_alignment,
    x_shift,
    line_spacing,
    stroke_enable,
    stroke_color_hex,
    stroke_thickness,
    stroke_alpha,
    bg_enable,
    bg_padding,
    bg_radius,
    bg_color_hex,
    bg_alpha,
    shadow_enable,
    shadow_distance,
    shadow_color_hex,
    shadow_alpha,
    animate,
    animation_kind,
    animation_frames,
    animation_ease,
    animation_opacity_target,
    pause_seconds_before_start,
):

        """
        Reads the video frame by frame, applies the same text overlay logic as the
        batch TextOverlay, and writes a new video file.
        Returns the full path string to the new video.

        Shows progress in:
          - ComfyUI (ProgressBar)
          - Console (tqdm)

        After writing the processed video, we mux the original audio track
        from `video_path` into the output file using ffmpeg, if available.
        """
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        out_dir = self._get_output_dir()
        out_path = self._make_unique_path(out_dir, filename_prefix, video_path)

        # Reuse your existing text overlay logic on each frame
        overlay = TextOverlay()

        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta.get("fps", 30)

        # Try to get a sane total frame count
        nframes_meta = meta.get("nframes", None)
        duration = meta.get("duration", None)  # seconds, if available
        
        # Convert pause_seconds_before_start (seconds) to frames
        try:
            pause_frames = max(0, int(round(float(pause_seconds_before_start) * float(fps))))
        except Exception:
            pause_frames = max(0, int(pause_seconds_before_start))

        total_frames = None

        # 1) Trust nframes only if it's a reasonable integer
        if isinstance(nframes_meta, (int, float)) and 0 < nframes_meta < 1e8:
            total_frames = int(nframes_meta)

        # 2) Otherwise, estimate from duration * fps if we have that
        elif isinstance(duration, (int, float)) and duration > 0 and fps > 0:
            total_frames = int(duration * fps)

        # ComfyUI progress bar
        comfy_pbar = None
        if ProgressBar is not None and isinstance(total_frames, int) and total_frames > 0:
            comfy_pbar = ProgressBar(total_frames)

        # tqdm progress bar
        if isinstance(total_frames, int) and total_frames > 0:
            frame_iter = tqdm(reader, total=total_frames, desc="TextOverlayVideo")
        else:
            frame_iter = tqdm(reader, desc="TextOverlayVideo")

        T = max(1, int(animation_frames)) if animate else 1

        writer = imageio.get_writer(
            out_path,
            fps=fps,
            macro_block_size=1  # avoid auto-resizing to multiples of 16
        )

        try:
            for i, frame in enumerate(frame_iter):
                pil_img = Image.fromarray(frame)

                if i < pause_frames:
                    # Before pause: pass the frame through with no overlay
                    out_img = pil_img
                else:
                    if animate:
                        # Animation timing: animate on frames [pause_frames .. pause_frames+T-1],
                        # then hold last pose afterwards
                        eff_idx = i - pause_frames
                        eff_t = min(eff_idx, T - 1)
                        p = animations.progress(eff_t, max(1, T - 1), animation_ease)
                        op = animations.compute_opacity(animation_kind, p, float(animation_opacity_target))
                        dx, dy = animations.compute_offsets(animation_kind, p, pil_img.width, pil_img.height)
                    else:
                        op = 1.0
                        dx = dy = 0

                    use_cache = (i > pause_frames)

                    out_img = overlay.draw_text(
                        pil_img,
                        text,
                        all_caps,
                        font_size,
                        letter_spacing,
                        font,
                        fill_color_hex,
                        fill_alpha,
                        stroke_enable,
                        stroke_color_hex,
                        stroke_alpha,
                        stroke_thickness,
                        padding,
                        horizontal_alignment,
                        vertical_alignment,
                        x_shift,
                        y_shift,
                        line_spacing,
                        bg_enable,
                        bg_color_hex,
                        bg_alpha,
                        bg_padding,
                        bg_radius,
                        shadow_enable,
                        shadow_color_hex,
                        shadow_alpha,
                        shadow_distance,
                        font_alignment,
                        use_cache=use_cache,
                        opacity_scale=op,
                        dx=dx,
                        dy=dy,
                    )

                writer.append_data(np.array(out_img))

                # Update ComfyUI progress
                if comfy_pbar is not None:
                    comfy_pbar.update(1)


        finally:
            writer.close()
            reader.close()

        # ---- NEW: mux original audio into the processed video using ffmpeg ----
        try:
            # We create a temporary output file, then replace the original out_path
            tmp_out = out_path + ".tmp_audio.mp4"

            # ffmpeg command:
            #   - input 0: processed video (no audio)
            #   - input 1: original video (with audio)
            #   - copy streams without re-encoding: -c copy
            #   - map video from 0, audio from 1
            cmd = [
                "ffmpeg",
                "-y",                   # overwrite without asking
                "-i", out_path,
                "-i", video_path,
                "-c", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                tmp_out,
            ]

            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if completed.returncode == 0:
                os.replace(tmp_out, out_path)
            else:
                # If something goes wrong, keep the silent video and print the error
                print("[TextOverlayVideo] ffmpeg failed to mux audio, keeping silent video.")
                print(completed.stderr.decode("utf-8", errors="ignore"))

        except Exception as e:
            # Fail gracefully: overlay still works, just no audio
            print(f"[TextOverlayVideo] Could not mux audio from original video: {e}")
        
        # NEW: optionally delete the original input video after processing is finished
        if delete_original:
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"[TextOverlayVideo] Failed to delete original video '{video_path}': {e}")

        return (out_path,)

NODE_CLASS_MAPPINGS = {
    "Advanced Text Overlay": TextOverlay,
    "Advanced Text Overlay - Video": TextOverlayVideo,
}
