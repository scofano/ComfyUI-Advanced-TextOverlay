import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

class TextOverlay:
    """
    Text overlay node with:
      - Fill/stroke alpha
      - Shadow and background box
      - Pixel-perfect stroke alignment for MULTILINE text
      - Even stroke width for crisper edges
      - Default vertical_alignment = 'middle'
    """

    _horizontal_alignments = ["left", "center", "right"]
    _vertical_alignments = ["top", "middle", "bottom"]

    def __init__(self, device="cpu"):
        self.device = device
        self._loaded_font = None
        # cache (lines, widths, heights, tops, min_top, min_left, block_w, block_h)
        self._cached = None
        self._x = None
        self._y = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # keep image first so the node wiring stays obvious
                "image": ("IMAGE",),

                # UI order (top → bottom)
                "text": ("STRING", {"multiline": True, "default": "the quick brown fox\njumps over the lazy dog"}),
                "all_caps": ("BOOLEAN", {"default": False}),

                # font, font-size, font color, font alpha
                "font": ("STRING", {"default": "ariblk.ttf"}),
                "font_size": ("INT", {"default": 32, "min": 1, "max": 9999, "step": 1}),
                "letter_spacing": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 50.0, "step": 0.5}),
                "fill_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "fill_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # padding, hor_align, x_shift, vert align, y_shift, line_spacing
                "padding": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 1}),
                "vertical_alignment": (cls._vertical_alignments, {"default": "middle"}),
                "y_shift": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "horizontal_alignment": (cls._horizontal_alignments, {"default": "center"}),
                "x_shift": ("INT", {"default": 0, "min": -1024, "max": 1024, "step": 1}),
                "line_spacing": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5}),

                # all strokes
                "stroke_enable": ("BOOLEAN", {"default": True}),
                "stroke_color_hex": ("STRING", {"default": "#000000"}),
                "stroke_thickness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stroke_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # all boxes
                "bg_enable": ("BOOLEAN", {"default": False}),
                "bg_padding": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "bg_radius": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "bg_color_hex": ("STRING", {"default": "#000000"}),
                "bg_alpha": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),

                # all shadows
                "shadow_enable": ("BOOLEAN", {"default": False}),
                "shadow_distance": ("INT", {"default": 4, "min": -50, "max": 50, "step": 1}),
                "shadow_color_hex": ("STRING", {"default": "#000000"}),
                "shadow_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_process"
    CATEGORY = "image/text"

    # ---------------- helpers ----------------

    def hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.strip().lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(ch * 2 for ch in hex_color)
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def _normalize_text(self, text: str) -> str:
        # Convert literal "\n" to actual newlines
        return text.replace("\\n", "\n").replace("\\N", "\n")

    def _load_font(self, font, font_size):
        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        font_path = os.path.join(fonts_dir, font)
        if not os.path.exists(font_path):
            font_path = font  # fallback to system path or provided absolute path
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font: {e} — using default font")
            return ImageFont.load_default()

    def _wrap_lines(self, draw, text, font, max_width, padding, letter_spacing):
        """
        Greedy wrapping that respects explicit newlines and preserves spaces
        (including multiple consecutive spaces).
        """
        import re

        paragraphs = self._normalize_text(text).split("\n")
        out_lines = []

        for para in paragraphs:
            if para == "":
                out_lines.append("")
                continue

            tokens = re.findall(r"\S+|\s+", para)

            line = ""
            for tok in tokens:
                candidate = line + tok
                # include tracking in the width test
                char_count = max(0, len(candidate) - 1)
                candidate_px = draw.textlength(candidate, font=font) + char_count * letter_spacing
                if candidate_px <= (max_width - 2 * padding):
                    line = candidate
                else:
                    if line == "":
                        out_lines.append(candidate)
                        line = ""
                    else:
                        out_lines.append(line)
                        line = tok.lstrip()

            out_lines.append(line)

        return out_lines


    # ---------------- core drawing ----------------

    def _compute_layout(self, img_w, img_h, draw, text, font, stroke_width, padding,
                    h_align, v_align, x_shift, y_shift, line_spacing, letter_spacing, use_cache):
        """
        Manual multiline layout so stroke/fill line positions are identical.

        IMPORTANT: We compute the visual extents using per-line bbox plus letter spacing,
        so background boxes and centering reflect tracking.
        Returns:
          (lines, widths, heights, tops, min_top, min_left, block_w, block_h, x0, visual_top_y)
        """
        need_recompute = True
        if self._cached is not None and use_cache:
            # the last element of the cache is the letter_spacing it was computed with
            cached_letter_spacing = self._cached[-1]
            if cached_letter_spacing == letter_spacing:
                need_recompute = False

        if need_recompute:
            # wrap with spacing-aware width checks
            lines = self._wrap_lines(draw, text, font, img_w, padding, letter_spacing)

            widths, heights, tops = [], [], []
            lefts, rights_sp = [], []
            for ln in lines:
                # bbox includes stroke when stroke_width > 0
                l, t, r, b = draw.textbbox((0, 0), ln, font=font, stroke_width=stroke_width)
                w = (r - l)
                h = (b - t)
                # add extra visual width from inter-char spacing
                extra = max(0, len(ln) - 1) * letter_spacing
                widths.append(w + extra)
                heights.append(h)
                tops.append(t)
                lefts.append(l)
                rights_sp.append(r + extra)

            min_left = min(lefts) if lefts else 0
            max_right = max(rights_sp) if rights_sp else 0
            block_w = max_right - min_left if rights_sp else 0
            block_h = sum(heights) + (len(heights) - 1) * line_spacing if heights else 0
            min_top = min(tops) if tops else 0  # may be negative for ascenders

            # cache results to reuse on subsequent frames (append letter_spacing)
            self._cached = (lines, widths, heights, tops, min_top, min_left, block_w, block_h, letter_spacing)

        lines, widths, heights, tops, min_top, min_left, block_w, block_h, _ = self._cached

        # Horizontal anchor for the *visual* block (uses min_left/max_right w/ spacing)
        if h_align == "left":
            x0 = padding
        elif h_align == "center":
            x0 = (img_w - block_w) / 2
        else:  # right
            x0 = img_w - block_w - padding

        # Vertical anchor for the *visual* block
        if v_align == "top":
            visual_top_y = padding
        elif v_align == "middle":
            visual_top_y = (img_h - block_h) / 2
        else:  # bottom
            visual_top_y = img_h - block_h - padding

        # Snap + shifts
        x0 = int(round(x0 + x_shift))
        visual_top_y = int(round(visual_top_y + y_shift))

        return lines, widths, heights, tops, min_top, min_left, block_w, block_h, x0, visual_top_y


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
        use_cache=False,
    ):
        # Prepare
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        self._loaded_font = self._load_font(font, font_size)
        draw = ImageDraw.Draw(image, "RGBA")

        # Apply ALL CAPS if requested (after literal "\n" handling via _wrap_lines)
        if all_caps:
            # Uppercase before wrapping so measurement/layout reflect final glyphs
            text = text.upper()

        # Compute even stroke width (recommended for crisp outlines)
        sw = int(round(font_size * stroke_thickness * 0.5)) if stroke_enable else 0
        if sw % 2 == 1 and sw > 0:
            sw += 1

        # Layout (manual lines)
        (lines, widths, heights, tops, min_top, min_left, block_w, block_h,
         x0, visual_top_y) = self._compute_layout(
            image.width, image.height, draw, text, self._loaded_font, sw,
            padding, horizontal_alignment, vertical_alignment, x_shift, y_shift,
            line_spacing, letter_spacing, use_cache
        )
        
        # Convert the visual top into the baseline y for the first line
        first_line_baseline_y = visual_top_y - min_top

        # This is the x where we actually draw glyphs so that the visual left aligns to x0
        x_draw = x0 - min_left

        # Background box (perfectly aligned with visual extents)
        if bg_enable and block_w > 0 and block_h > 0:
            br, bgc, bb = self.hex_to_rgb(bg_color_hex)
            ba = int(max(0.0, min(1.0, bg_alpha)) * 255)
            rect = [x0 - bg_padding, visual_top_y - bg_padding,
                    x0 + block_w + bg_padding, visual_top_y + block_h + bg_padding]
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay, "RGBA")
            try:
                od.rounded_rectangle(rect, radius=max(0, int(bg_radius)), fill=(br, bgc, bb, ba))
            except Exception:
                od.rectangle(rect, fill=(br, bgc, bb, ba))
            image = Image.alpha_composite(image, overlay)

        # Shadow (per-line, before stroke/fill)
        if shadow_enable and block_w > 0 and block_h > 0:
            sh_r, sh_g, sh_b = self.hex_to_rgb(shadow_color_hex)
            sh_a = int(max(0.0, min(1.0, shadow_alpha)) * 255)
            sdx = sdy = int(shadow_distance)
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay, "RGBA")

            yy = first_line_baseline_y
            for ln, h in zip(lines, heights):
                xx = x_draw
                for ch in ln:
                    od.text((xx + sdx, yy + sdy), ch, font=self._loaded_font, fill=(sh_r, sh_g, sh_b, sh_a))
                    xx += draw.textlength(ch, font=self._loaded_font) + letter_spacing
                yy += int(round(h + line_spacing))

            image = Image.alpha_composite(image, overlay)


        # Stroke + Fill (per-line; identical geometry/positions)
        fr, fg, fb = self.hex_to_rgb(fill_color_hex)
        fa = int(max(0.0, min(1.0, fill_alpha)) * 255)
        sr, sg, sb = self.hex_to_rgb(stroke_color_hex)
        sa = int(max(0.0, min(1.0, stroke_alpha)) * 255)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay, "RGBA")

        yy = first_line_baseline_y
        for ln, h in zip(lines, heights):
            # draw one character at a time to inject letter spacing
            xx = x_draw
            for ch in ln:
                if sw > 0 and sa > 0:
                    od.text(
                        (xx, yy),
                        ch,
                        font=self._loaded_font,
                        fill=(sr, sg, sb, sa),
                        stroke_width=sw,
                        stroke_fill=(sr, sg, sb, sa),
                    )
                if fa > 0:
                    od.text(
                        (xx, yy),
                        ch,
                        font=self._loaded_font,
                        fill=(fr, fg, fb, fa),
                    )

                # advance by glyph width + custom letter spacing
                xx += draw.textlength(ch, font=self._loaded_font) + letter_spacing

            yy += int(round(h + line_spacing))

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
    ):
        """Handles both single and batch image processing (with layout caching across frames)."""

        # Single image
        if len(image.shape) == 3:
            np_img = image.cpu().numpy()
            pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
            out_img = self.draw_text(
                pil_img, text, all_caps,
                font_size, letter_spacing, font,
                fill_color_hex, fill_alpha,
                stroke_enable,
                stroke_color_hex, stroke_alpha, stroke_thickness,
                padding, horizontal_alignment, vertical_alignment,
                x_shift, y_shift, line_spacing,
                bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
                shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance,
                use_cache=False,
            )
            out = np.array(out_img).astype(np.float32) / 255.0
            return (torch.tensor(out),)

        # Batch of images
        else:
            np_imgs = image.cpu().numpy()
            outs = []
            use_cache = False
            for arr in np_imgs:
                pil_img = Image.fromarray((arr * 255).astype(np.uint8))
                out_img = self.draw_text(
                    pil_img, text, all_caps,
                    font_size, letter_spacing, font,
                    fill_color_hex, fill_alpha,
                    stroke_enable,
                    stroke_color_hex, stroke_alpha, stroke_thickness,
                    padding, horizontal_alignment, vertical_alignment,
                    x_shift, y_shift, line_spacing,
                    bg_enable, bg_color_hex, bg_alpha, bg_padding, bg_radius,
                    shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance,
                    use_cache=use_cache,  # reuse wrapped lines/metrics for next frames
                )
                outs.append(np.array(out_img).astype(np.float32) / 255.0)
                use_cache = True
            return (torch.tensor(np.stack(outs)),)

NODE_CLASS_MAPPINGS = {"Advanced Text Overlay": TextOverlay}
