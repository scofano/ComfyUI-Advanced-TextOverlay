import os
import subprocess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageChops

# NEW:
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
                "font": ("STRING", {"default": "ariblk.ttf"}),
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_process"
    CATEGORY = "Advanced Text Overlay"

    # ---------------- helpers ----------------

    def hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.strip().lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(ch * 2 for ch in hex_color)
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def _normalize_text(self, text: str) -> str:
        return text.replace("\\n", "\n").replace("\\N", "\n")

    def _load_font(self, font, font_size):
        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        font_path = os.path.join(fonts_dir, font)
        if not os.path.exists(font_path):
            font_path = font
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font: {e} — using default font")
            return ImageFont.load_default()

    def _wrap_lines(self, draw, text, font, max_width, padding, letter_spacing):
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
                        h_align, v_align, x_shift, y_shift, line_spacing, letter_spacing, font_size, use_cache):
        try:
            current_font_id = (font.getname(), getattr(font, "path", None))
        except Exception:
            current_font_id = (None, None)

        need_recompute = True
        if hasattr(self, "_cached") and self._cached is not None and use_cache:
            (*_, cached_letter_spacing, cached_stroke_width,
             cached_text, cached_font_id, cached_font_size, cached_padding, cached_line_spacing) = self._cached
            if (cached_letter_spacing == letter_spacing and
                cached_stroke_width   == stroke_width   and
                cached_text           == text           and
                cached_font_id        == current_font_id and
                cached_font_size      == font_size      and
                cached_padding        == padding        and
                cached_line_spacing   == line_spacing):
                need_recompute = False

        if need_recompute:
            lines = self._wrap_lines(draw, text, font, img_w, padding, letter_spacing)
            widths, heights, tops = [], [], []
            lefts, rights_sp = [], []
            for ln in lines:
                l, t, r, b = draw.textbbox((0, 0), ln, font=font, stroke_width=stroke_width)
                w = (r - l)
                h = (b - t)
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
            min_top = min(tops) if tops else 0

            self._cached = (
                lines, widths, heights, tops, min_top, min_left,
                block_w, block_h,
                letter_spacing, stroke_width,
                text, current_font_id, font_size, padding, line_spacing
            )

        (lines, widths, heights, tops, min_top, min_left,
         block_w, block_h, _cached_letter_spacing, _cached_stroke_width, *_) = self._cached

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

        if all_caps:
            text = text.upper()

        opacity_scale = max(0.0, min(1.0, float(opacity_scale)))
        fill_alpha = max(0.0, min(1.0, float(fill_alpha) * opacity_scale))
        stroke_alpha = max(0.0, min(1.0, float(stroke_alpha) * opacity_scale))
        x_shift = int(round(x_shift + dx))
        y_shift = int(round(y_shift + dy))

        sw = int(round(font_size * stroke_thickness * 0.5)) if stroke_enable else 0
        if sw % 2 == 1 and sw > 0:
            sw += 1

        (lines, widths, heights, tops, min_top, min_left, block_w, block_h,
         x0, visual_top_y) = self._compute_layout(
            image.width, image.height, draw, text, loaded_font, sw,
            padding, horizontal_alignment, vertical_alignment, x_shift, y_shift,
            line_spacing, letter_spacing, font_size, use_cache
        )

        first_line_baseline_y = visual_top_y - min_top
        x_draw = x0 - min_left

        def _line_offset(i):
            if font_alignment == "left":
                return 0
            elif font_alignment == "center":
                return int(round((block_w - widths[i]) / 2))
            else:
                return int(round(block_w - widths[i]))

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

        # Shadow (animated alpha)
        if shadow_enable and block_w > 0 and block_h > 0:
            sh_r, sh_g, sh_b = self.hex_to_rgb(shadow_color_hex)
            sh_a = int(max(0.0, min(1.0, float(shadow_alpha) * opacity_scale)) * 255)
            sdx = sdy = int(shadow_distance)
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay, "RGBA")

            yy = first_line_baseline_y
            for i, (ln, h) in enumerate(zip(lines, heights)):
                xx = x_draw + _line_offset(i)
                for ch in ln:
                    od.text((xx + sdx, yy + sdy), ch, font=loaded_font,
                            fill=(sh_r, sh_g, sh_b, sh_a))
                    xx += draw.textlength(ch, font=loaded_font) + letter_spacing
                yy += int(round(h + line_spacing))

            image = Image.alpha_composite(image, overlay)

        # Stroke + Fill
        fr, fg, fb = self.hex_to_rgb(fill_color_hex)
        fa = int(max(0.0, min(1.0, fill_alpha)) * 255)
        sr, sg, sb = self.hex_to_rgb(stroke_color_hex)
        sa = int(max(0.0, min(1.0, stroke_alpha)) * 255)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay, "RGBA")

        yy = first_line_baseline_y
        for i, (ln, h) in enumerate(zip(lines, heights)):
            xx = x_draw + _line_offset(i)
            for ch in ln:
                if sw > 0 and sa > 0:
                    od.text((xx, yy), ch, font=loaded_font,
                            fill=(sr, sg, sb, sa),
                            stroke_width=sw, stroke_fill=(sr, sg, sb, sa))
                if fa > 0:
                    od.text((xx, yy), ch, font=loaded_font, fill=(fr, fg, fb, fa))
                xx += draw.textlength(ch, font=loaded_font) + letter_spacing
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
        font_alignment,
        animate=False,
        animation_kind='fade_in',
        animation_frames=24,
        animation_ease='ease_out',
        animation_opacity_target=1.0
    ):
        """
        Single image (H,W,C):
          - animate=False  -> one output image
          - animate=True   -> returns exactly `animation_frames` frames

        Batch (B,H,W,C) video:
          - animate=False  -> draw per-frame with no animation changes
          - animate=True   -> animation starts at frame 0 and completes at frame `animation_frames-1`.
                              From then on, the final pose is held until frame B-1.
        """

        # Single image (H, W, C)
        if len(image.shape) == 3:
            np_img = image.cpu().numpy()
            pil_img = Image.fromarray((np_img * 255).astype(np.uint8))

            if not animate:
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

            # Prime layout cache
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
                p = animations.progress(t_idx, max(1, T - 1), animation_ease)
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

        if not animate:
            out_list = []
            for i in range(B):
                np_img = image[i].cpu().numpy()
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
                    shadow_enable, shadow_color_hex, shadow_alpha, shadow_distance, font_alignment,
                    use_cache=False,
                )
                out_list.append(np.array(out_img).astype(np.float32) / 255.0)
            return (torch.tensor(np.stack(out_list)),)

        # Animated batch: animate on frames [0 .. T-1], then hold on frames [T .. B-1]
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

            eff_t = min(i, T - 1)  # frames beyond T-1 hold the last pose
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
        """
        Mirrors TextOverlay.INPUT_TYPES but replaces `image` with `video_path`
        and adds `filename_prefix`.
        """
        base = TextOverlay.INPUT_TYPES()["required"].copy()
        # Remove image, we don't take an IMAGE tensor here
        base.pop("image")

        required = {
            # video path instead of tensor
            "video_path": ("STRING", {"multiline": False, "default": ""}),
            # filename prefix for the output in ComfyUI's output dir
            "filename_prefix": ("STRING", {"default": "TxtOver"}),
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
        text,
        all_caps,
        font,
        font_size,
        letter_spacing,
        font_alignment,
        fill_color_hex,
        fill_alpha,
        stroke_enable,
        stroke_color_hex,
        stroke_thickness,
        stroke_alpha,
        padding,
        vertical_alignment,
        y_shift,
        horizontal_alignment,
        x_shift,
        line_spacing,
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

                # Animation timing: animate on frames [0 .. T-1], then hold last pose
                if animate:
                    eff_t = min(i, T - 1)
                    p = animations.progress(eff_t, max(1, T - 1), animation_ease)
                    op = animations.compute_opacity(animation_kind, p, float(animation_opacity_target))
                    dx, dy = animations.compute_offsets(animation_kind, p, pil_img.width, pil_img.height)
                else:
                    op = 1.0
                    dx = dy = 0

                use_cache = (i > 0)

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

        # Return STRING so it can be wired or ignored; node still runs even if not connected
        return (out_path,)

NODE_CLASS_MAPPINGS = {
    "Advanced Text Overlay": TextOverlay,
    "Advanced Text Overlay - Video": TextOverlayVideo,
}
