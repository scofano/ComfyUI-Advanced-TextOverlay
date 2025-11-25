# 🖋️ ComfyUI Advanced Text Overlay

**Repository:** [scofano/ComfyUI-Advanced-TextOverlay](https://github.com/scofano/ComfyUI-Advanced-TextOverlay)

**Forked from:** [munkyfoot/ComfyUI-TextOverlay](https://github.com/munkyfoot/ComfyUI-TextOverlay)

**Originally inspired by:** [mikkel/ComfyUI-text-overlay](https://github.com/mikkel/ComfyUI-text-overlay)

This module provides the **most feature‑rich and precise text overlay system available for ComfyUI**, including:

* Completely re‑engineered layout engine
* Pixel‑perfect multiline stroke alignment
* Opacity‑aware backgrounds and shadows
* Letter/line spacing, padding, and alignment controls
* Full animation engine (fade + directional movement)
* Batch-aware rendering with smart caching
* Full video processing node with audio‑preserving re‑mux


## ✨ Key Features

### ✔️ 1. Rebuilt Rendering Engine

* Accurate bounding boxes for multiline text
* Consistent stroke widths (even‑pixel enforcement)
* Crisp, stable text edges with alpha‑aware compositing

### ✔️ 2. Advanced Layout and Typography

* **Letter spacing** and **line spacing**
* **Padding** around the text block
* Horizontal and vertical alignment: `left` / `center` / `right`, `top` / `middle` / `bottom`
* **Per‑axis offset** (`x_shift`, `y_shift`)
* Automatic multiline wrapping based on image width

### ✔️ 3. Styling Options

* Fill + alpha
* Stroke + alpha
* Shadow + alpha + offset
* Rounded background box with padding, color, radius, and alpha

### ✔️ 4. Animation System

Supports:

* `fade_in`
* `fade_out`
* `move_from_top`
* `move_from_bottom`
* `move_from_left`
* `move_from_right`

With easing:

* `linear`
* `ease_in`
* `ease_out`
* `ease_in_out`

Animation Parameters:

* `animation_frames`
* `animation_opacity_target`
* `pause_frames_before_start`

### ✔️ 5. Batch-Smart Processing

* Automatically caches layout on first frame
* Maintains perfect consistency across all frames
* For animated batches: animates through first *N* frames, then holds the final pose

### ✔️ 6. Full Video Support

The **Advanced Text Overlay – Video** node:

* Reads a video frame‑by‑frame
* Applies the same overlay logic
* Writes a new MP4
* **Automatically re‑injects the original audio track with ffmpeg**
* Optional `delete_original` flag

---

## 📥 Installation

Clone the repository inside your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/scofano/ComfyUI-Advanced-TextOverlay
```

Restart ComfyUI — the nodes will appear under **Advanced Text Overlay**.

---

## 🚀 Usage

### **Image Node: `Advanced Text Overlay`**

Connect an image or batch → configure text parameters → render.

### **Video Node: `Advanced Text Overlay – Video`**

Provide a path to a video → configure overlay → output is written to ComfyUI's output directory.

---

## 🔧 Input Parameters

### Text & Typography

| Parameter        | Description                               |
| ---------------- | ----------------------------------------- |
| `text`           | Text to draw; supports multiline `\n`     |
| `all_caps`       | Force uppercase                           |
| `font`           | Font name or file; auto-searches `/fonts` |
| `font_size`      | Pixel size                                |
| `letter_spacing` | Per-character spacing                     |
| `line_spacing`   | Spacing between lines                     |

### Color & Stroke

| Parameter          | Description                              |
| ------------------ | ---------------------------------------- |
| `fill_color_hex`   | Hex fill color                           |
| `fill_alpha`       | Opacity 0–1                              |
| `stroke_enable`    | Toggle outline                           |
| `stroke_color_hex` | Hex outline color                        |
| `stroke_alpha`     | Outline opacity                          |
| `stroke_thickness` | Relative thickness (scaled by font size) |

### Layout & Position

| Parameter              | Description                 |
| ---------------------- | --------------------------- |
| `padding`              | Padding around text block   |
| `horizontal_alignment` | `left` / `center` / `right` |
| `vertical_alignment`   | `top` / `middle` / `bottom` |
| `x_shift`, `y_shift`   | Pixel offsets               |

### Background Box

| Parameter      | Description               |
| -------------- | ------------------------- |
| `bg_enable`    | Toggle background box     |
| `bg_padding`   | Padding around text block |
| `bg_radius`    | Rounded corner radius     |
| `bg_color_hex` | Hex color                 |
| `bg_alpha`     | Opacity                   |

### Shadow

| Parameter          | Description   |
| ------------------ | ------------- |
| `shadow_enable`    | Toggle shadow |
| `shadow_distance`  | Pixel offset  |
| `shadow_color_hex` | Shadow color  |
| `shadow_alpha`     | Opacity       |

### Animation

| Parameter                   | Description                     |
| --------------------------- | ------------------------------- |
| `animate`                   | Enable animation                |
| `animation_kind`            | Fade or movement animation      |
| `animation_frames`          | How many frames animation lasts |
| `animation_opacity_target`  | Target alpha                    |
| `animation_ease`            | Easing curve                    |
| `pause_frames_before_start` | Delay before animation begins   |

### Video-Specific Parameters

| Parameter                    | Description                          |
| ---------------------------- | ------------------------------------ |
| `video_path`                 | Input video                          |
| `filename_prefix`            | Naming prefix for output             |
| `delete_original`            | Remove input after processing        |
| `pause_seconds_before_start` | Pause duration (converted to frames) |

---

## 🧪 Output

* Image node → modified image/batch
* Video node → MP4 written to ComfyUI output directory, with audio preserved

---

## 🧠 Credits

* Original concept by [mikkel](https://github.com/mikkel/ComfyUI-text-overlay)
* Fork base by [munkyfoot](https://github.com/munkyfoot/ComfyUI-TextOverlay)
* Advanced reimplementation and feature expansion by [scofano](https://github.com/scofano)
