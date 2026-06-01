# 🖋️ ComfyUI Advanced Text Overlay

**Repository:** [scofano/ComfyUI-Advanced-TextOverlay](https://github.com/scofano/ComfyUI-Advanced-TextOverlay)

**Forked from:** [munkyfoot/ComfyUI-TextOverlay](https://github.com/munkyfoot/ComfyUI-TextOverlay)

**Originally inspired by:** [mikkel/ComfyUI-text-overlay](https://github.com/mikkel/ComfyUI-text-overlay)

This module provides the **most feature‑rich and precise text overlay system available for ComfyUI**, including:

* Completely re‑engineered layout engine
* Pixel‑perfect multiline stroke alignment
* Opacity‑aware backgrounds and shadows
* Inline rich text with HTML-like tags for bold, italic, per-segment text color, and per-segment highlight backgrounds
* Letter/line spacing, padding, and alignment controls
* Full animation engine (fade + directional movement)
* Batch-aware rendering with smart caching
* Full video processing node with audio‑preserving re‑mux
* Font dropdown with system font discovery
  - Automatically detects installed fonts on Windows, macOS, and Linux
  - Cross-platform font scanning
  - Fallback to common font names for compatibility
  - Reusable font system for future nodes
* **NEW: Instagram Question Box nodes** — render a two-panel sticker overlay (dark header + light body) matching the Instagram "Ask me a question" story format, with full typography, color, shadow, and animation controls


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
* Inline rich text styling with HTML-like tags

### ✔️ 4. Inline Rich Text

The `text` field now supports a small HTML-like subset for styling parts of the text independently.

Supported tags:

* `<b>bold</b>`
* `<i>italic</i>`
* `<br>` for line breaks
* `<span color="#FF0000">red text</span>`
* `<span bg="#000000">highlighted text</span>`
* `<span align="center">centered line block</span>`
* `<div align="right">right-aligned line block</div>`
* `<span style="color:#00FF00; background-color:#222222">combined styling</span>`

Supported attributes on `<span>`:

| Attribute | Purpose |
| --------- | ------- |
| `color`   | Per-segment text color |
| `fill`    | Alias for text color |
| `fg`      | Alias for text color |
| `bg`      | Per-segment highlight/background color |
| `background` | Alias for background color |
| `background-color` | Alias for background color |
| `align`   | Overrides line alignment for that styled portion (`left`, `center`, `right`) |
| `style`   | Inline CSS-like support for `color` and `background-color` |

Examples:

```html
Hello <b>bold</b> and <i>italic</i><br>
<span color="#ff0000">red</span> and <span color="#00aaff" bg="#111111">blue on dark background</span>
```

Notes:

* This is **not full HTML/CSS rendering** — it is a safe, limited inline styling system.
* Plain text still works exactly as before.
* Bold and italic use best-effort font variant matching. If the selected font has no matching variant, the regular font is used.
* Inline background highlights use the node's `bg_alpha` / `bg_radius` settings for opacity and rounding.
* Inline alignment overrides the node's `font_alignment` for the wrapped portion only.
* Invalid inline colors fall back to the node's normal `fill_color_hex` or `bg_color_hex` values.
* Unsupported HTML tags are ignored for styling; their text content still renders.

### ✔️ 5. Rich Text Examples

#### Mixed emphasis

```html
This is <b>bold</b>, <i>italic</i>, and <b><i>both</i></b>.
```

#### Multi-color text

```html
<span color="#ff5555">Red</span>
<span color="#55ff55">Green</span>
<span color="#5599ff">Blue</span>
```

#### Highlighted words only

```html
Normal text with <span bg="#6b1d1d">highlighted words</span> in the middle.
```

#### Combined color + highlight + emphasis

```html
<span color="#ffffff" bg="#004488"><b>Important label</b></span>
```

#### Using inline style

```html
<span style="color:#ffd700; background-color:#202020">Styled with inline attributes</span>
```

#### Multiline rich text

```html
Title: <b>Episode 01</b><br>
<span color="#cccccc">Subtitle line</span><br>
<span color="#ffffff" bg="#7a0018">Now Playing</span>
```

#### Alignment override inside a left-aligned node

```html
Intro line<br>
<span align="center"><b>Centered title</b></span><br>
Ending line
```

#### Right-aligned callout block

```html
<div align="right"><span bg="#222222" color="#ffffff">Right callout</span></div>
```

### ✔️ 6. Animation System

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

### ✔️ 7. Batch-Smart Processing

* Automatically caches layout on first frame
* Maintains perfect consistency across all frames
* For animated batches: animates through first *N* frames, then holds the final pose

### ✔️ 8. Full Video Support

The **Advanced Text Overlay – Video** node:

* Reads a video frame‑by‑frame
* Applies the same overlay logic
* Writes a new MP4
* **Automatically re‑injects the original audio track with ffmpeg**
* Optional `delete_original` flag

### ✔️ 9. Instagram Question Box

Two dedicated nodes — **`Instagram Question Box`** (image) and **`Instagram Question Box – Video`** — render a two-panel sticker that replicates the Instagram "Ask me a question" story format.

**Layout**

```
┌──────────────────────────────────┐
│  Ask me a question  ← header     │  ← header_height px, header_color_hex
├──────────────────────────────────┤
│                                  │
│   Your answer text here          │  ← body_height px, body_color_hex
│                                  │
└──────────────────────────────────┘
```

* All four outer corners are fully rounded (`corner_radius`) with **anti-aliased edges** (4× supersampled rendering)
* The header/body divider is flat — only the outer corners are rounded
* Optional soft drop-shadow (`box_shadow_enable`, `box_shadow_blur`, `box_shadow_distance`)

**Typography**

* Separate font, size, letter spacing, and fill color/alpha for **question** (header) and **answer** (body)
* `answer_line_spacing` controls vertical gap between wrapped lines
* `answer_padding` sets inner padding on all sides of the body text area
* Answer text supports the same **HTML-like rich text** as the main overlay nodes (`<b>`, `<i>`, `<br>`, `<span color="…">`, `\n`) — bold/italic load the correct font variant automatically

**Positioning & Animation**

* `box_width`, `header_height`, `body_height` — independent pixel dimensions
* `horizontal_alignment` / `vertical_alignment` + `x_shift` / `y_shift` for placement
* Same full animation system as the main nodes (`fade_in`, `move_from_*`, easing, `pause_frames_before_start`)

**Video variant extras**

* `pause_seconds_before_start` (converted to frames using source FPS)
* Audio preserved via ffmpeg re-mux
* Optional `delete_original`

**Default values** are tuned for a 1080-wide portrait story:

| Parameter | Default |
| --------- | ------- |
| `box_width` | 950 |
| `header_height` | 90 |
| `body_height` | 200 |
| `corner_radius` | 32 |
| `question_font` | Arial Black |
| `question_font_size` | 30 |
| `answer_font_size` | 50 |
| `box_shadow_blur` | 15 |

---

## 📥 Installation

Clone the repository inside your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/scofano/ComfyUI-Advanced-TextOverlay
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Restart ComfyUI — the nodes will appear under **Advanced Text Overlay**.

## 🎨 Font Selection & Name Resolution

The node includes a robust font management system that automatically populates with fonts installed on your system:

- **Windows**: Scans `C:\Windows\Fonts\`
- **macOS**: Scans `/System/Library/Fonts/`, `/Library/Fonts/`, and `~/Library/Fonts/`
- **Linux**: Scans `/usr/share/fonts/`, `/usr/local/share/fonts/`, `~/.fonts/`, and `~/.local/share/fonts/`

### True Font Name Resolution

Unlike many other nodes that use the font's filename (e.g., `califb.ttf` → "califb"), this module uses the **True Font Family Name** (e.g., "Californian FB"). 

- **Why this matters**: Rendering engines often fail to find fonts by filename, causing them to fall back to **Arial**. By resolving the internal metadata name, this module ensures your selected font is rendered correctly every time.
- **Performance Caching**: Scanning hundreds of system fonts can be slow. This module automatically creates a `font_cache.json` file. The first scan takes a few seconds, but subsequent loads are near-instant (<1ms).

### Using the Font Dropdown

1. Select your desired font from the dropdown menu.
2. The node will automatically resolve the "True Name" to the correct file path.
3. Text will be rendered using the selected font, avoiding the "Arial fallback" bug.
4. Works with both the image and video overlay nodes.

### For Developers

The font system is modular and can be reused in other ComfyUI nodes:

```python
from .font_utils import get_available_fonts, get_font_path

class MyTextNode:
    @classmethod
    def INPUT_TYPES(cls):
        fonts = get_available_fonts()
        return {
            "required": {
                "font": (fonts, {"default": fonts[0] if fonts else "Arial"}),
                # ... other inputs
            }
        }
```

This provides a consistent font selection experience across all your nodes.


---

## 🚀 Usage

### **Image Node: `Advanced Text Overlay`**

Connect an image or batch → configure text parameters → render.

You can enter either plain text:

```text
Hello world
Second line
```

or rich text:

```html
Hello <b>world</b><br><span color="#00d0ff">Second line</span>
```

You can also override line alignment for just part of the content:

```html
<span align="center">Centered line</span>
```

### **Video Node: `Advanced Text Overlay – Video`**

Provide a path to a video → configure overlay → output is written to ComfyUI's output directory.

---

## 🔧 Input Parameters

### Text & Typography

| Parameter        | Description                               |
| ---------------- | ----------------------------------------- |
| `text`           | Text to draw; supports multiline `\n` and inline HTML-like tags such as `<b>`, `<i>`, `<br>`, and `<span color/bg>` |
| `all_caps`       | Force uppercase                           |
| `font`           | Font name or file; auto-searches `/fonts` |
| `font_size`      | Pixel size                                |
| `letter_spacing` | Per-character spacing                     |
| `line_spacing`   | Spacing between lines                     |

### Rich Text Reference

| Syntax | Result |
| ------ | ------ |
| `<b>Text</b>` | Bold text when a matching font variant is available |
| `<i>Text</i>` | Italic text when a matching font variant is available |
| `<br>` | Line break |
| `<span color="#RRGGBB">Text</span>` | Changes text color for that segment |
| `<span bg="#RRGGBB">Text</span>` | Draws a background highlight behind that segment |
| `<span align="center">Text</span>` | Centers the wrapped portion regardless of the node's `font_alignment` |
| `<div align="right">Text</div>` | Right-aligns the wrapped portion regardless of the node's `font_alignment` |
| `<span style="color:#fff; background-color:#000">Text</span>` | Combined inline styling |

### Rich Text Examples

```html
Normal <b>bold</b> <i>italic</i>
```

```html
<span color="#FFD700">Gold</span> <span color="#00FFFF">Cyan</span>
```

```html
<span bg="#8B0000">Red highlight</span> and <span color="#FFFFFF" bg="#004488">white on blue</span>
```

```html
<span color="#FFD700"><b>Golden title</b></span><br>
<span style="color:#ffffff; background-color:#333333">subtitle</span>
```

```html
Top line<br>
<span align="center" color="#ffffff" bg="#444444"><b>Centered mid title</b></span><br>
Bottom line
```

### Rich Text Limitations

* This is **not** a full browser engine and does not support general HTML layout or arbitrary CSS.
* Supported styling is intentionally limited to inline emphasis, line breaks, text color, and segment background color.
* Per-segment backgrounds wrap naturally with the text layout; if text wraps, the highlight is drawn for each wrapped fragment.
* Alignment overrides operate at the wrapped line/block level. If alignment changes mid-text, the renderer starts a new aligned line block for that portion.
* Whole-block background settings (`bg_enable`, `bg_padding`, `bg_color_hex`, etc.) still apply independently from inline segment highlights.

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
