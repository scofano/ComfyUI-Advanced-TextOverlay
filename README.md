# 🖋️ ComfyUI Advanced Text Overlay

**Repository:** [scofano/ComfyUI-Advanced-TextOverlay](https://github.com/scofano/ComfyUI-Advanced-TextOverlay)

**Forked from:** [munkyfoot/ComfyUI-TextOverlay](https://github.com/munkyfoot/ComfyUI-TextOverlay)

**Originally inspired by:** [mikkel/ComfyUI-text-overlay](https://github.com/mikkel/ComfyUI-text-overlay)

This node extends and refines the classic ComfyUI Text Overlay concept with a **re-engineered rendering engine**, **precise alignment logic**, **new customization parameters**, and **enhanced batch handling** — making it the most advanced text overlay solution available for ComfyUI.


---
![Advanced Text Overlay Screenshot](https://github.com/scofano/ComfyUI-Advanced-TextOverlay/blob/main/animated.gif)

## ✨ Key Improvements & New Features

### 🧱 1. Rebuilt Rendering Engine

* Fully restructured text rendering logic for higher precision and cleaner anti-aliasing.
* Pixel-accurate bounding box calculations ensure text aligns perfectly to the intended position.
* Consistent stroke behavior across font sizes with improved blending and color stability.

### 🎨 2. Extended Styling and Layout Control

* **Line spacing**, **letter spacing** and **padding** parameters for advanced multi-line composition.
* **Independent X/Y shifting** for fine-tuned manual positioning.
* **Full hex color control** for both text fill and stroke.
* Supports transparent overlays for non-destructive compositing.

### 🧭 3. Smarter Alignment and Positioning

* Expanded support for:

  * Horizontal alignment: `left`, `center`, `right`
  * Vertical alignment: `top`, `middle`, `bottom`
* Reliable anchor-based alignment even on varying resolutions and batch images.

### 🔠 4. Font Management

* Custom fonts can be loaded directly from the repo’s `/fonts` directory.
* Supports both `.ttf` and `.otf` formats.
* Graceful fallback to a system font if a specified font is unavailable.

### 🎞️ 5. New Animation Feature

* The node now supports smooth text animations via the new animations.py module.
* Each animation type manipulates opacity and/or position offsets frame-by-frame.

Supported Animation Types:
![Fade_in](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2E0cjU5bnViOHh5enljOXE1aGJ3cHByanR0aW1hbjUzM2FpcHhnYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/eGUpaQNTuS2qDeTfGB/giphy.gif)
*Fade_in*

Easing Modes:
| Easing          | Behavior                              |
| --------------- | ------------------------------------- |
| **linear**      | Uniform speed.                        |
| **ease_in**     | Starts slow, speeds up.               |
| **ease_out**    | Starts fast, slows down.              |
| **ease_in_out** | Smooth acceleration and deceleration. |

Animation Parameters
| Parameter                    | Type    | Description                                                 |
| ---------------------------- | ------- | ----------------------------------------------------------- |
| **animate**                  | Boolean | Enable or disable animation.                                |
| **animation_kind**           | Enum    | Selects animation type (`fade_in`, `move_from_left`, etc.). |
| **animation_frames**         | Integer | Number of frames to complete the animation.                 |
| **animation_ease**           | Enum    | Easing curve for smooth motion.                             |
| **animation_opacity_target** | Float   | Target opacity multiplier for animation.                    |

Behavior:

* For single images, the node outputs either one static frame or an animated sequence.

* For batches/videos, animation applies over the first animation_frames frames, then holds the final pose.

Technical Summary:

The animation module provides:

* progress(t, T_minus1, ease): easing interpolation over frames.

* compute_opacity(kind, p, target): calculates alpha fade transitions.

* compute_offsets(kind, p, img_w, img_h): determines X/Y displacement for move animations.

These functions are integrated directly into the node’s rendering loop for seamless animated overlays

### ⚙️ 6. Batch and Workflow Enhancements

* Native **batch support** — automatically applies consistent overlay logic across all input images.
* Optimized performance and memory use when processing large image sets.
* Node parameters persist and preview correctly in ComfyUI sessions.

### 🧩 7. Developer & Extensibility Features

* Modularized code for easier maintenance and feature additions.
* Clean separation between text layout and rendering logic.
* Ready for further extensions such as shadow layers, gradient text, or dynamic variable injection.

---

## ⚙️ Installation

To install the **Advanced Text Overlay** node:

1. Locate your ComfyUI `custom_nodes` directory.
   Typically found at:

   ```
   ComfyUI/custom_nodes/
   ```
2. Clone this repository into that directory:

   ```bash
   git clone https://github.com/scofano/ComfyUI-Advanced-TextOverlay
   ```
3. Restart ComfyUI.
   The node will appear under the **`image/text`** category as **“Advanced Text Overlay”**.

---

## 🚀 Usage

1. Add the **Advanced Text Overlay** node to your ComfyUI workflow.
2. Connect your image source to the node’s **image input**.
3. Configure text and style parameters:

   * Text content
   * Font and size
   * Fill and stroke colors
   * Alignment, spacing, and offset adjustments
4. Connect the node’s output to your desired destination (e.g., preview, save image).

---

## 🔧 Input Parameters

| Parameter                | Type          | Description                                                                   |
| ------------------------ | ------------- | ----------------------------------------------------------------------------- |
| **image**                | Image / Batch | Input image(s) to overlay text onto.                                          |
| **text**                 | String        | The text content to render on the image. Supports multi-line text.            |
| **font**                 | String        | Font name or filename (e.g., `arial.ttf`). Looks in `/fonts` or system fonts. |
| **font_size**            | Integer       | Size of the text in pixels.                                                   |
| **fill_color_hex**       | String        | Text fill color in hex format (e.g., `#FFFFFF`).                              |
| **stroke_color_hex**     | String        | Stroke (outline) color in hex format (e.g., `#000000`).                       |
| **stroke_thickness**     | Float         | Thickness of the text outline.                                                |
| **padding**              | Integer       | Additional space around the text block. Useful for multi-line layouts.        |
| **horizontal_alignment** | Enum          | `left`, `center`, or `right` horizontal alignment.                            |
| **vertical_alignment**   | Enum          | `top`, `middle`, or `bottom` vertical alignment.                              |
| **x_shift**              | Integer       | Horizontal offset adjustment (pixels).                                        |
| **y_shift**              | Integer       | Vertical offset adjustment (pixels).                                          |
| **line_spacing**         | Float         | Vertical spacing between lines of text.                                       |
| **letter_spacing**       | Float         | Horizontal spacing between letters.                                           |

---

## 🖼️ Output

The node outputs the modified image (or batch of images) with the specified text overlay applied according to the given configuration.

---

## 🧠 Credits

* Original concept by [mikkel](https://github.com/mikkel/ComfyUI-text-overlay)
* Fork base by [munkyfoot](https://github.com/munkyfoot/ComfyUI-TextOverlay)
* Advanced reimplementation and feature expansion by [scofano](https://github.com/scofano)
