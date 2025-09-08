#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python filmstab.py 01.mov -m template1.png --show --tm-threshold 1.00 
filmstab_perfect.py
Estabilización PERFECTA de película 16mm/8mm.
Genera keyframes HOLD exactos para After Effects con cálculos matemáticamente correctos.
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
from datetime import datetime

# RAW/DNG libraries
try:
    import rawpy
    DNG_SUPPORT = True
    print("DNG support: ENABLED (rawpy found)")
except Exception:
    rawpy = None
    DNG_SUPPORT = False
    print("DNG support: DISABLED (install rawpy: pip install rawpy)")

# Pillow
try:
    from PIL import Image
    PIL_SUPPORT = True
except Exception:
    Image = None
    PIL_SUPPORT = False
    print("PIL support: DISABLED (install Pillow: pip install Pillow)")

# tifffile (preferred for uint16 TIFF)
try:
    import tifffile
    TIFFFILE_SUPPORT = True
except Exception:
    tifffile = None
    TIFFFILE_SUPPORT = False

# Optional DNG writing backends detection (we won't rely on them for the 16-bit TIFF output)
SAVE_DNG_BACKEND = None
try:
    from process_raw import DngFile as ProcessRawDngFile
    SAVE_DNG_BACKEND = "process_raw"
    print("DNG writer: process_raw available (may be used for native DNG writing).")
except Exception:
    ProcessRawDngFile = None
    try:
        import pidng
        SAVE_DNG_BACKEND = "pidng"
        print("DNG writer: pidng available (may be used for native DNG writing).")
    except Exception:
        pidng = None
        print("DNG writer: process_raw/pidng not available — will write 16-bit TIFF for DNG sources.")

# ----------------------------
parser = argparse.ArgumentParser(description="Perfect film stabilization with exact HOLD keyframes and full-width strips (DNG-aware)")
parser.add_argument('Path', help='video file or image sequence or folder')
parser.add_argument('-m','--template', required=True, help='sprocket hole template')
parser.add_argument('-o','--output-path', default='out', help='folder for ROI outputs and AE script')
parser.add_argument('--alt-output-path', default=None, help='folder for full-width strips (default: <output-path>_fullh)')
parser.add_argument('--tm-threshold', type=float, default=0.90)
parser.add_argument('--min-threshold', type=float, default=0.60)
parser.add_argument('--threshold-step', type=float, default=0.05)
parser.add_argument('--show', action='store_true')
parser.add_argument('--pad-mode', choices=['constant','replicate','reflect'], default='replicate')
parser.add_argument('--pad-color', type=int, nargs=3, default=(0,0,0))
parser.add_argument('--roi-offset-x', type=int, default=0)
parser.add_argument('--roi-offset-y', type=int, default=0)
parser.add_argument('--roi-width', type=int, default=None)
parser.add_argument('--roi-height', type=int, default=None)
parser.add_argument('--round', action='store_true', help='Round positions to integers (recommended)')
parser.add_argument('--comp-width', type=int, default=1920, help='After Effects comp width')
parser.add_argument('--comp-height', type=int, default=1080, help='After Effects comp height')
parser.add_argument('--frame-rate', type=int, default=24, help='Frame rate for AE script')
args = parser.parse_args()

if args.alt_output_path is None:
    args.alt_output_path = args.output_path + "_fullh"

# If your older workflow rotated demosaiced DNG 90° CW for display and extraction,
# keep ROTATE_DNG_90_CW = True to preserve behavior (raw array will be rotated the same).
ROTATE_DNG_90_CW = True

# ----------------------------
def load_dng_rgb_for_preview(filepath):
    """Load DNG and return demosaiced BGR (8-bit) for template matching/display."""
    if not DNG_SUPPORT:
        return None, None
    try:
        raw = rawpy.imread(filepath)
        rgb8 = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=8,
            user_flip=0,
            bright=1.0,
            highlight_mode=1,
            gamma=(2.2, 4.5),
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
        )
        bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        if ROTATE_DNG_90_CW:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        return bgr, raw
    except Exception as e:
        print(f"Error reading DNG for preview {filepath}: {e}")
        return None, None

def load_image_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.dng' and DNG_SUPPORT:
        bgr, _ = load_dng_rgb_for_preview(filepath)
        return bgr
    else:
        return cv2.imread(filepath)

# --- Utilities ---
def get_border_type(mode):
    if mode == 'constant': return cv2.BORDER_CONSTANT
    if mode == 'replicate': return cv2.BORDER_REPLICATE
    if mode == 'reflect': return cv2.BORDER_REFLECT
    return cv2.BORDER_CONSTANT

def pad_extract_exact(src, roi_x, roi_y, roi_w, roi_h, pad_mode, pad_color):
    ih, iw = src.shape[:2]
    src_x0 = max(0, roi_x); src_y0 = max(0, roi_y)
    src_x1 = min(iw, roi_x + roi_w); src_y1 = min(ih, roi_y + roi_h)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        available = np.zeros((0,0,3), dtype=np.uint8)
    else:
        available = src[src_y0:src_y1, src_x0:src_x1].copy()
    pad_left = max(0, 0 - roi_x)
    pad_top = max(0, 0 - roi_y)
    pad_right = max(0, (roi_x + roi_w) - iw)
    pad_bottom = max(0, (roi_y + roi_h) - ih)
    padded = False
    if pad_left or pad_top or pad_right or pad_bottom:
        padded = True
        borderType = get_border_type(pad_mode)
        if borderType == cv2.BORDER_CONSTANT:
            bcol = tuple(int(c) for c in pad_color)
            roi = cv2.copyMakeBorder(available, pad_top, pad_bottom, pad_left, pad_right, borderType, value=bcol)
        else:
            roi = cv2.copyMakeBorder(available, pad_top, pad_bottom, pad_left, pad_right, borderType)
        if roi is None or roi.size == 0:
            roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
            if borderType == cv2.BORDER_CONSTANT:
                roi[:,:,:] = bcol
    else:
        roi = available
    if roi.shape[0] != roi_h or roi.shape[1] != roi_w:
        final = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        h = min(roi.shape[0], roi_h)
        w = min(roi.shape[1], roi_w)
        if h > 0 and w > 0:
            final[0:h, 0:w] = roi[0:h, 0:w]
        roi = final
    return roi, padded, (pad_left, pad_top, pad_right, pad_bottom)

def extract_full_width_strip_from_bgr(src_bgr, roi_y, roi_h, pad_mode, pad_color):
    ih, iw = src_bgr.shape[:2]
    roi_y_i = int(round(roi_y))
    src_y0 = max(0, roi_y_i)
    src_y1 = min(ih, roi_y_i + roi_h)
    if src_y0 >= src_y1:
        canvas = np.zeros((roi_h, iw, 3), dtype=np.uint8)
        canvas[:] = tuple(int(c) for c in pad_color)
        return canvas
    available = src_bgr[src_y0:src_y1, 0:iw].copy()
    top_pad = max(0, 0 - roi_y_i)
    bottom_pad = max(0, (roi_y_i + roi_h) - ih)
    if top_pad or bottom_pad:
        borderType = get_border_type(pad_mode)
        if borderType == cv2.BORDER_CONSTANT:
            bcol = tuple(int(c) for c in pad_color)
            filled = cv2.copyMakeBorder(available, top_pad, bottom_pad, 0, 0, borderType, value=bcol)
        else:
            filled = cv2.copyMakeBorder(available, top_pad, bottom_pad, 0, 0, borderType)
        if filled.shape[0] != roi_h or filled.shape[1] != iw:
            final = np.zeros((roi_h, iw, 3), dtype=np.uint8)
            h = min(filled.shape[0], roi_h)
            w = min(filled.shape[1], iw)
            if h > 0 and w > 0:
                final[0:h, 0:w] = filled[0:h, 0:w]
            return final
        return filled
    else:
        if available.shape[0] != roi_h:
            final = np.zeros((roi_h, iw, 3), dtype=np.uint8)
            final[:] = tuple(int(c) for c in pad_color)
            final[0:available.shape[0], 0:iw] = available
            return final
        return available

def save_image_preserve_format(output_path, image_bgr, source_path, fallback_basename="frame"):
    """Save BGR image using the source extension; non-DNG paths use regular saving."""
    if source_path:
        base, ext = os.path.splitext(os.path.basename(source_path))
        ext = ext.lower()
    else:
        base, ext = (fallback_basename, ".png")
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        outname = os.path.join(output_path, f"{base}{ext if ext!='.jpeg' else '.jpg'}")
        ok = cv2.imwrite(outname, image_bgr)
        if not ok:
            raise IOError(f"cv2.imwrite failed for {outname}")
        return outname
    else:
        outname = os.path.join(output_path, f"{base}.png")
        cv2.imwrite(outname, image_bgr)
        print(f"Warning: unknown source extension '{ext}' — wrote PNG to {outname}.")
        return outname

# --- New: HIGH-QUALITY demosaiced 16-bit TIFF strip from DNG (recommended) ---
def save_full_strip_from_dng_as_16bit_tiff(source_dng_path, out_folder, out_basename, roi_y, roi_h, pad_mode, pad_color):
    """
    Reads DNG with rawpy, demosaics to 16-bit RGB (gamma=(1,1) to keep linear), rotates if needed,
    extracts vertical strip [roi_y:roi_y+roi_h] across full width, applies padding if required,
    and writes a 16-bit TIFF (prefer tifffile, fallback to Pillow).
    """
    if not DNG_SUPPORT:
        print("rawpy not available: cannot create 16-bit TIFF from DNG.")
        return None
    try:
        with rawpy.imread(source_dng_path) as raw:
            # Demosaic to 16-bit RGB, gamma=(1,1) to keep linear (best for color grading)
            rgb16 = raw.postprocess(
                use_camera_wb=False,
                use_auto_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
                user_flip=0,
                bright=1.0,
                highlight_mode=1,
                gamma=(1,1),  # linear output (do color grading later)
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
            )  # returns HxWx3 uint16
        # Convert RGB->BGR for OpenCV operations if needed
        bgr16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
        if ROTATE_DNG_90_CW:
            # rotate same as preview (90 CW)
            bgr16 = np.rot90(bgr16, k=3)  # rotate 90° CW
        ih, iw = bgr16.shape[:2]
        roi_y_i = int(round(roi_y))
        src_y0 = max(0, roi_y_i)
        src_y1 = min(ih, roi_y_i + roi_h)
        if src_y0 >= src_y1:
            # empty area -> create filled canvas with pad_color (converted to 16-bit)
            canvas = np.zeros((roi_h, iw, 3), dtype=np.uint16)
            # pad_color is 0..255 -> scale to 0..65535
            pc = [int(c) * 257 for c in pad_color]
            canvas[:] = tuple(pc)
            strip = canvas
        else:
            available = bgr16[src_y0:src_y1, 0:iw].copy()
            top_pad = max(0, 0 - roi_y_i)
            bottom_pad = max(0, (roi_y_i + roi_h) - ih)
            if top_pad or bottom_pad:
                borderType = get_border_type(pad_mode)
                if borderType == cv2.BORDER_CONSTANT:
                    pc = [int(c) * 257 for c in pad_color]
                    # OpenCV expects same dtype; it supports uint16 for copyMakeBorder
                    filled = cv2.copyMakeBorder(available, top_pad, bottom_pad, 0, 0, borderType, value=tuple(pc))
                else:
                    filled = cv2.copyMakeBorder(available, top_pad, bottom_pad, 0, 0, borderType)
                # Ensure exact shape
                if filled.shape[0] != roi_h or filled.shape[1] != iw:
                    final = np.zeros((roi_h, iw, 3), dtype=np.uint16)
                    h = min(filled.shape[0], roi_h)
                    w = min(filled.shape[1], iw)
                    if h > 0 and w > 0:
                        final[0:h, 0:w] = filled[0:h, 0:w]
                    strip = final
                else:
                    strip = filled
            else:
                if available.shape[0] != roi_h:
                    final = np.zeros((roi_h, iw, 3), dtype=np.uint16)
                    final[:] = 0
                    final[0:available.shape[0], 0:iw] = available
                    strip = final
                else:
                    strip = available

        # Save strip as 16-bit TIFF
        outpath = os.path.join(out_folder, f"{out_basename}.tiff")
        try:
            if TIFFFILE_SUPPORT:
                # tifffile expects RGB order; we have BGR16 -> convert to RGB16
                rgb_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
                tifffile.imwrite(outpath, rgb_strip, photometric='rgb')
            elif PIL_SUPPORT:
                # PIL can save multi-channel uint16 TIFF in many cases
                rgb_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_strip)
                pil_img.save(outpath, format='TIFF')
            else:
                # fallback: scale to 8-bit and save PNG (lossy) — should be rare
                scaled = np.clip((strip.astype(np.float32) / 65535.0) * 255.0, 0, 255).astype(np.uint8)
                out_png = os.path.join(out_folder, f"{out_basename}.png")
                cv2.imwrite(out_png, scaled)
                print(f"Warning: neither tifffile nor Pillow available — wrote 8-bit PNG to {out_png}.")
                return out_png
            print(f"Saved high-quality 16-bit TIFF: {outpath}")
            return outpath
        except Exception as e:
            print(f"Error saving 16-bit TIFF for {outpath}: {e}")
            # fallback try pillow conversion if tifffile failed earlier
            if PIL_SUPPORT:
                try:
                    rgb_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_strip)
                    pil_img.save(outpath, format='TIFF')
                    print(f"Saved 16-bit TIFF via Pillow: {outpath}")
                    return outpath
                except Exception:
                    pass
            # final fallback: 8-bit PNG
            try:
                scaled = np.clip((strip.astype(np.float32) / max(1, strip.max())) * 255.0, 0, 255).astype(np.uint8)
                out_png = os.path.join(out_folder, f"{out_basename}.png")
                cv2.imwrite(out_png, scaled)
                print(f"Final fallback: wrote PNG to {out_png}")
                return out_png
            except Exception as e2:
                print(f"Final fallback also failed: {e2}")
                return None
    except Exception as e:
        print(f"Error creating 16-bit strip from {source_dng_path}: {e}")
        return None

# ----------------------------
def open_input(path):
    if os.path.isdir(path):
        exts = ('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.dng')
        files = sorted([os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith(exts)])
        return True, files, None
    if '%' in path:
        cap = cv2.VideoCapture(path, cv2.CAP_IMAGES)
        if not cap.isOpened():
            cap.release(); cap = cv2.VideoCapture(path)
        return False, [], cap
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("No pude abrir la entrada:", path); sys.exit(1)
    return False, [], cap

def match_template_adaptive(image_gray, template_gray, initial_threshold, min_threshold, threshold_step):
    if image_gray.shape[0] < template_gray.shape[0] or image_gray.shape[1] < template_gray.shape[1]:
        return 0.0, -1, -1, initial_threshold
    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    current_threshold = initial_threshold
    while current_threshold >= min_threshold:
        if maxv >= current_threshold:
            return float(maxv), int(maxloc[0]), int(maxloc[1]), current_threshold
        current_threshold -= threshold_step
    return float(maxv), int(maxloc[0]), int(maxloc[1]), current_threshold

# ----------------------------
use_list, image_files, cap = open_input(args.Path)
seed_tpl = cv2.imread(args.template)
if seed_tpl is None:
    print("No pude leer la plantilla"); sys.exit(1)

tpl_h, tpl_w = seed_tpl.shape[:2]
seed_tpl_gray = cv2.cvtColor(seed_tpl, cv2.COLOR_BGR2GRAY)

roi_width = args.roi_width if args.roi_width is not None else tpl_w
roi_height = args.roi_height if args.roi_height is not None else tpl_h

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(args.alt_output_path, exist_ok=True)

csv_path = os.path.join(args.output_path, "extraction_log.csv")
csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
csv_w = csv.writer(csv_f)
csv_w.writerow(["frame", "src_file", "roi_file", "full_strip_file", "hole_x", "hole_y", "roi_x", "roi_y", "score", "padded"])

total_frames = len(image_files) if use_list else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print(f"Total frames: {total_frames}")
print(f"Template: {tpl_w}x{tpl_h}")
print(f"ROI: {roi_width}x{roi_height}, offset: ({args.roi_offset_x}, {args.roi_offset_y})")
print(f"Target composition: {args.comp_width}x{args.comp_height}")
print(f"Supported formats: PNG, JPG, TIFF, BMP" + (", DNG" if DNG_SUPPORT else ""))

frame_idx = 0
saved_count = 0
hole_positions = []
roi_positions = []
reference_hole_pos = None
reference_roi_pos = None
img_width = None
img_height = None
display_width = None
display_height = None
display_scale = 1.0

ae_filename = os.path.join(args.output_path, "perfect_stabilization.jsx")

def calculate_display_size(img_width, img_height, max_width=1000, max_height=1000):
    scale_w = max_width / img_width if img_width > max_width else 1.0
    scale_h = max_height / img_height if img_height > max_height else 1.0
    scale = min(scale_w, scale_h)
    return int(img_width*scale), int(img_height*scale), scale

# ----------------------------
while True:
    if use_list:
        if frame_idx >= total_frames:
            break
        src_path = image_files[frame_idx]
        ext = os.path.splitext(src_path)[1].lower()
        if ext == '.dng' and DNG_SUPPORT:
            src_bgr, _ = load_dng_rgb_for_preview(src_path)
            src = src_bgr
        else:
            src = load_image_file(src_path)
        frame_idx += 1
        cap_frame_number = frame_idx
    else:
        ret, src = cap.read()
        if not ret:
            break
        frame_idx += 1
        src_path = args.Path
        cap_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or frame_idx)

    if src is None:
        print(f"[{cap_frame_number:04d}] Warning: could not load frame {src_path} - skipping")
        continue

    if img_width is None:
        img_height, img_width = src.shape[:2]
        print(f"Image dimensions: {img_width}x{img_height}")
        if args.show:
            display_width, display_height, display_scale = calculate_display_size(img_width, img_height)
            print(f"Preview dimensions: {display_width}x{display_height} (scale: {display_scale:.2f})")

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    tm_score, mx, my, used_threshold = match_template_adaptive(
        gray, seed_tpl_gray, args.tm_threshold, args.min_threshold, args.threshold_step
    )
    print(f"[{cap_frame_number:04d}] Score: {tm_score:.3f} at ({mx},{my}) threshold: {used_threshold:.3f}")

    if mx >= 0 and my >= 0:
        hole_center_x = mx + tpl_w / 2.0
        hole_center_y = my + tpl_h / 2.0

        roi_x = mx + args.roi_offset_x
        roi_y = my + args.roi_offset_y

        roi, padded, pads = pad_extract_exact(src, roi_x, roi_y, roi_width, roi_height, args.pad_mode, args.pad_color)
        out_name = os.path.join(args.output_path, f"frame_{saved_count:05d}.png")
        cv2.imwrite(out_name, roi)

        full_outname = ""
        if src_path.lower().endswith('.dng') and DNG_SUPPORT:
            # Create HIGH QUALITY 16-bit demosaiced TIFF strip from original DNG
            base = os.path.splitext(os.path.basename(src_path))[0] + "_strip"
            saved = save_full_strip_from_dng_as_16bit_tiff(src_path, args.alt_output_path, base, roi_y, roi_height, args.pad_mode, args.pad_color)
            if saved:
                full_outname = saved
            else:
                # fallback: save demosaiced 8-bit strip (from preview)
                full_strip = extract_full_width_strip_from_bgr(src, roi_y, roi_height, args.pad_mode, args.pad_color)
                try:
                    full_outname = save_image_preserve_format(args.alt_output_path, full_strip, src_path, fallback_basename=base)
                except Exception as e:
                    print(f"Error saving full strip fallback: {e}")
                    full_outname = ""
        else:
            full_strip = extract_full_width_strip_from_bgr(src, roi_y, roi_height, args.pad_mode, args.pad_color)
            try:
                full_outname = save_image_preserve_format(args.alt_output_path, full_strip, src_path, fallback_basename=os.path.splitext(os.path.basename(src_path))[0] + "_strip")
            except Exception as e:
                print(f"Error saving full strip: {e}")
                full_outname = ""

        roi_center_x = roi_x + roi_width / 2.0
        roi_center_y = roi_y + roi_height / 2.0

        hole_positions.append((hole_center_x, hole_center_y))
        roi_positions.append((roi_center_x, roi_center_y))

        if reference_hole_pos is None:
            reference_hole_pos = (hole_center_x, hole_center_y)
            reference_roi_pos = (roi_center_x, roi_center_y)
            print(f"REFERENCE set - Hole: {reference_hole_pos}, ROI: {reference_roi_pos}")

        csv_w.writerow([cap_frame_number, src_path, out_name, full_outname, hole_center_x, hole_center_y, roi_center_x, roi_center_y, tm_score, padded])
        saved_count += 1

        if args.show:
            dbg = src.copy()
            cv2.rectangle(dbg, (mx, my), (mx + tpl_w, my + tpl_h), (0, 255, 0), 2)
            cv2.rectangle(dbg, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
            cv2.circle(dbg, (int(hole_center_x), int(hole_center_y)), 3, (0, 255, 255), -1)
            cv2.circle(dbg, (int(roi_center_x), int(roi_center_y)), 5, (0, 0, 255), -1)
            dbg_resized = cv2.resize(dbg, (display_width, display_height))
            cv2.imshow("Debug", dbg_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        if len(hole_positions) > 0:
            last_hole = hole_positions[-1]
            last_roi = roi_positions[-1]
            hole_positions.append(last_hole)
            roi_positions.append(last_roi)
            roi_x = int(last_roi[0] - roi_width / 2.0)
            roi_y = int(last_roi[1] - roi_height / 2.0)

            roi, padded, pads = pad_extract_exact(src, roi_x, roi_y, roi_width, roi_height, args.pad_mode, args.pad_color)
            out_name = os.path.join(args.output_path, f"frame_{saved_count:05d}.png")
            cv2.imwrite(out_name, roi)

            full_strip = extract_full_width_strip_from_bgr(src, roi_y, roi_height, args.pad_mode, args.pad_color)
            try:
                full_outname = save_image_preserve_format(args.alt_output_path, full_strip, src_path, fallback_basename=os.path.splitext(os.path.basename(src_path))[0] + "_strip")
            except Exception as e:
                print(f"Error saving full strip: {e}")
                full_outname = ""

            csv_w.writerow([cap_frame_number, src_path, out_name, full_outname, last_hole[0], last_hole[1], last_roi[0], last_roi[1], 0.0, padded])
            saved_count += 1

            print(f"[{cap_frame_number:04d}] No template - using last position")
            if args.show:
                dbg = src.copy()
                cv2.rectangle(dbg, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)
                cv2.circle(dbg, (int(last_roi[0]), int(last_roi[1])), 5, (0, 0, 255), -1)
                dbg_resized = cv2.resize(dbg, (display_width, display_height))
                cv2.imshow("Debug", dbg_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print(f"[{cap_frame_number:04d}] No template and no reference yet - skipping")

# Close and postprocess
csv_f.close()
if not use_list:
    cap.release()
if args.show:
    cv2.destroyAllWindows()

# ---- AE script generation (fixed safe template) ----
if len(roi_positions) > 0 and reference_roi_pos is not None:
    print(f"\nGenerating After Effects script for {len(roi_positions)} frames...")
    comp_center_x = args.comp_width / 2.0
    comp_center_y = args.comp_height / 2.0
    img_center_x = img_width / 2.0
    img_center_y = img_height / 2.0
    ref_roi_x, ref_roi_y = reference_roi_pos

    with open(ae_filename, 'w', encoding='utf-8') as f:
        f.write("// PERFECT FILM STABILIZATION SCRIPT\n")
        f.write("// Auto-generated - do not edit manually\n\n")
        f.write("var keyframes = {\n")
        for i, (roi_x, roi_y) in enumerate(roi_positions):
            frame_num = i + 1
            movement_x = roi_x - ref_roi_x
            movement_y = roi_y - ref_roi_y
            final_x = comp_center_x - movement_x
            final_y = comp_center_y - movement_y
            if args.round:
                final_x = int(round(final_x))
                final_y = int(round(final_y))
            f.write(f"   {frame_num}: [{final_x:.2f}, {final_y:.2f}],\n")
        f.write("};\n\n")

        js_template = """var frameRate = {frameRate};
var compWidth = {compWidth};
var compHeight = {compHeight};
var sourceWidth = {sourceWidth};
var sourceHeight = {sourceHeight};

var comp = app.project.activeItem;
if (comp && comp instanceof CompItem) {{
    var selectedLayers = comp.selectedLayers;
    if (selectedLayers.length > 0) {{
        app.beginUndoGroup("Perfect Film Stabilization");

        var layer = selectedLayers[0];
        var transform = layer.property("Transform");
        var position = transform.property("Position");
        var anchorPoint = transform.property("Anchor Point");
        var scale = transform.property("Scale");
        var rotation = transform.property("Rotation");

        try {{
            layer.startTime = 0;
            if (layer.parent) layer.parent = null;
            layer.threeDLayer = false;
            layer.motionBlur = false;

            var srcW = (layer.source && layer.source.width) ? layer.source.width : sourceWidth;
            var srcH = (layer.source && layer.source.height) ? layer.source.height : sourceHeight;
            anchorPoint.setValue([srcW/2, srcH/2]);

            scale.setValue([100, 100]);
            rotation.setValue(0);

            while (position.numKeys > 0) {{
                position.removeKey(1);
            }}

            var keyCount = 0;
            for (var frameNum in keyframes) {{
                var time = (parseInt(frameNum) - 1) / frameRate;
                var pos = keyframes[frameNum];
                position.setValueAtTime(time, pos);
                keyCount++;
            }}

            for (var i = 1; i <= position.numKeys; i++) {{
                try {{
                    position.setInterpolationTypeAtKey(i, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD);
                    position.setSpatialTangentsAtKey(i, [0,0], [0,0]);
                }} catch(e) {{}}
            }}

            app.endUndoGroup();
            alert("PERFECT STABILIZATION applied successfully!\\n" +
                  "Keyframes: " + keyCount + " (HOLD interpolation)\\n" +
                  "Composition: " + compWidth + "x" + compHeight + "\\n" +
                  "Source: " + sourceWidth + "x" + sourceHeight + "\\n" +
                  "Reference ROI: {ref_x}, {ref_y}");
        }} catch(error) {{
            app.endUndoGroup();
            alert("Error applying stabilization: " + error.toString());
        }}

    }} else {{
        alert("Error: Please select the layer with your film sequence.");
    }}
}} else {{
    alert("Error: Please open a composition first.");
}}
"""
        js_filled = js_template.format(
            frameRate=args.frame_rate,
            compWidth=args.comp_width,
            compHeight=args.comp_height,
            sourceWidth=img_width,
            sourceHeight=img_height,
            ref_x=f"{ref_roi_x:.1f}",
            ref_y=f"{ref_roi_y:.1f}"
        )
        f.write(js_filled)

    print(f"AE script written to: {ae_filename}")

    # Stats
    movements_x = [abs(roi_positions[i][0] - roi_positions[i-1][0]) for i in range(1, len(roi_positions))]
    movements_y = [abs(roi_positions[i][1] - roi_positions[i-1][1]) for i in range(1, len(roi_positions))]
    if movements_x and movements_y:
        avg_mov_x = sum(movements_x) / len(movements_x)
        avg_mov_y = sum(movements_y) / len(movements_y)
        max_mov_x = max(movements_x)
        max_mov_y = max(movements_y)
        print(f"\n=== STABILIZATION STATISTICS ===")
        print(f"Detected movement in ROI:")
        print(f"  Average: X={avg_mov_x:.2f}px, Y={avg_mov_y:.2f}px")
        print(f"  Maximum: X={max_mov_x:.2f}px, Y={max_mov_y:.2f}px")
        print(f"Keyframes will CANCEL this movement for perfect stabilization.")

# Final status
print(f"\n=== PROCESS COMPLETED ===")
print(f"Extracted frames (ROI): {saved_count}")
print(f"Output folder (ROI & AE script): {args.output_path}")
print(f"Output folder (full-width strips): {args.alt_output_path}")
print(f"CSV log: {csv_path}")
print(f"AE script: {ae_filename}")
print("\nNOTAS:")
print("- Este script ahora genera TIFF de 16 bits (demosaicado, linear) para DNG de entrada cuando es posible.")
print("- Recomendado instalar: rawpy, tifffile, Pillow: pip install rawpy tifffile Pillow")
print(f"- ROTATE_DNG_90_CW = {ROTATE_DNG_90_CW} (ajusta si no quieres rotación).")
