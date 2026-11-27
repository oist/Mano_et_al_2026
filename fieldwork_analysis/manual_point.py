#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Click‑to‑zoom point picker — resized display version.
Click to zoom around point, 'c' to reset, 'q' to save.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

# ─────── CONFIG ───────
fieldwork_data_dir = Path.home() / 'bucket/ReiterU/Gio/Polarization/pol-ccard_depth-dist'
if not fieldwork_data_dir.is_dir():
    fieldwork_data_dir = Path('/bucket/ReiterU/Gio/Polarization/pol-ccard_depth-dist')
IMG_DIR    = fieldwork_data_dir / '20250501-depth-cloudy/series2/reverse'
PATTERN    = '*green.png'
WIN_NAME   = 'Image'
WIN_WIDTH  = 1280
WIN_HEIGHT = 960
ZOOM_STEP  = 2.0
MAX_ZOOM   = 16.0

# ─────── ENHANCE CONTRAST ───────
def clahe_bgr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ─────── ZOOM + CROP + RESIZE ───────
def get_display_view(img, cx, cy, zoom):
    H, W = img.shape[:2]
    crop_w = int(W / zoom)
    crop_h = int(H / zoom)

    x1 = max(0, min(W - crop_w, cx - crop_w // 2))
    y1 = max(0, min(H - crop_h, cy - crop_h // 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (WIN_WIDTH, WIN_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return resized, (x1, y1), (crop_w, crop_h)

# ─────── MOUSE CALLBACK ───────
def make_mouse_callback(state):
    def callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = state['offset']
        cw, ch = state['crop_size']
        cx = int(x * cw / WIN_WIDTH) + ox
        cy = int(y * ch / WIN_HEIGHT) + oy
        state['point'] = (cx, cy)
        new_zoom = min(state['zoom'] * ZOOM_STEP, MAX_ZOOM)
        state.update(dict(center=(cx, cy), zoom=new_zoom, needs_redraw=True))
    return callback

# ─────── MAIN LOOP ───────


file_list = sorted(glob.glob(os.path.join(IMG_DIR, PATTERN)))
if not file_list:
    raise RuntimeError('No images found.')

for fpath in file_list:
    print(fpath)
    orig = clahe_bgr(cv2.imread(fpath))
    H, W = orig.shape[:2]

    state = {
        'zoom': 1.0,
        'center': (W // 2, H // 2),
        'offset': (0, 0),
        'crop_size': (W, H),
        'point': None,
        'needs_redraw': True
    }
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_NAME, make_mouse_callback(state))

    while state['needs_redraw']:
       
        view, offset, crop_size = get_display_view(orig, *state['center'], state['zoom'])
        state['offset'] = offset
        state['crop_size'] = crop_size
        cv2.imshow(WIN_NAME, view)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            state.update(dict(zoom=1.0, center=(W//2, H//2), needs_redraw=True))
        elif key == ord('q'):
            state['needs_redraw'] = False
            cv2.destroyAllWindows()
            break


    if state['point'] is not None:
        out_path = os.path.splitext(fpath)[0] + '_point.npy'
        np.save(out_path, np.array(state['point']))
        print(f"✔ {os.path.basename(fpath)} → point saved: {state['point']}")
    else:
        print(f"⚠ {os.path.basename(fpath)} → no point selected.")
