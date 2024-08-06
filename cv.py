#!/usr/bin/env python

import sys
import argparse
import concurrent.futures
import logging

import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)

def nul_quirk(img):
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def threshold_adaptive(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def threshold_binary(img):
    ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    return img

def threshold_white(img):
    if len(img.shape) > 2 and img.shape[2] > 1:
       img = to_gray(img)

    lo = np.array([254])
    hi = np.array([255])

    mask = cv2.inRange(img, lo, hi)
    img[mask>0] = (0)

    #img =
    return threshold_binary(img)

def resize(img, rate=2):
    shape = img.shape
    half = cv2.resize(img, None, fx=1/rate, fy=1/rate, interpolation = cv2.INTER_CUBIC)
    return cv2.resize(half, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)

def quirk_crop(img):
    h = img.shape[0]
    w = img.shape[1]
    c = img[h-w:h,0:w]
    return c

QUIRKS = {
    'none': nul_quirk,
    'thresh_adaptive': threshold_adaptive,
    'thresh_binary': threshold_binary,
    'gray': to_gray,
    'crop': quirk_crop,
    'thresh_white': threshold_white,
    'resize': resize,
}

DEFAULT_QUIRKS = ['none', 'gray', 'thresh_white', 'resize', 'thresh_binary', 'crop']

def show(img):
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        pts = np.array(points[0], np.int32)
        pts = pts.reshape((-1,1,2))
        img = cv2.polylines(img, [pts], True, (0,0,255), 10)
    except:
        pass

    cv2.imshow('show', img)
    cv2.waitKey(0)

def process_img(filename, args):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    qcd = cv2.QRCodeDetector()
    for q in args.quirks:
        LOG.info(f"{filename}: trying QUIRK {q}", file=sys.stderr)
        img = QUIRKS[q](img)
        if args.debug:
            show(img)

        retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)
        if retval:
           return decoded_info[0]

    raise ValueError(f"Could not decode {filename}, tried {args.quirks}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument('-v', '--verbose', action="count", default=0)
    parser.add_argument('-q', '--quirks', nargs="+", default=DEFAULT_QUIRKS)
    parser.add_argument('-D', '--decoders', nargs="+", default=DECODERS)
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()
    class stats:
        success = 0
        error = 0


    with tqdm(total=len(args.filename)) as bar:
        with logging_redirect_tqdm():
            with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
                future_to_result = {executor.submit(process_img, filename, args): filename for filename in args.filename}
                for future in concurrent.futures.as_completed(future_to_result):
                    filename = future_to_result[future]
                    bar.update(1)
                    try:
                        result = future.result()
                    except Exception as e:
                        LOG.info('%r generated an exception: %s' % (filename, e))
                        stats.error +=1
                    else:
                        tqdm.write(f"{filename},{result}")
                        stats.success +=1
                    bar.set_description(f"OK: {stats.success}, E: {stats.error}")
    print("all done", stats)
