#!/usr/bin/env python

import sys
import argparse
import concurrent.futures

import cv2
import numpy as np
def nul_quirk(img):
    return img

def threshold_white(img):
    lo = np.array([252])
    hi = np.array([255])

    mask = cv2.inRange(img, lo, hi)
    img[mask>0] = (0)
    #ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img

def resize(img):
    shape = img.shape
    w, h = int(shape[1]/2), int(shape[0]/2)
    half = cv2.resize(img, (w, h), interpolation = cv2.INTER_CUBIC)
    return cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC)

QUIRKS = {
    'null': nul_quirk,
    'thresh_white': threshold_white,
    'resize': resize
}

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
        img = QUIRKS[q](img)
        if args.debug:
            show(img)

        retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)
        if retval:
           return decoded_info[0]

    raise ValueError(f"Could not decode {filename}, tried {args.quirks}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument('-q', '--quirks', nargs="+", default=QUIRKS)
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()
    class stats: pass
    stats.success = 0
    stats.error = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        future_to_result = {executor.submit(process_img, filename, args): filename for filename in args.filename}
        for future in concurrent.futures.as_completed(future_to_result):
            filename = future_to_result[future]
            try:
                result = future.result()
            except Exception as e:
                print('%r generated an exception: %s' % (filename, e))
                stats.error +=1
            else:
                print(f"{filename},{result}")
                stats.success +=1
    print("all done", stats)
