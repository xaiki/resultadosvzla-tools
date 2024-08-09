#!/usr/bin/env python

import sys
import argparse
import concurrent.futures
import logging
import os
import traceback

import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)
COLUMNS = ["Archivo","Acta","Nulos","Vacios","Maduro","Martinez","Bertucci","Brito","Ecarri","Fermin","Ceballos","Gonzalez","Marquez","Rausseo"]
CANDIDATES = {
    "Maduro": ['PSUV', 'PCV', 'TUPAMARO', 'PPT', 'MSV', 'PODEMOS', 'MEP', 'APC', 'ORA', 'UPV', 'EV', 'PVV', 'PFV'],
    "Martinez": ['AD', 'COPEI', 'MR', 'BR', 'DDP', 'UNE'],
    "Bertucci": ['EL CAMBIO'],
    "Brito": ['PV', 'VU', 'UVV', 'MPJ'],
    "Ecarri": ['AP', 'MOVEV', 'CMC', 'FV', 'ALIANZA DEL LAPIZ', 'MIN UNIDAD'],
    "Fermin": ['SPV'],
    "Ceballos": ['VPA', 'AREPA'],
    "Gonzalez": ['UNTC', 'MPV', 'MUD'],
    "Marquez": ['CENTRADOS'],
    "Rausseo": ['CONDE']
}
PARTIES = [p for c in CANDIDATES for p in CANDIDATES[c]]

def load_csv(fn, columns=COLUMNS):
    try:
        df = pd.read_csv(fn)
    except Exception as e:
        df = pd.DataFrame(columns=columns)
    return df


qcd = cv2.QRCodeDetector()
def decode_cv2(img):
    retval, result, points, straight_qrcode = qcd.detectAndDecodeMulti(img)
    return result[0]

def decode_zbar(img):
    barcodes = decode(img, symbols=[ZBarSymbol.QRCODE])
    LOG.warning(barcodes)
    return  barcodes[0].data.decode('utf-8')

DECODERS = {
    'zbar': decode_zbar,
    'cv2': decode_cv2,
}

def nul_quirk(img):
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def threshold_adaptive(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def threshold_binary(img, args=[55, 255]):
    dist, max = 55, 255
    try:
       dist = args[0]
       max = args[1]
    except:
       pass

    ret, img = cv2.threshold(img, 255 - int(dist), 255, cv2.THRESH_BINARY)
    return img

def threshold_white(img, dist=1):
    if len(img.shape) > 2 and img.shape[2] > 1:
       img = to_gray(img)

    lo = np.array([255 - int(dist)])
    hi = np.array([255])

    mask = cv2.inRange(img, lo, hi)
    img[mask>0] = (0)

    return img

def resize(img, rate=2):
    shape = img.shape
    half = cv2.resize(img, None, fx=1/rate, fy=1/rate, interpolation = cv2.INTER_CUBIC)
    return cv2.resize(half, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)

def quirk_crop(img):
    h = img.shape[0]
    w = img.shape[1]
    c = img[h-int(w*1.1):h,0:w]
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
    img = cv2.imread(filename)
    if not isinstance(img, np.ndarray):
        raise FileNotFoundError(f"file not found: {filename}")

    quirks = {}

    for d in args.decoders:
        img_cache = img
        for quirk in args.quirks:
            LOG.debug(f"{filename}: trying DECODER {d}, QUIRK {quirk}")
            q, a = None, None
            try:
                q, a = quirk.split(':')
                try:
                    a = [float(v) for v in a.split(',')]
                except:
                    pass
            except Exception:
                q = quirk

            result = None

            try:
                proc_img = None
                if a != None:
                    try:
                       proc_img = QUIRKS[q](img, float(a))
                    except:
                       proc_img = QUIRKS[q](img, a)
                else:
                    proc_img = QUIRKS[q](img)

                if args.non_destructive == True:
                    LOG.debug("non destructive mode")
                else:
                    img = proc_img


                if args.debug:
                    show(proc_img)

                result = DECODERS[d](img)
                a, r, n, v = result.split('!')
                r = r.split(',')
                votes = {p: int(v) for p, v in zip(PARTIES, r)}
                votes = {c: sum([votes[p] for p in CANDIDATES[c]]) for c in CANDIDATES}
                return([a[:9], int(n), int(v)] + [votes[v] for v in votes])

            except Exception as e:
                LOG.warning(f"{filename}, decoder {d}, quirk {q} failed with {(result, e)}")
                quirks[f"{d}:{q}"] = (result, e)
        img = img_cache

    raise ValueError(f"Could not decode {filename}, tried {quirks}")

def sumi(s):
    return sum([int(i) for i in s])

if __name__ == '__main__':
    LOG_LEVELS = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="*")
    parser.add_argument('-v', '--verbose', action="count", default=0)
    parser.add_argument('-f', '--force', action="store_true")
    parser.add_argument('-q', '--quirks', nargs="+", default=DEFAULT_QUIRKS)
    parser.add_argument('-D', '--decoders', nargs="+", default=DECODERS)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-c', '--csv', default="../decoded.csv")
    parser.add_argument('-F', '--failed-csv', default="../failed.csv")
    parser.add_argument('-P', '--max-procs', default=32, type=int)
    parser.add_argument('-n', '--non-destructive', action='store_true')

    args = parser.parse_args()

    if args.debug:
        concurrent_executor = concurrent.futures.ProcessPoolExecutor
    else:
        concurrent_executor = concurrent.futures.ThreadPoolExecutor

    logging.basicConfig(level=LOG_LEVELS[min(len(LOG_LEVELS) - 1, args.verbose)])
    class stats:
        success = 0
        error = 0

    df = load_csv(args.csv)
    fdf = load_csv(args.failed_csv, columns=['Archivo'])

    to_process = []
    solved = df.to_records()['Archivo']

    if os.path.isdir(args.filename[0]):
        args.filename = [os.path.join(args.filename[0], f) for f in os.listdir(args.filename[0])]

    if args.force:
        to_process = args.filename
    else:
        tqdm.write("trimming solved files")
        for fn in tqdm(args.filename):
            if not fn in solved:
                to_process.append(fn)
    skipped = len(args.filename) - len(to_process)

    with tqdm(total=len(to_process)) as bar:
        with logging_redirect_tqdm():
            with concurrent_executor(max_workers=args.max_procs) as executor:
                future_to_result = {
                    executor.submit(process_img, filename, args) : filename for filename in to_process}
                for future in concurrent.futures.as_completed(future_to_result):
                    filename = future_to_result[future]

                    try:
                        result = future.result()
                    except Exception as e:
                        traceback.print_exception(e)

                        LOG.info('%r generated an exception: %s' % (filename, e))
                        if fdf.loc[fdf['Archivo'] == filename].empty:
                            fdf.loc[len(fdf)] = [filename]
                            fdf.to_csv(args.failed_csv, index = False)
                        stats.error +=1
                    else:
                        row = [filename] + result
                        if df.loc[(df['Acta'] == int(result[0]))].empty:
                            df.loc[len(df)] = row
                        else:
                            df.loc[(df['Acta'] == result[0])] = row

                        df.to_csv(args.csv, index = False)
                        stats.success +=1

                    bar.update(1)
                    bar.set_description(f"OK: {stats.success + skipped}, N: {stats.success}, S: {skipped}, E: {stats.error}")


    print("all done")
    df.to_csv(args.csv, index = False)
