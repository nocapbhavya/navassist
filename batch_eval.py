"""Process a slice of an image folder and append results (lets long runs be split up)."""
import csv, glob, os, sys, time
import cv2
from navassist.pipeline import NavAssist
from navassist.visualize import draw
from run import result_row

src, out, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "*.jpg")))[start:end]
nav = NavAssist(use_ocr=True)
csv_path = os.path.join(out, "results.csv")
FIELDS = ["image", "usable", "skip_reason", "blur_score", "brightness", "n_obstacles", "n_text",
          "top_object", "top_zone", "top_band", "level", "message", "t_preprocess_ms",
          "t_detection_ms", "t_depth_ms", "t_ocr_ms", "t_reasoning_ms", "t_total_ms"]
new = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
    if new:
        w.writeheader()
    for f in files:
        r = nav.process(cv2.imread(f))
        cv2.imwrite(os.path.join(out, os.path.basename(f)[:-4] + "_out.jpg"), draw(r))
        w.writerow(result_row(os.path.basename(f), r))
        fh.flush()
print("done", start, end)
