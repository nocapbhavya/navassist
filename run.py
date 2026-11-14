"""Run NavAssist on an image, a folder of images, a video file, or a webcam.

Examples:
    python run.py --input samples/            # folder of images
    python run.py --input corridor.mp4        # video file
    python run.py --input 0 --speak           # webcam with voice output
"""
import argparse, csv, glob, os, sys
import cv2
from navassist.pipeline import NavAssist
from navassist.visualize import draw

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def result_row(name, r):
    t = r["timings"]
    top = r["decision"]["top"]
    return {
        "image": name, "usable": r["quality"]["usable"], "skip_reason": r["quality"]["reason"] or "",
        "blur_score": round(r["quality"]["blur_score"], 1), "brightness": round(r["quality"]["brightness"], 1),
        "n_obstacles": len(r["hazards"]), "n_text": len(r["texts"]),
        "top_object": top["label"] if top else "", "top_zone": top["zone"] if top else "",
        "top_band": top["band"] if top else "", "level": r["decision"]["level"],
        "message": r["decision"]["message"] or "",
        **{f"t_{k}_ms": round(v * 1000, 1) for k, v in t.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--every", type=int, default=5, help="process every Nth video frame")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nav = NavAssist(use_ocr=not args.no_ocr, speak=args.speak)
    rows = []

    if os.path.isdir(args.input) or args.input.lower().endswith(IMG_EXT):
        files = sorted(f for f in glob.glob(os.path.join(args.input, "*")) if f.lower().endswith(IMG_EXT)) \
            if os.path.isdir(args.input) else [args.input]
        for i, f in enumerate(files, 1):
            frame = cv2.imread(f)
            if frame is None:
                continue
            r = nav.process(frame)
            name = os.path.basename(f)
            cv2.imwrite(os.path.join(args.out, os.path.splitext(name)[0] + "_out.jpg"), draw(r))
            rows.append(result_row(name, r))
            print(f"[{i}/{len(files)}] {name}: {r['decision']['level']:8s} {r['decision']['message'] or ''}")
    else:
        src = int(args.input) if args.input.isdigit() else args.input
        cap = cv2.VideoCapture(src)
        writer, n = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % args.every:
                continue
            r = nav.process(frame)
            vis = draw(r)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(args.out, "output.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 6, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
            rows.append(result_row(f"frame_{n:05d}", r))
            print(f"frame {n}: {r['decision']['level']:8s} {r['decision']['message'] or ''}")
            if isinstance(src, int):
                cv2.imshow("NavAssist", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if writer:
            writer.release()

    if rows:
        with open(os.path.join(args.out, "results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), restval="")
            w.writeheader(); w.writerows(rows)
        print(f"Saved {len(rows)} results to {args.out}/results.csv")


if __name__ == "__main__":
    sys.exit(main())
