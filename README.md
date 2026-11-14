# NavAssist – Intelligent Navigation Assistance for Blind and Visually Impaired People

Major Project-I (18B19CI791), Group RG 39, JUIT – Dr Nancy Singla (Supervisor)

A camera-only pipeline that detects obstacles, estimates how near they are, reads signs, and gives
one short prioritised voice instruction (e.g. "Stop. Chair ahead. Move left.").

```
camera frame -> preprocessing (resize, blur/dark check)
             -> YOLOv8n (objects) + MiDaS small (relative depth) + PP-OCRv4 (text)
             -> reasoning (zones, near/mid/far, HIGH/MEDIUM/LOW, free side, hand filter)
             -> text-to-speech with repeat suppression
```

## Setup (Python 3.10 – 3.13, 64-bit)
```bash
pip install -r requirements.txt
```
All five ONNX models are already in `models/`. If any are missing, run `python download_models.py`.

## Run
```bash
python run.py --input samples/                 # folder of images -> outputs/ (+ results.csv)
python run.py --input corridor.mp4 --every 5   # video file -> outputs/output.mp4
python run.py --input 0 --speak                # live webcam with voice (press q to quit)
python run.py --input samples/ --no-ocr        # faster, without text reading
```

## Evaluate and test
```bash
python batch_eval.py data/vizwiz results/vizwiz 0 213   # evaluation on VizWiz images
python make_figures.py                                    # charts in figures/
python -m pytest -v tests                                 # 18 unit tests
```

## Project layout
| File | Purpose |
|---|---|
| `navassist/config.py` | all thresholds in one place |
| `navassist/preprocess.py` | resizing, blur (variance of Laplacian) and darkness checks |
| `navassist/detector.py` | YOLOv8n ONNX inference, letterboxing, NMS, obstacle classes |
| `navassist/depth.py` | MiDaS inference, per-frame normalised closeness, near/mid/far bands |
| `navassist/ocr.py` | PP-OCRv4 scene text reading (DB detection, angle classifier, SVTR recognition, CTC decoding) run directly with ONNX Runtime |
| `navassist/reasoning.py` | zones, priority rules, free direction, hand filter, message generation |
| `navassist/speech.py` | pyttsx3 output with 3 s repeat suppression |
| `navassist/pipeline.py` | ties the modules together and times each stage |
| `navassist/visualize.py` | annotated output frames |

## Known limitations
- MiDaS gives relative depth, so distance is reported as near / mid / far, not metres.
- COCO has no classes for doors, stairs, walls or poles.
- Hand-held close-up photos can still trigger false "person" alerts despite the hand filter.
