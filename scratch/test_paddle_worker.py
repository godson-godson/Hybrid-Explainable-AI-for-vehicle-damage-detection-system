import os
import sys
import json
import time

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

def run_ocr(image_path: str):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    res = ocr.ocr(image_path)
    lines = []
    scores = []
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
        for item in res:
            rec_texts = item.get("rec_texts", [])
            rec_scores = item.get("rec_scores", [])
            lines.extend([str(t).strip() for t in rec_texts if str(t).strip()])
            scores.extend([float(s) for s in rec_scores])
    elif isinstance(res, list):
        for page in res:
            if isinstance(page, list):
                for line in page:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        txt_score = line[1]
                        if isinstance(txt_score, (list, tuple)) and len(txt_score) >= 2:
                            lines.append(str(txt_score[0]).strip())
                            scores.append(float(txt_score[1]))
    return {"lines": lines, "scores": scores, "corpus": "\n".join(lines)}

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sample_rc_test.jpg"
    t0 = time.perf_counter()
    out = run_ocr(img_path)
    dt = time.perf_counter() - t0
    out["latency_ms"] = round(dt * 1000, 2)
    print(json.dumps(out))
