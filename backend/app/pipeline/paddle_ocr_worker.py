"""
PaddleOCR Dedicated Worker Script.
Executed inside .venv_paddle (Python 3.12) to provide deterministic
text line extraction for the vehicle damage assessment document intake pipeline.
Supports both one-shot CLI execution and robust output shape parsing.
"""

import os
import sys
import json
import time
import warnings
from typing import Dict, Any, List, Tuple

# Suppress warnings and Paddle C++ logs to ensure clean stdout
warnings.filterwarnings("ignore")
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"


def _extract_text_lines(raw_res: Any) -> Tuple[List[str], List[float]]:
    """
    Parse text lines and confidence scores agnostic to PaddleOCR version.
    Handles PaddleOCR 3.x dict format (rec_texts/rec_scores) and
    PaddleOCR 2.x nested list-of-tuples format.
    """
    lines: List[str] = []
    scores: List[float] = []

    if not raw_res:
        return lines, scores

    # Format A: PaddleOCR 3.x dict output
    if isinstance(raw_res, list) and len(raw_res) > 0 and isinstance(raw_res[0], dict):
        for item in raw_res:
            rec_texts = item.get("rec_texts", [])
            rec_scores = item.get("rec_scores", [])
            for t, s in zip(rec_texts, rec_scores):
                clean_t = str(t).strip()
                if clean_t:
                    lines.append(clean_t)
                    scores.append(round(float(s), 4))
        return lines, scores

    # Format B: PaddleOCR 2.x nested list output [[ [bbox, (text, score)], ... ]]
    if isinstance(raw_res, list):
        for page in raw_res:
            if isinstance(page, list):
                for element in page:
                    if isinstance(element, (list, tuple)) and len(element) >= 2:
                        txt_score = element[1]
                        if isinstance(txt_score, (list, tuple)) and len(txt_score) >= 2:
                            clean_t = str(txt_score[0]).strip()
                            if clean_t:
                                lines.append(clean_t)
                                scores.append(round(float(txt_score[1]), 4))

    return lines, scores


def process_image(image_path: str) -> Dict[str, Any]:
    """Execute PaddleOCR inference on the document image."""
    t0 = time.perf_counter()

    if not os.path.exists(image_path):
        return {
            "status": "ERROR",
            "error": f"Image file not found at: {image_path}",
            "lines": [],
            "scores": [],
            "corpus": "",
            "char_count": 0,
            "latency_ms": 0.0,
        }

    try:
        from paddleocr import PaddleOCR
        # Initialize lightweight model on CPU without UVDoc unwarping to optimize latency
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="en",
            enable_mkldnn=False,
        )

        res = ocr.ocr(image_path)
        lines, scores = _extract_text_lines(res)
        corpus = "\n".join(lines)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "status": "SUCCESS",
            "lines": lines,
            "scores": scores,
            "corpus": corpus,
            "char_count": len(corpus.strip()),
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "status": "ERROR",
            "error": str(exc),
            "lines": [],
            "scores": [],
            "corpus": "",
            "char_count": 0,
            "latency_ms": latency_ms,
        }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "ERROR",
            "error": "Usage: python paddle_ocr_worker.py <image_path>",
            "lines": [],
            "scores": [],
            "corpus": "",
            "char_count": 0,
            "latency_ms": 0.0,
        }))
        sys.exit(1)

    image_path = sys.argv[1].strip()
    result = process_image(image_path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
