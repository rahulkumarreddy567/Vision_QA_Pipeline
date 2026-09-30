"""
Smoke test — hit /health then POST a dummy image to /predict.
Works against local dev server or the live Render deployment.

Usage:
    python scripts/smoke_predict.py                                          # local
    python scripts/smoke_predict.py --url https://vision-qa-pipeline.onrender.com
"""
import io
import sys
import argparse
import httpx
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="Base URL of the running API")
    parser.add_argument("--image", default=None,
                        help="Path to an image file (uses a grey dummy if omitted)")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"Target: {base}\n")

    # 1. Health check
    try:
        h = httpx.get(f"{base}/health", timeout=30).json()
        print(f"[health]  status={h['status']}  mode={h['mode']}  "
              f"yolo={h['yolo_loaded']}  resnet={h['resnet_loaded']}")
    except Exception as e:
        print(f"[health]  FAILED — {e}")
        sys.exit(1)

    # 2. Predict
    if args.image:
        with open(args.image, "rb") as f:
            img_bytes = f.read()
        fname = args.image
    else:
        img = Image.new("RGB", (224, 224), color=(120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        fname = "dummy.jpg"

    try:
        res = httpx.post(
            f"{base}/predict",
            files={"file": (fname, img_bytes, "image/jpeg")},
            timeout=60,
        )
        res.raise_for_status()
        data = res.json()
        print(f"[predict] status={res.status_code}  defects={data['num_defects']}  "
              f"latency={data['inference_time_ms']} ms  id={data['request_id']}")
        for d in data["detections"]:
            print(f"          bbox={d['bbox']}  type={d['defect_type']}  "
                  f"conf={d['classification_confidence']:.2f}")
        print("\n✅ Smoke test passed")
    except Exception as e:
        print(f"[predict] FAILED — {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
