"""
authenx/tests/test_api.py

Integration tests for all AuthenX API endpoints.

Run: python tests/test_api.py
(Requires the server to be running at http://localhost:5000)
"""

import io
import json
import sys
import requests
from PIL import Image

BASE_URL = "http://localhost:5000"


def green(text):  return f"\033[92m{text}\033[0m"
def red(text):    return f"\033[91m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"


def assert_response(resp, expected_status=200, required_keys=None):
    if resp.status_code != expected_status:
        print(red(f"  ✗ Expected {expected_status}, got {resp.status_code}"))
        print(red(f"    Body: {resp.text[:300]}"))
        return False

    data = resp.json()
    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(red(f"  ✗ Missing keys in response: {missing}"))
            return False

    print(green(f"  ✓ Status {resp.status_code}"))
    return True


# ─── Test: Health ──────────────────────────────────────────────────────────────

def test_health():
    print("\n[1] GET /health")
    resp = requests.get(f"{BASE_URL}/health")
    ok = assert_response(resp, 200, ["status"])
    if ok:
        print(f"     {resp.json()}")
    return ok


# ─── Test: Image Detection ─────────────────────────────────────────────────────

def _create_test_image_bytes() -> bytes:
    """Generate a small synthetic JPEG image for testing."""
    img = Image.new("RGB", (224, 224), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_image_detection():
    print("\n[2] POST /detect/image (synthetic JPEG)")
    image_bytes = _create_test_image_bytes()

    resp = requests.post(
        f"{BASE_URL}/detect/image",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    ok = assert_response(resp, 200, ["prediction", "confidence_score", "filename", "raw_probs"])
    if ok:
        data = resp.json()
        print(f"     prediction:       {data['prediction']}")
        print(f"     confidence_score: {data['confidence_score']}%")
        print(f"     raw_probs:        {data['raw_probs']}")
    return ok


def test_image_invalid_type():
    print("\n[3] POST /detect/image (invalid file type → expect 422)")
    fake_pdf = b"%PDF-1.4 fake content"
    resp = requests.post(
        f"{BASE_URL}/detect/image",
        files={"file": ("document.pdf", fake_pdf, "application/pdf")},
    )
    ok = assert_response(resp, 422)
    if ok:
        print(f"     error: {resp.json().get('error', '')}")
    return ok


# ─── Test: Headline Verification ───────────────────────────────────────────────

def test_headline_real():
    print("\n[4] POST /verify/headline (plausibly real headline)")
    resp = requests.post(
        f"{BASE_URL}/verify/headline",
        json={"headline": "Scientists discover new exoplanet in habitable zone"},
        headers={"Content-Type": "application/json"},
    )
    ok = assert_response(resp, 200, ["prediction", "confidence_score", "sources_checked"])
    if ok:
        data = resp.json()
        print(f"     prediction:       {data['prediction']}")
        print(f"     confidence_score: {data['confidence_score']}%")
        print(f"     sources checked:  {len(data['sources_checked'])}")
        if data["sources_checked"]:
            print(f"     top source:       {data['sources_checked'][0].get('title', '')[:60]}")
    return ok


def test_headline_fake():
    print("\n[5] POST /verify/headline (implausible headline)")
    resp = requests.post(
        f"{BASE_URL}/verify/headline",
        json={"headline": "Aliens have officially landed and are running the Federal Reserve"},
        headers={"Content-Type": "application/json"},
    )
    ok = assert_response(resp, 200, ["prediction", "confidence_score"])
    if ok:
        data = resp.json()
        print(f"     prediction:       {data['prediction']}")
        print(f"     confidence_score: {data['confidence_score']}%")
    return ok


def test_headline_empty():
    print("\n[6] POST /verify/headline (empty headline → expect 422)")
    resp = requests.post(
        f"{BASE_URL}/verify/headline",
        json={"headline": ""},
        headers={"Content-Type": "application/json"},
    )
    ok = assert_response(resp, 422)
    if ok:
        print(f"     error: {resp.json().get('error', '')}")
    return ok


def test_missing_file():
    print("\n[7] POST /detect/image (no file field → expect 400)")
    resp = requests.post(f"{BASE_URL}/detect/image", data={})
    ok = assert_response(resp, 400)
    if ok:
        print(f"     error: {resp.json().get('error', '')}")
    return ok


# ─── Runner ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AuthenX API Integration Tests")
    print("=" * 60)

    tests = [
        test_health,
        test_image_detection,
        test_image_invalid_type,
        test_headline_real,
        test_headline_fake,
        test_headline_empty,
        test_missing_file,
    ]

    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception as e:
            print(red(f"  ✗ Exception: {e}"))
            results.append(False)

    passed = sum(results)
    total  = len(results)

    print("\n" + "=" * 60)
    if passed == total:
        print(green(f"  All {total} tests passed ✓"))
    else:
        print(yellow(f"  {passed}/{total} tests passed"))
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
