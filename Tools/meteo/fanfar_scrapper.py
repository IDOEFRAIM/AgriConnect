#!/usr/bin/env python3
# Agents/Meteo/fanfar_floods_burkina.py
"""
Pipeline autonome :
- Scrape FANFAR PIV (playwright) en bloquant CSS/JS/fonts/analytics
- Sauvegarde uniquement JSON utiles et images
- Filtre GeoJSON pour le Burkina Faso et pour les features liées aux inondations
- Nettoie images trop petites ou quasi-monochromes (noir/vert)
Sortie principale : fanfar_output/floods_burkina_aggregated.json
"""

import asyncio
import json
import re
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import httpx
from shapely.geometry import shape, box
from PIL import Image
import io

from playwright.async_api import async_playwright, Request, Response, TimeoutError as PlaywrightTimeoutError

# ---------- Configuration ----------
OUTPUT_DIR = Path("fanfar_output")
OUTPUT_DIR.mkdir(exist_ok=True)
CAPTURED_PATH = OUTPUT_DIR / "captured_requests.json"
OUT_AGG = OUTPUT_DIR / "floods_burkina_aggregated.json"

TARGET_URL = "https://fanfar.eu/fr/piv/"
MAX_SAVE = 400

# bbox approximatif Burkina Faso [minx, miny, maxx, maxy]
BURKINA_BBOX = (-5.5, 9.4, 2.4, 15.1)

# image cleaning thresholds
MIN_IMAGE_PIXELS = 50 * 50
SAMPLE_PIXELS = 2000
MONOCHROME_THRESHOLD = 0.92

# patterns pour capturer endpoints potentiellement utiles (flood/geojson)
REQUEST_PATTERNS = [
    re.compile(r".*geojson.*", re.IGNORECASE),
    re.compile(r".*features.*", re.IGNORECASE),
    re.compile(r".*stations.*", re.IGNORECASE),
    re.compile(r".*hype.*", re.IGNORECASE),
    re.compile(r".*forecast.*", re.IGNORECASE),
    re.compile(r".*fanfar.*", re.IGNORECASE),
    re.compile(r".*piv.*", re.IGNORECASE),
    re.compile(r".*api.*", re.IGNORECASE),
    re.compile(r".*hydro.*", re.IGNORECASE),
    re.compile(r".*flood.*", re.IGNORECASE),
]

# mots-clés pour filtrer propriétés liées aux inondations
FLOOD_KEYWORDS = ("flood", "inond", "inondation", "débit", "seuil", "alerte", "hype", "subid", "crue")

# ---------- Utilitaires image ----------
def is_mostly_black_or_green(img: Image.Image,
                             sample_pixels: int = SAMPLE_PIXELS,
                             threshold: float = MONOCHROME_THRESHOLD) -> bool:
    img = img.convert("RGB")
    w, h = img.size
    total = w * h
    if total == 0:
        return True
    if total < MIN_IMAGE_PIXELS:
        return True
    to_sample = min(sample_pixels, total)
    black_or_green = 0
    if total <= to_sample:
        pixels = list(img.getdata())
        for r, g, b in pixels:
            if (r < 40 and g < 40 and b < 40):
                black_or_green += 1
            elif (g > 80 and g > 1.4 * r and g > 1.4 * b):
                black_or_green += 1
    else:
        for _ in range(to_sample):
            x = random.randrange(w)
            y = random.randrange(h)
            r, g, b = img.getpixel((x, y))
            if (r < 40 and g < 40 and b < 40):
                black_or_green += 1
            elif (g > 80 and g > 1.4 * r and g > 1.4 * b):
                black_or_green += 1
    proportion = black_or_green / to_sample
    return proportion >= threshold

# ---------- Playwright scraper (bloque CSS/JS/fonts/analytics) ----------
def should_capture_url(url: str) -> bool:
    for pat in REQUEST_PATTERNS:
        if pat.search(url):
            return True
    return False

async def save_response(context, url: str, name: str) -> Optional[str]:
    try:
        resp: Response = await context.request.get(url, timeout=30000)
    except Exception:
        return None
    ct = (resp.headers.get("content-type") or "").lower()
    try:
        if ct.startswith("image/"):
            body = await resp.body()
            ext = ".png"
            if "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            elif "svg" in ct:
                ext = ".svg"
            path = OUTPUT_DIR / f"{name}{ext}"
            path.write_bytes(body)
            return str(path)
        else:
            text = await resp.text()
            stripped = text.lstrip()
            if "application/json" in ct or stripped.startswith("{") or stripped.startswith("["):
                path = OUTPUT_DIR / f"{name}.json"
                path.write_text(text, encoding="utf-8")
                return str(path)
            else:
                # ignore CSS/JS/text that is not JSON
                return None
    except Exception:
        return None

async def run_scraper(target_url: str = TARGET_URL, headless: bool = True, max_save: int = MAX_SAVE) -> List[Dict[str, Any]]:
    captured: List[Dict[str, Any]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        context.set_default_timeout(90000)
        page = await context.new_page()

        # route handler : ABORT CSS/JS/fonts/analytics to avoid saving them
        async def route_handler(route, request):
            url = request.url.lower()
            resource = request.resource_type
            if resource in ("stylesheet", "script", "font"):
                await route.abort()
                return
            if "google-analytics" in url or "doubleclick" in url or "googletagmanager" in url:
                await route.abort()
                return
            await route.continue_()

        await page.route("**/*", route_handler)

        # capture requests matching patterns
        async def on_request(request: Request):
            url = request.url
            if should_capture_url(url):
                entry = {"url": url, "method": request.method, "headers": dict(request.headers), "timestamp": time.time()}
                captured.append(entry)
                # incremental save
                try:
                    CAPTURED_PATH.write_text(json.dumps(captured, indent=2), encoding="utf-8")
                except Exception:
                    pass

        page.on("request", on_request)

        print("Loading FANFAR PIV page...")
        try:
            await page.goto(target_url, timeout=90000, wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            print("Warning: goto timed out, continuing")

        # try to open layer controls and click flood-like labels
        layer_selectors = ["button[title='Layers']", "button[aria-label*='Layers']", ".leaflet-control-layers", ".mapboxgl-ctrl-group", ".ol-control"]
        for sel in layer_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    try:
                        await el.click()
                        await asyncio.sleep(0.6)
                    except Exception:
                        pass
            except Exception:
                pass

        possible_labels = ["Inondation", "Inondations", "Flood", "Hydrologie", "Seuil", "Aléa"]
        for label in possible_labels:
            try:
                el = await page.query_selector(f"text={label}")
                if el:
                    try:
                        await el.click()
                        await asyncio.sleep(0.6)
                    except Exception:
                        pass
            except Exception:
                pass

        # search Burkina Faso to force local data
        try:
            search_selectors = ["input[placeholder*='Rechercher']", "input[placeholder*='Search']", "input[type='search']"]
            for sel in search_selectors:
                el = await page.query_selector(sel)
                if el:
                    try:
                        await el.fill("Burkina Faso")
                        await asyncio.sleep(0.5)
                        await el.press("Enter")
                        await asyncio.sleep(2)
                        break
                    except Exception:
                        pass
        except Exception:
            pass

        await asyncio.sleep(6)
        try:
            await page.wait_for_load_state("networkidle", timeout=45000)
        except PlaywrightTimeoutError:
            pass

        # ensure captured file exists
        try:
            CAPTURED_PATH.write_text(json.dumps(captured, indent=2), encoding="utf-8")
        except Exception:
            pass

        # fetch and save responses for captured urls (only JSON/images)
        saved = 0
        for i, req in enumerate(captured):
            if saved >= max_save:
                break
            url = req["url"]
            name = f"req_{i}"
            saved_path = await save_response(context, url, name)
            if saved_path:
                saved += 1
            await asyncio.sleep(0.12)

        await browser.close()
    print("Scraper finished. Captured requests:", len(captured), "Saved responses:", saved)
    return captured

# ---------- Post-processing : test candidates via httpx, aggregate GeoJSON ----------
def test_and_save_candidates_from_captured(captured_path: Path = CAPTURED_PATH, out_dir: Path = OUTPUT_DIR):
    if not captured_path.exists():
        return
    with captured_path.open(encoding="utf-8") as f:
        captured = json.load(f)
    client = httpx.Client(timeout=30.0)
    saved = 0
    for i, req in enumerate(captured):
        url = req.get("url")
        if not url:
            continue
        low = url.lower()
        if any(k in low for k in ("geojson", "features", "flood", "forecast", "stations", "hype", "api")):
            try:
                r = client.get(url)
                r.raise_for_status()
                ct = (r.headers.get("content-type") or "").lower()
                text = r.text
                if "application/json" in ct or text.lstrip().startswith("{") or text.lstrip().startswith("["):
                    fname = out_dir / f"captured_req_{i}.json"
                    fname.write_text(r.text, encoding="utf-8")
                    saved += 1
                # else ignore non-json responses
            except Exception:
                pass
    client.close()
    print("test_and_save_candidates: saved", saved, "JSON files")

def aggregate_geojson_for_burkina(out_dir: Path = OUTPUT_DIR, out_path: Path = OUT_AGG, bbox: tuple = BURKINA_BBOX):
    b = box(*bbox)
    aggregated = {"type": "FeatureCollection", "features": []}
    candidate_files = []
    for p in sorted(out_dir.glob("captured_req_*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict) and "features" in raw:
            candidate_files.append(p.name)
            for feat in raw.get("features", []):
                geom = feat.get("geometry")
                if not geom:
                    continue
                try:
                    shp = shape(geom)
                except Exception:
                    continue
                if not shp.intersects(b):
                    continue
                # filter by properties containing flood keywords
                props = feat.get("properties", {}) or {}
                props_text = json.dumps(props).lower()
                if any(k in props_text for k in FLOOD_KEYWORDS) or any(k in (feat.get("id","") or "").lower() for k in FLOOD_KEYWORDS):
                    aggregated["features"].append(feat)
    out_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated {len(aggregated['features'])} flood-related features from files: {candidate_files}")

# ---------- Image cleanup ----------
def clean_images(out_dir: Path = OUTPUT_DIR) -> Tuple[List[str], List[str]]:
    removed = []
    kept = []
    for p in sorted(out_dir.glob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".bin"):
            continue
        if p.suffix.lower() == ".svg":
            if p.stat().st_size < 800:
                p.unlink(missing_ok=True)
                removed.append(p.name)
            else:
                kept.append(p.name)
            continue
        try:
            data = p.read_bytes()
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            if w * h < MIN_IMAGE_PIXELS:
                p.unlink(missing_ok=True)
                removed.append(p.name)
                continue
            if is_mostly_black_or_green(img):
                p.unlink(missing_ok=True)
                removed.append(p.name)
            else:
                kept.append(p.name)
        except Exception:
            try:
                p.unlink(missing_ok=True)
                removed.append(p.name)
            except Exception:
                pass
    print(f"Image cleanup done. Removed: {len(removed)}. Kept: {len(kept)}.")
    return removed, kept

# ---------- Main pipeline ----------
async def main_pipeline(headless: bool = True):
    # 1) run scraper
    captured = await run_scraper(headless=headless)
    # 2) ensure captured file exists
    try:
        CAPTURED_PATH.write_text(json.dumps(captured, indent=2), encoding="utf-8")
    except Exception:
        pass
    # 3) test and save candidate JSONs via httpx (in case some endpoints require direct fetch)
    test_and_save_candidates_from_captured()
    # 4) aggregate GeoJSON for Burkina and filter flood-related features
    aggregate_geojson_for_burkina()
    # 5) clean images
    clean_images()
    print("Pipeline complete. Output:", OUT_AGG)

if __name__ == "__main__":
    # exécute visible la première fois pour debug, change headless=True pour production
    asyncio.run(main_pipeline(headless=False))