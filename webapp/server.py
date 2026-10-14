"""Drag-and-drop web front end for NV-Reason-CT.

    python webapp/server.py                 # loads the model, serves http://127.0.0.1:7860
    python webapp/server.py --mock          # UI and preprocessing only, canned answers

Uploads (DICOM, zip, NIfTI, JPEG/PNG stacks) are converted to a CT volume in HU,
cropped with the model's own anatomy-aware preprocessing, and answered with streamed
generation. Nothing leaves the machine running this server.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import gzip
import json
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import ConversionError, InputFile, JpegOptions, build_volume, expand_zips  # noqa: E402


STATIC_DIR = Path(__file__).resolve().parent / "static"
REGIONS = ("chest", "abdomen")
MAX_CASES = 6
MAX_UPLOAD_FILES = 20000


# --------------------------------------------------------------------------------------
# Model


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Engine:
    """Loads the processor and model in the background and runs one generation at a time."""

    def __init__(self, model_id, device, mock):
        self.model_id = model_id
        self.device = device
        self.mock = mock
        self.state = "loading"
        self.detail = "Loading processor"
        self.processor = None
        self.model = None
        self.gpu_lock = threading.Lock()
        self.busy = False

    def load(self):
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            if self.mock:
                self.state, self.detail = "ready", "Mock mode: preprocessing is real, answers are canned"
                return
            self.detail = f"Loading model weights onto {self.device} (first run downloads ~10 GB)"
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
                attn_implementation="sdpa",
            )
            self.model = model.eval().to(self.device)
            self.state, self.detail = "ready", f"Model ready on {self.device}"
        except Exception as exc:  # surfaced in the UI instead of killing the server
            self.state, self.detail = "error", f"{type(exc).__name__}: {exc}"
            raise

    def status(self):
        return {
            "state": self.state,
            "detail": self.detail,
            "device": self.device,
            "mock": self.mock,
            "model": self.model_id,
            "busy": self.busy,
        }

    @property
    def image_loader(self):
        if self.processor is None:
            raise HTTPException(503, "The processor is still loading; try again in a moment.")
        return self.processor.image_processor_3d


class StopFlag(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


# --------------------------------------------------------------------------------------
# Uploaded cases


class Case:
    def __init__(self, workdir):
        self.id = uuid.uuid4().hex[:12]
        self.workdir = workdir
        self.files: list[InputFile] = []
        self.volume = None
        self.series = []
        self.series_uid = None
        self.jpeg = JpegOptions()
        self.crops: dict[str, np.ndarray] = {}
        self.crop_paths: dict[str, Path] = {}
        self.lock = threading.Lock()

    def convert(self):
        for path in self.workdir.glob("crop_*.nii"):
            path.unlink()
        self.crops.clear()
        self.crop_paths.clear()
        self.volume, self.series, self.series_uid = build_volume(
            self.files, self.workdir, series_uid=self.series_uid, jpeg=self.jpeg
        )

    def to_json(self):
        return {
            "id": self.id,
            "volume": self.volume.to_json(),
            "series": [s.to_json() for s in self.series],
            "series_uid": self.series_uid,
            "jpeg": self.jpeg.to_json() if self.volume.source == "jpeg" else None,
        }

    def crop(self, engine, region):
        """Model-input crop in HU, (Z, Y, X) at 2 mm, computed by the model's own preprocessing.

        The crop is also saved as an LPS NIfTI; feeding that file back with
        ``anatomy_region=None`` reproduces exactly the same model input, so follow-up
        questions skip the expensive resample and crop.
        """
        with self.lock:
            if region not in self.crops:
                loader = engine.image_loader
                image = loader.load_image(str(self.volume.path), normalize_mode=0, anatomy_region=region)
                vol = image[0].numpy()
                spacing = [float(s) for s in getattr(loader, "pixdim", (2.0, 2.0, 2.0))]
                affine = np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])
                path = self.workdir / f"crop_{region}.nii"
                nib.save(nib.Nifti1Image(vol.transpose(2, 1, 0), affine), str(path))
                self.crops[region] = vol
                self.crop_paths[region] = path
            return self.crops[region], self.crop_paths[region]


class CaseStore:
    def __init__(self, root):
        self.root = Path(root)
        self.cases: OrderedDict[str, Case] = OrderedDict()
        self.lock = threading.Lock()

    def new(self):
        case = Case(Path(tempfile.mkdtemp(prefix="case_", dir=self.root)))
        with self.lock:
            self.cases[case.id] = case
            while len(self.cases) > MAX_CASES:
                _, old = self.cases.popitem(last=False)
                shutil.rmtree(old.workdir, ignore_errors=True)
        return case

    def get(self, case_id):
        with self.lock:
            case = self.cases.get(case_id)
            if case is not None:
                self.cases.move_to_end(case_id)
        if case is None or case.volume is None:
            raise HTTPException(404, "Volume not found or expired; please drop the files again.")
        return case

    def drop(self, case_id):
        with self.lock:
            case = self.cases.pop(case_id, None)
        if case is not None:
            shutil.rmtree(case.workdir, ignore_errors=True)


# --------------------------------------------------------------------------------------
# App


def create_app(engine: Engine, store: CaseStore) -> FastAPI:
    app = FastAPI(title="NV-Reason-CT", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return engine.status()

    @app.post("/api/cases")
    async def create_case(request: Request):
        form = await request.form(max_files=MAX_UPLOAD_FILES, max_fields=50)
        uploads = [v for k, v in form.multi_items() if k == "files" and hasattr(v, "read")]
        if not uploads:
            raise HTTPException(400, "No files received.")
        case = store.new()
        raw = case.workdir / "raw"
        raw.mkdir()
        try:
            for i, upload in enumerate(uploads):
                name = upload.filename or f"file{i}"
                suffix = ".nii.gz" if name.lower().endswith(".nii.gz") else Path(name).suffix.lower()
                target = raw / f"{i:06d}{suffix[:12]}"
                with open(target, "wb") as fh:
                    shutil.copyfileobj(upload.file, fh)
                case.files.append(InputFile(target, name))
            options = json.loads(form.get("options") or "{}")
            case.jpeg = JpegOptions.from_json(options.get("jpeg"))
            case.files = await run_in_threadpool(expand_zips, case.files, case.workdir)
            await run_in_threadpool(case.convert)
        except ConversionError as exc:
            store.drop(case.id)
            raise HTTPException(422, str(exc))
        except Exception:
            store.drop(case.id)
            raise
        finally:
            await form.close()
        return case.to_json()

    @app.post("/api/cases/{case_id}/convert")
    async def reconvert(case_id: str, request: Request):
        case = store.get(case_id)
        body = await request.json()
        if body.get("series_uid"):
            case.series_uid = body["series_uid"]
        if body.get("jpeg") is not None:
            try:
                case.jpeg = JpegOptions.from_json(body["jpeg"])
            except ConversionError as exc:
                raise HTTPException(422, str(exc))
        try:
            await run_in_threadpool(case.convert)
        except ConversionError as exc:
            raise HTTPException(422, str(exc))
        return case.to_json()

    @app.delete("/api/cases/{case_id}")
    def delete_case(case_id: str):
        store.drop(case_id)
        return {"ok": True}

    @app.get("/api/cases/{case_id}/crop/{region}")
    async def crop(case_id: str, region: str):
        if region not in REGIONS:
            raise HTTPException(400, "Region must be chest or abdomen.")
        case = store.get(case_id)
        vol, _ = await run_in_threadpool(case.crop, engine, region)
        payload = np.clip(np.rint(vol), -32768, 32767).astype("<i2").tobytes()
        return Response(
            gzip.compress(payload, compresslevel=1),
            media_type="application/octet-stream",
            headers={
                "Content-Encoding": "gzip",
                "X-Shape": ",".join(str(n) for n in vol.shape),
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/cases/{case_id}/download/{kind}")
    def download(case_id: str, kind: str):
        case = store.get(case_id)
        if kind == "volume":
            path = case.volume.path
        elif kind in REGIONS and kind in case.crop_paths:
            path = case.crop_paths[kind]
        else:
            raise HTTPException(404, "Nothing to download yet.")
        name = f"nvreasonct_{kind}{''.join(path.suffixes)}"
        return FileResponse(path, filename=name, media_type="application/octet-stream")

    @app.post("/api/generate")
    async def generate(request: Request):
        body = await request.json()
        case = store.get(body.get("case_id", ""))
        region = body.get("region", "chest")
        prompt = (body.get("prompt") or "").strip()
        if region not in REGIONS or not prompt:
            raise HTTPException(400, "A prompt and a chest/abdomen region are required.")
        if engine.state != "ready":
            raise HTTPException(503, f"Model not ready: {engine.detail}")
        thinking = bool(body.get("thinking", True))
        max_new_tokens = int(min(max(int(body.get("max_new_tokens", 2048)), 16), 8192))
        history = [
            {"role": h["role"], "content": [{"type": "text", "text": str(h["content"])}]}
            for h in body.get("history", [])
            if h.get("role") in ("user", "assistant") and str(h.get("content", "")).strip()
        ]

        _, crop_path = await run_in_threadpool(case.crop, engine, region)
        stop = threading.Event()

        if engine.mock:
            source = mock_stream(prompt, region, thinking, stop)
        else:
            source = model_stream(engine, crop_path, history, prompt, thinking, max_new_tokens, stop)

        async def events():
            started = time.time()
            produced = 0
            try:
                if engine.gpu_lock.locked():
                    yield sse({"type": "status", "message": "Waiting for the previous request to finish"})
                yield sse({"type": "status", "message": "Reading the volume"})
                iterator = iter(source)
                while True:
                    item = await run_in_threadpool(next, iterator, None)
                    if item is None:
                        break
                    if isinstance(item, dict):
                        yield sse(item)
                        continue
                    produced += 1
                    yield sse({"type": "token", "text": item})
                yield sse({"type": "done", "seconds": round(time.time() - started, 1), "chunks": produced})
            except Exception as exc:
                yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                stop.set()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def model_stream(engine, crop_path, history, prompt, thinking, max_new_tokens, stop):
    """Yield status dicts, then decoded text chunks, from a background generate() call."""
    processor, model = engine.processor, engine.model
    messages = history + [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    messages[0]["content"].insert(0, {"type": "image"})
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    # The cached crop is already resampled and cropped; anatomy_region=None keeps it as-is.
    inputs = processor(text=text, images3d=[str(crop_path)], anatomy_region=None, return_tensors="pt")
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    failure = []

    # The worker owns the GPU lock and releases it only once generate() has really
    # returned, so a cancelled request never overlaps with the next one.
    engine.gpu_lock.acquire()
    if stop.is_set():  # client went away while queued
        engine.gpu_lock.release()
        return
    engine.busy = True

    def run():
        try:
            with torch.inference_mode():
                model.generate(
                    **inputs.to(engine.device),
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([StopFlag(stop)]),
                )
        except Exception as exc:
            failure.append(exc)
            streamer.end()
        finally:
            engine.busy = False
            engine.gpu_lock.release()

    threading.Thread(target=run, daemon=True).start()
    yield {
        "type": "status",
        "message": f"Encoding {int(inputs['input_ids'].shape[1]):,} tokens on {engine.device} "
                   "(the first token can take a while)",
    }
    if thinking:
        yield "<think>\n"
    for chunk in streamer:
        if chunk:
            yield chunk
    if failure:
        raise failure[0]


def mock_stream(prompt, region, thinking, stop):
    yield {"type": "status", "message": "Mock mode: no model weights loaded"}
    parts = []
    if thinking:
        parts.append(
            "<think>\nThis is mock mode, so no model ran. The real model would reason through "
            f"the {region} CT here before answering.\n</think>\n\n"
        )
    parts.append(
        f"**Mock response** (start the server without `--mock` for real inference).\n\n"
        f"Prompt received: _{prompt}_\n\nRegion: {region}. The volume was converted and cropped "
        "exactly as it would be for the model; check the viewer on the left to confirm the crop "
        "covers the anatomy you expect."
    )
    for word in "".join(parts).split(" "):
        if stop.is_set():
            return
        time.sleep(0.02)
        yield word + " "


# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 only on a trusted network")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default="nvidia/NV-Reason-CT", help="Hugging Face model ID or local path")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:1, mps or cpu")
    parser.add_argument("--mock", action="store_true", help="Skip model weights; stream canned answers")
    args = parser.parse_args()

    engine = Engine(args.model, pick_device(args.device), args.mock)
    if engine.device == "cpu" and not args.mock:
        print("WARNING: no GPU found; CPU inference of a 4.7B model with 13,824 image tokens is very slow.")
    upload_root = Path(tempfile.mkdtemp(prefix="nvreasonct_"))
    atexit.register(shutil.rmtree, upload_root, True)
    store = CaseStore(upload_root)

    threading.Thread(target=engine.load, daemon=True).start()
    print(f"NV-Reason-CT web UI: http://{args.host}:{args.port}  (device: {engine.device}{', mock' if args.mock else ''})")
    uvicorn.run(create_app(engine, store), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
