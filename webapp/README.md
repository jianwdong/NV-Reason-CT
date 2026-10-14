# NV-Reason-CT web UI

A drag-and-drop page for running NV-Reason-CT on your own scans. Drop a CT and
the page converts it into the 3D volume the model expects, shows the exact crop
the model will read, and streams back the model's reasoning and answer.

**Research and education use only.** Model output can be wrong and must be
reviewed by a qualified professional. Do not use it for clinical diagnosis or
treatment decisions.

## What you can drop

| Input | How it is handled |
|---|---|
| DICOM folder, files, or `.zip` (any file names, nested folders) | Grouped by series. The largest CT series is picked by default, and you can switch series in the page. Slices are ordered by `ImagePositionPatient`, rescaled to Hounsfield units, and kept in true patient geometry, including oblique and tilted acquisitions. Localizers in the same series are ignored. Compressed transfer syntaxes (JPEG, JPEG-LS, JPEG 2000) are decoded with pylibjpeg. |
| NIfTI (`.nii`, `.nii.gz`) | Used as is (the model's native input). |
| JPEG/PNG stack of axial slices | Sorted by file name (`slice 2` comes before `slice 10`) and turned into an approximate volume using the export window, display FOV, and slice spacing you enter. |

**JPEGs are a fallback.** A display image stores window-levelled grey values,
not HU, so anything outside the export window is lost. For example, a
soft-tissue window turns aerated lung into roughly −160 HU, which the model may
read as consolidation. If you have to use JPEGs, export them with a wide
window (WL 0 / WW 2000) and enter the real FOV and slice spacing. A single
image, or a handful of them, is not enough for a 3D model; the page warns you
when coverage is too thin.

## Run it

You need Python 3.11+ and the model requirements:

```bash
git clone https://github.com/<you>/NV-Reason-CT.git
cd NV-Reason-CT
python -m pip install -r requirements.txt -r webapp/requirements.txt
python webapp/server.py
```

Then open <http://127.0.0.1:7860>. The first start downloads the ~10.6 GB
checkpoint from Hugging Face into `~/.cache/huggingface`.

| Hardware | Notes |
|---|---|
| NVIDIA GPU (Linux), 24 GB+ | The supported setup (NVIDIA tested on H100 and L40S, and the official demo runs on an A10G). |
| Apple Silicon Mac (MPS), 32 GB+ unified memory | Works but is slower. The model's linear-attention layers fall back to plain PyTorch on Macs. |
| CPU only | Starts, but a single answer can take a very long time. |

Useful flags:

```bash
python webapp/server.py --device cuda:1          # pick a GPU (default: auto)
python webapp/server.py --model /path/to/ckpt    # local or fine-tuned checkpoint
python webapp/server.py --mock                   # no weights: real conversion + preview, canned answers
python webapp/server.py --no-prefix-cache        # re-encode the scan for every question (see below)
python webapp/server.py --host 0.0.0.0           # listen on the network (trusted networks only)
```

### One-click start on a Mac

Create a conda environment named `nvreasonct` once:

```bash
conda create -n nvreasonct python=3.11
conda run -n nvreasonct python -m pip install -r requirements.txt -r webapp/requirements.txt
```

After that, double-click `webapp/start_mac.command` in Finder. It starts the
server in a Terminal window, opens the page in your default browser when the
server is ready, and stops the server when you close that window. To keep it
handy, drag the file to the right-hand side of the Dock. To use a different
Python, set `NVRCT_PYTHON`.

Run `server.py` as a script (as above), not from the repository root with
`python -c` or `python -m`. Otherwise, the repository's `accelerate/` config
folder shadows the `accelerate` package, and Transformers fails with
`Invalid version: 'N/A'`.

### Using a remote GPU server

Run the server on the GPU machine, keep it bound to localhost, and tunnel the
port to your laptop:

```bash
ssh -L 7860:127.0.0.1:7860 user@gpu-box   # then on the GPU box: python webapp/server.py
```

Open <http://127.0.0.1:7860> on your laptop. Dropped files are uploaded only to
that server.

## Privacy

- The page loads no third-party scripts, fonts, or analytics. Uploaded files go
  only to the server you started.
- Uploads and converted volumes are kept in a temporary folder. The four most
  recent cases are kept, older ones are deleted, and the folder is removed when
  the server exits. **Clear** deletes a case immediately.
- The converted NIfTI keeps only voxel data and geometry, with no DICOM header
  fields. You are still responsible for de-identification and for handling
  data under your institution's rules.

## Using the page

1. Drop files anywhere on the page, or use **Choose files** / **Choose folder**.
2. Check the **Model input** viewer. It shows the exact 192 × 192 × 192 crop at
   2 mm that the model reads, in axial, coronal, and sagittal planes. Scroll
   or use the arrow keys to move through slices. Press **1** for soft tissue
   (W 400 / L 40), **2** for lung (W 1500 / L −600), **3** for liver
   (W 150 / L 30), or **4** for bone (W 1800 / L 400). The keys are ignored
   while you type in a text box. Display windows don't change what the model
   reads, which is always the full HU range. To make the image bigger, drag
   the divider between the image and the conversation (your browser remembers
   the width; double-click the divider to reset it), or press **F** or click
   **Expand** to fill the window with the viewer (**Esc** or **F** to return).
   The image is the 2 mm model input, so enlarging it magnifies 2 mm voxels
   rather than revealing finer detail. If the anatomy you care about is
   cut off, switch between **Chest** and **Abdomen**. Switching starts a new
   conversation, because the model input changes.
3. Use a prompt chip (**Structured report**, **Reasoning analysis**, a sample
   yes/no question) or type your own. Follow-up questions keep the context of
   the conversation.
4. Turn off **Show reasoning** for short, direct answers. Use **Stop** to
   cancel a long generation.
5. Download the converted volume or the model-input crop (both `.nii`)
   to inspect them in 3D Slicer, ITK-SNAP, or another viewer.

## Large, full-resolution scans

Drop them as they are. Don't downsample or thin them out first.

- The converted volume keeps the scan's full resolution. The **Volume** panel
  shows its real matrix and voxel size.
- The model itself always reads a 192 × 192 × 192 crop at 2 mm (a 38.4 cm
  cube). The model's own preprocessing does that reduction (bilinear
  resampling to 2 mm, then the anatomy-aware crop), exactly as in NVIDIA's
  pipeline. The page does no extra downsizing of its own, so what the model
  sees is what it was evaluated on.
- The flip side: detail finer than about 2 mm, such as the edges of a 3 mm
  nodule, is averaged out before the model sees it. A 0.6 mm reconstruction
  and a 2 mm reconstruction of the same scan give the model nearly the same
  input.

Measured on an M2 Max (32 GB) with a 512 × 512 × 1,091-slice series (0.78 ×
0.78 × 0.625 mm, 573 MB, 1,091 files): the volume was ready 3.6 s after the
drop, and the model-input crop showed at 6 s. Peak memory was about 1.2 GB
during conversion and 2.4 GB during cropping. Both grow roughly linearly with
the number of slices.

## Faster follow-up questions

Every prompt about a scan starts with the same ~13,830 tokens: the opening of
the first message and the 13,824 image tokens. Encoding them is most of the
wait before the first word of an answer. The server encodes the image part
once per scan and region and keeps the model's state, so every later question
on that scan starts from it and reads only the new text. This covers
follow-ups, new questions after **New conversation**, and switching back to a
region you already used. The prompt the model sees is exactly the same as
before; only the work is split at the end of the image. The page shows
"Reusing the encoded scan" when it applies.

- The last two scan/region combinations are kept, at about 0.5 GB each.
  **Clear** and rebuilding the volume (switching series or changing the JPEG
  settings) discard them.
- Qwen3.5's linear-attention layers need a small patch to continue from a
  saved state when given several new tokens at once (Transformers 5.6.2 only
  supports this one token at a time). See `prefix_cache.py`.
- `python webapp/test_prefix_cache.py` checks that answers from the reused
  scan match re-encoding the whole prompt. It needs the model and takes a few
  minutes. If you ever doubt the cache, start the server with
  `--no-prefix-cache`.

## Files

- `server.py` — FastAPI server: uploads, conversion, cropping, and streamed generation.
- `convert.py` — DICOM, JPEG/PNG, and NIfTI conversion to a HU volume.
- `prefix_cache.py` — reuses the encoded scan across questions.
- `static/index.html` — the page (plain HTML/CSS/JS, no build step).
- `test_convert.py` — round-trip tests on a synthetic phantom: `python -m pytest webapp/test_convert.py`.
- `test_prefix_cache.py` — checks that reusing the encoded scan gives the same answers (needs the model).

## API

The page uses a small JSON API, which you can also script against:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Model loading state and device |
| `POST` | `/api/cases` | Multipart upload (`files` fields, optional `options` JSON); returns the case |
| `POST` | `/api/cases/{id}/convert` | Re-convert with `{"series_uid": ...}` or `{"jpeg": {...}}` |
| `GET` | `/api/cases/{id}/crop/{chest\|abdomen}` | Model-input crop as gzip-compressed little-endian int16 HU, shape in `X-Shape` (Z,Y,X) |
| `GET` | `/api/cases/{id}/download/{volume\|chest\|abdomen}` | Download the converted volume or a crop |
| `POST` | `/api/generate` | `{"case_id", "region", "prompt", "history", "thinking", "max_new_tokens"}`; Server-Sent Events stream |
| `DELETE` | `/api/cases/{id}` | Delete a case and its files |
