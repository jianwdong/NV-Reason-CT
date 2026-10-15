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

   Press **O** or click **Original** to see the converted scan at its full
   resolution instead of the 2 mm model input. A dashed outline shows the
   model's 38 cm cube for the chosen region, so you can see what it includes
   and what it leaves out. Slices load as you scroll, and coronal and
   sagittal views keep the scan's true proportions. Press **O** again to
   return to the model input.

   To measure, hover over the image to read the HU of the voxel under the
   pointer. Press **R** (Ruler) and drag for a distance in mm, or press
   **E** (Oval ROI) and drag out an ellipse for mean ± SD HU, min, max, area
   and pixel count. A pixel counts as inside the ellipse when its center is.
   Measurements work in all three planes and stay on the slice where you drew
   them. The list under the image jumps back to that slice when you click an
   entry; **×**, **Delete** or **Clear measurements** removes them. On the
   original scan, measurements use its own voxel size. On the 2 mm model
   input, distances are approximate (about ±2 mm) and HU near edges are
   blurred by resampling. Model-input measurements are cleared when the crop
   changes (switching region), and all measurements when the volume changes
   (new scan or rebuild).

   If the anatomy you care about is
   cut off, switch between **Chest** and **Abdomen**. Switching starts a new
   conversation, because the model input changes.
3. Use a prompt chip (**Structured report**, **Reasoning analysis**, a sample
   yes/no question) or type your own. Follow-up questions include every
   earlier exchange shown in the conversation, including stopped or cut-off
   answers as far as they got. The note under each answer says how many
   earlier messages the model saw. Switching region, loading another scan,
   **New conversation**, or reloading the page starts a new conversation.
4. Turn off **Show reasoning** for short, direct answers. Use **Stop** to
   cancel a long generation.
5. Click **Export summary** to save the case as a single HTML file, named
   after the **Study name** in the Volume card. That name comes from the
   DICOM study description (or the file or folder name), and you can edit it.
   The file contains the scan details, the current image and every slice you
   measured on (with the measurements drawn in), the measurement table, and
   the whole conversation including reasoning. It opens in any browser and
   prints to PDF. It includes no patient name, ID or scan dates from the
   DICOM headers, but check the study name and any burned-in text before
   sharing.
6. Download the converted volume or the model-input crop (both `.nii`)
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

## Files

- `server.py` — FastAPI server: uploads, conversion, cropping, and streamed generation.
- `convert.py` — DICOM, JPEG/PNG, and NIfTI conversion to a HU volume.
- `static/index.html` — the page (plain HTML/CSS/JS, no build step).
- `test_convert.py` — round-trip tests on a synthetic phantom: `python -m pytest webapp/test_convert.py`.
- `LICENSE` — MIT License for the reader (this folder). NVIDIA's model and code keep their OpenMDW license.

## API

The page uses a small JSON API, which you can also script against:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/status` | Model loading state and device |
| `POST` | `/api/cases` | Multipart upload (`files` fields, optional `options` JSON); returns the case |
| `POST` | `/api/cases/{id}/convert` | Re-convert with `{"series_uid": ...}` or `{"jpeg": {...}}` |
| `GET` | `/api/cases/{id}/crop/{chest\|abdomen}` | Model-input crop as gzip-compressed little-endian int16 HU, shape in `X-Shape` and voxel size in mm in `X-Spacing` (both Z,Y,X) |
| `GET` | `/api/cases/{id}/original` | Original scan's shape and voxel size (Z,Y,X) and the crop outline per region, as `[z0, z1, y0, y1, x0, x1]` in image coordinates |
| `GET` | `/api/cases/{id}/original/{axial\|coronal\|sagittal}/{k}` | One original slice as gzip-compressed int16 HU, rows × columns in `X-Shape`, oriented as on screen |
| `GET` | `/api/cases/{id}/download/{volume\|chest\|abdomen}` | Download the converted volume or a crop |
| `POST` | `/api/generate` | `{"case_id", "region", "prompt", "history", "thinking", "max_new_tokens"}`; Server-Sent Events stream |
| `DELETE` | `/api/cases/{id}` | Delete a case and its files |

The server does not remember conversations: every `/api/generate` call must
send the earlier turns in `history`, oldest first, as
`[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]`.
For assistant turns, send the answer text. Any reasoning before `</think>` is
dropped by the model's chat template for earlier turns anyway. A call with an
empty `history` starts a new conversation about the scan.
