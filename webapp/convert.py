"""Convert dropped DICOM, JPEG/PNG or NIfTI files into a NIfTI CT volume in Hounsfield units.

NV-Reason-CT reads 3D CT volumes in Hounsfield units (HU). This module turns whatever
the user dropped on the page into one such volume:

* DICOM: files are grouped by SeriesInstanceUID, slices are ordered along the slice
  normal using ImagePositionPatient, pixel values are rescaled to HU and the patient
  geometry (LPS) is written into a RAS NIfTI affine.
* JPEG/PNG: a stack of axial slices is approximated. Display images carry no HU, so
  pixel values are mapped back through the window they were exported with, and the
  geometry comes from user-supplied field of view and slice spacing.
* NIfTI: validated and used as-is.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NON_DICOM_SUFFIXES = {".txt", ".xml", ".json", ".pdf", ".html", ".htm", ".csv", ".md", ".exe", ".zip", ".ini"}
SKIP_NAMES = {"dicomdir", "thumbs.db", "desktop.ini"}
MAX_ZIP_BYTES = 8 * 1024**3

# Model field of view: 192 voxels at 2 mm in every direction.
MODEL_FOV_MM = 384.0


class ConversionError(ValueError):
    """Raised when the dropped files cannot be turned into a CT volume."""


@dataclass
class InputFile:
    path: Path
    name: str  # original (relative) name from the browser, used for ordering


@dataclass
class SeriesInfo:
    uid: str
    description: str
    modality: str
    images: int
    rows: int
    cols: int
    files: list[Path] = field(default_factory=list, repr=False)

    def to_json(self):
        return {
            "uid": self.uid,
            "description": self.description,
            "modality": self.modality,
            "images": self.images,
            "size": f"{self.cols}×{self.rows}",
        }


@dataclass
class JpegOptions:
    window_center: float = 40.0
    window_width: float = 400.0
    fov_mm: float = 350.0
    slice_spacing_mm: float = 2.5
    first_is_superior: bool = True

    @classmethod
    def from_json(cls, data):
        data = data or {}
        opts = cls()
        for key in ("window_center", "window_width", "fov_mm", "slice_spacing_mm"):
            if data.get(key) not in (None, ""):
                setattr(opts, key, float(data[key]))
        if "first_is_superior" in data:
            opts.first_is_superior = bool(data["first_is_superior"])
        if opts.window_width <= 0 or opts.fov_mm <= 0 or opts.slice_spacing_mm <= 0:
            raise ConversionError("Window width, field of view and slice spacing must be positive.")
        return opts

    def to_json(self):
        return dict(self.__dict__)


@dataclass
class Volume:
    path: Path
    source: str  # "dicom" | "jpeg" | "nifti"
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    warnings: list[str]
    description: str = ""

    def to_json(self):
        extent = [round(n * s, 1) for n, s in zip(self.shape, self.spacing)]
        return {
            "source": self.source,
            "shape": list(self.shape),
            "spacing": [round(s, 3) for s in self.spacing],
            "extent_mm": extent,
            "warnings": self.warnings,
            "description": self.description,
        }


# --------------------------------------------------------------------------------------
# Sorting and classification


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _is_nifti(name):
    lower = name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _suffix(name):
    return Path(name).suffix.lower()


def expand_zips(files, workdir):
    """Replace every .zip in ``files`` with its members, extracted under ``workdir``."""
    out = []
    for item in files:
        if _suffix(item.name) != ".zip":
            out.append(item)
            continue
        try:
            archive = zipfile.ZipFile(item.path)
        except zipfile.BadZipFile as exc:
            raise ConversionError(f"{item.name} is not a valid zip file.") from exc
        with archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if sum(m.file_size for m in members) > MAX_ZIP_BYTES:
                raise ConversionError(f"{item.name} expands to more than 8 GB; unzip it first.")
            dest = workdir / f"zip_{len(out):04d}"
            dest.mkdir(parents=True, exist_ok=True)
            for i, member in enumerate(members):
                inner = member.filename
                if "__MACOSX" in inner or Path(inner).name.startswith("."):
                    continue
                # Never trust member paths: write to an index-based name, keep the suffix.
                suffix = ".nii.gz" if inner.lower().endswith(".nii.gz") else _suffix(inner)
                target = dest / f"{i:06d}{suffix}"
                with archive.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(InputFile(target, f"{item.name}/{inner}"))
    return out


def _looks_like_dicom(path):
    with open(path, "rb") as fh:
        head = fh.read(132)
    if len(head) == 132 and head[128:132] == b"DICM":
        return True
    # Some exports omit the 128-byte preamble; let pydicom decide.
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return False
    return "Rows" in ds and "Columns" in ds


def classify(files):
    """Split input files into DICOM, NIfTI and 2D image groups."""
    groups = {"dicom": [], "nifti": [], "image": []}
    for item in files:
        base = Path(item.name).name
        if base.startswith(".") or base.lower() in SKIP_NAMES or "__MACOSX" in item.name:
            continue
        if _is_nifti(item.name):
            groups["nifti"].append(item)
        elif _suffix(base) in IMAGE_SUFFIXES:
            groups["image"].append(item)
        elif _suffix(base) not in NON_DICOM_SUFFIXES and _looks_like_dicom(item.path):
            groups["dicom"].append(item)
    return groups


# --------------------------------------------------------------------------------------
# DICOM


def _num_frames(ds):
    try:
        return int(ds.get("NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        return 1


def scan_dicom_series(files):
    """Read headers and group files by series, best CT candidate first."""
    by_uid = defaultdict(list)
    frame_counts = Counter()
    headers = {}
    for item in files:
        try:
            ds = pydicom.dcmread(item.path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if "Rows" not in ds or "Columns" not in ds:
            continue
        uid = str(ds.get("SeriesInstanceUID", "unknown"))
        by_uid[uid].append(item.path)
        frame_counts[uid] += _num_frames(ds)
        headers.setdefault(uid, ds)

    series = []
    for uid, paths in by_uid.items():
        ds = headers[uid]
        frames = frame_counts[uid]
        number = ds.get("SeriesNumber", "")
        desc = str(ds.get("SeriesDescription", "") or "").strip()
        label = " · ".join(x for x in [f"Series {number}" if number not in ("", None) else "", desc] if x)
        series.append(SeriesInfo(
            uid=uid,
            description=label or "Unnamed series",
            modality=str(ds.get("Modality", "") or "?"),
            images=frames,
            rows=int(ds.Rows),
            cols=int(ds.Columns),
            files=sorted(paths),
        ))
    series.sort(key=lambda s: (s.modality != "CT", -s.images))
    return series


def _float_list(value, n):
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return out if len(out) == n else None


def _first_item(seq_parent, keyword):
    seq = seq_parent.get(keyword) if seq_parent is not None else None
    return seq[0] if seq else None


def _rescale(ds):
    slope = ds.get("RescaleSlope", None)
    intercept = ds.get("RescaleIntercept", None)
    return (float(slope) if slope not in (None, "") else None,
            float(intercept) if intercept not in (None, "") else None)


def _to_gray(pixels, photometric):
    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        rgb = pixels[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], True
    if photometric == "MONOCHROME1":
        return pixels.max() - pixels.astype(np.float32), False
    return pixels, False


def _dicom_slices(path):
    """Yield one dict per 2D slice (expands enhanced multi-frame objects)."""
    ds = pydicom.dcmread(path, force=True)
    pixels = ds.pixel_array
    photometric = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    frames = _num_frames(ds)
    if frames == 1:  # pydicom drops the frame axis for single-frame images
        pixels = pixels[np.newaxis]

    shared = _first_item(ds, "SharedFunctionalGroupsSequence")
    per_frame = ds.get("PerFrameFunctionalGroupsSequence")

    for f in range(frames):
        frame_groups = per_frame[f] if per_frame is not None and f < len(per_frame) else None

        def lookup(seq_keyword, attr):
            for group in (frame_groups, shared):
                item = _first_item(group, seq_keyword)
                if item is not None and attr in item:
                    return item.get(attr)
            return None

        ipp = _float_list(lookup("PlanePositionSequence", "ImagePositionPatient")
                          or ds.get("ImagePositionPatient"), 3)
        iop = _float_list(lookup("PlaneOrientationSequence", "ImageOrientationPatient")
                          or ds.get("ImageOrientationPatient"), 6)
        spacing = _float_list(lookup("PixelMeasuresSequence", "PixelSpacing")
                              or ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing"), 2)
        thickness = lookup("PixelMeasuresSequence", "SliceThickness") or ds.get("SliceThickness")
        slope = lookup("PixelValueTransformationSequence", "RescaleSlope")
        intercept = lookup("PixelValueTransformationSequence", "RescaleIntercept")
        if slope is None and intercept is None:
            slope, intercept = _rescale(ds)

        gray, was_color = _to_gray(pixels[f], photometric)
        yield {
            "pixels": gray,
            "color": was_color,
            "ipp": np.array(ipp) if ipp else None,
            "iop": np.array(iop) if iop else None,
            "spacing": spacing,
            "thickness": float(thickness) if thickness not in (None, "") else None,
            "slope": float(slope) if slope not in (None, "") else None,
            "intercept": float(intercept) if intercept not in (None, "") else None,
            "instance": int(ds.get("InstanceNumber", 0) or 0) * 10000 + f,
            "modality": str(ds.get("Modality", "") or ""),
            "between": ds.get("SpacingBetweenSlices"),
        }


def dicom_to_nifti(series, out_path):
    warnings = []
    slices = []
    unreadable = 0
    for path in series.files:
        try:
            slices.extend(_dicom_slices(path))
        except Exception as exc:  # compressed syntax without a decoder, truncated file, ...
            unreadable += 1
            last_error = exc
    if not slices:
        msg = f"Could not decode any image in {series.description}."
        if unreadable:
            msg += f" Last error: {last_error}"
        raise ConversionError(msg)
    if unreadable:
        warnings.append(f"Skipped {unreadable} file(s) that could not be decoded.")

    # Keep the dominant geometry: drops localizers or reformats mixed into one series.
    def geom_key(s):
        iop = tuple(np.round(s["iop"], 2)) if s["iop"] is not None else None
        return (s["pixels"].shape, iop)

    counts = Counter(geom_key(s) for s in slices)
    main_key, _ = counts.most_common(1)[0]
    if len(counts) > 1:
        dropped = len(slices) - counts[main_key]
        warnings.append(f"Ignored {dropped} image(s) with a different size or orientation (e.g. localizers).")
    slices = [s for s in slices if geom_key(s) == main_key]

    modality = slices[0]["modality"]
    if modality and modality != "CT":
        warnings.append(f"Series modality is {modality}; NV-Reason-CT was trained on CT in Hounsfield units.")
    if any(s["color"] for s in slices):
        warnings.append("Color DICOM images (likely screenshots) were converted to grayscale; values are not HU.")
    if all(s["slope"] is None and s["intercept"] is None for s in slices):
        warnings.append("No RescaleSlope/RescaleIntercept found; pixel values were used as HU directly.")

    spacing = slices[0]["spacing"] or [1.0, 1.0]
    if slices[0]["spacing"] is None:
        warnings.append("No PixelSpacing found; assumed 1 mm pixels.")
    row_spacing, col_spacing = spacing  # between rows, between columns

    has_geometry = all(s["ipp"] is not None and s["iop"] is not None for s in slices)
    if has_geometry:
        iop = slices[0]["iop"]
        row_cos, col_cos = iop[:3], iop[3:]
        normal = np.cross(row_cos, col_cos)
        slices.sort(key=lambda s: float(np.dot(s["ipp"], normal)))
        positions = np.array([float(np.dot(s["ipp"], normal)) for s in slices])

        keep = [0]
        for i in range(1, len(slices)):
            if abs(positions[i] - positions[keep[-1]]) > 1e-3:
                keep.append(i)
        if len(keep) < len(slices):
            warnings.append(f"Dropped {len(slices) - len(keep)} duplicate slice position(s).")
            slices = [slices[i] for i in keep]
            positions = positions[keep]

        if len(slices) > 1:
            gaps = np.diff(positions)
            step = float(np.median(gaps))
            if gaps.max() - gaps.min() > max(0.1 * step, 0.05):
                warnings.append(
                    f"Slice spacing is not uniform ({gaps.min():.2f}–{gaps.max():.2f} mm); "
                    f"using the median {step:.2f} mm."
                )
            slice_vec = (slices[-1]["ipp"] - slices[0]["ipp"]) / (len(slices) - 1)
            tilt = np.degrees(np.arccos(np.clip(abs(np.dot(slice_vec, normal)) / np.linalg.norm(slice_vec), -1, 1)))
            if tilt > 1.0:
                warnings.append(f"Gantry tilt of {tilt:.1f}° detected; geometry is kept as acquired.")
            if abs(np.linalg.norm(slice_vec) - step) > 0.05 * step and tilt <= 1.0:
                slice_vec = normal * step
        else:
            thickness = slices[0]["thickness"] or 1.0
            slice_vec = normal * thickness

        lps = np.eye(4)
        lps[:3, 0] = row_cos * col_spacing
        lps[:3, 1] = col_cos * row_spacing
        lps[:3, 2] = slice_vec
        lps[:3, 3] = slices[0]["ipp"]
    else:
        warnings.append("Images lack patient position/orientation tags; assumed axial slices in instance order.")
        slices.sort(key=lambda s: s["instance"])
        between = slices[0]["between"] or slices[0]["thickness"] or 1.0
        lps = np.diag([col_spacing, row_spacing, -float(between), 1.0])  # first slice superior

    hu = []
    for s in slices:
        px = s["pixels"].astype(np.float32)
        slope = s["slope"] if s["slope"] is not None else 1.0
        intercept = s["intercept"] if s["intercept"] is not None else 0.0
        hu.append(px * slope + intercept)
    # (rows, cols, slices) -> NIfTI (i=column, j=row, k=slice)
    vol = np.stack(hu, axis=-1).transpose(1, 0, 2)
    vol = np.clip(np.rint(vol), -32768, 32767).astype(np.int16)

    ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps
    _save(vol, ras, out_path)
    if len(slices) < 16:
        warnings.append(_thin_coverage_warning(len(slices), np.linalg.norm(lps[:3, 2])))
    zooms = (col_spacing, row_spacing, float(np.linalg.norm(lps[:3, 2])))
    return Volume(out_path, "dicom", vol.shape, zooms, warnings, series.description)


# --------------------------------------------------------------------------------------
# JPEG / PNG stacks


def _load_gray(path):
    with Image.open(path) as im:
        mode = im.mode
        color = False
        if mode in ("RGB", "RGBA", "P", "LA", "CMYK", "YCbCr"):
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
            color = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()) > 2.0
            arr = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        elif mode in ("I;16", "I;16B", "I;16L", "I"):
            raw = np.asarray(im, dtype=np.float32)
            arr = raw / (65535.0 if raw.max() > 255 else 255.0)
        else:
            arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return arr, color


def images_to_nifti(files, out_path, opts):
    files = sorted(files, key=lambda f: natural_key(f.name))
    low, high = opts.window_center - opts.window_width / 2.0, opts.window_center + opts.window_width / 2.0
    warnings = [
        "JPEG/PNG images carry no Hounsfield units. Values were reconstructed from the "
        f"window you entered (WL {opts.window_center:g} / WW {opts.window_width:g}), so everything "
        f"below {low:g} HU or above {high:g} HU is clipped and burned-in text becomes 'tissue'. "
        "Treat results as approximate; DICOM is always better."
    ]
    if low > -700:
        warnings.append(
            f"With this window, aerated lung (about -850 HU) is flattened to {low:g} HU and may be read "
            "as consolidation or ground glass. If your viewer allows, export with a wide window such as "
            "WL 0 / WW 2000."
        )
    images = []
    color = False
    for item in files:
        try:
            arr, is_color = _load_gray(item.path)
        except Exception as exc:
            raise ConversionError(f"Could not read image {item.name}: {exc}") from exc
        images.append(arr)
        color |= is_color
    if color:
        warnings.append("Some images contain color (annotations or overlays?); they were converted to grayscale.")

    shapes = Counter(a.shape for a in images)
    target, _ = shapes.most_common(1)[0]
    if len(shapes) > 1:
        warnings.append(f"Images had different sizes; resized all to {target[1]}×{target[0]}.")
        images = [
            a if a.shape == target
            else np.asarray(Image.fromarray(a).resize((target[1], target[0]), Image.BILINEAR))
            for a in images
        ]

    stack = np.stack(images, axis=-1) * opts.window_width + low  # (rows, cols, slices) in HU
    vol = np.clip(np.rint(stack.transpose(1, 0, 2)), -32768, 32767).astype(np.int16)

    pixel = opts.fov_mm / float(target[1])
    dz = opts.slice_spacing_mm
    # Standard radiological display: column 0 = patient right, row 0 = anterior.
    ras = np.diag([-pixel, -pixel, -dz if opts.first_is_superior else dz, 1.0])
    _save(vol, ras, out_path)
    if len(images) < 16:
        warnings.append(_thin_coverage_warning(len(images), dz))
    desc = f"{len(images)} image slice(s)"
    return Volume(out_path, "jpeg", vol.shape, (pixel, pixel, dz), warnings, desc)


# --------------------------------------------------------------------------------------
# NIfTI


def nifti_passthrough(item, out_path):
    try:
        img = nib.load(str(item.path))
        shape = img.shape
    except Exception as exc:
        raise ConversionError(f"Could not read NIfTI {item.name}: {exc}") from exc
    warnings = []
    if len(shape) == 4 and shape[3] == 1:
        img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine, img.header)
    elif len(shape) != 3:
        raise ConversionError(f"{item.name} has shape {shape}; expected a 3D volume.")
    data = np.asarray(img.dataobj)
    if float(np.percentile(data, 1)) > -200:
        warnings.append("Intensities do not look like Hounsfield units (no air near -1000 HU).")
    if len(shape) == 3:
        shutil.copyfile(item.path, out_path)
    else:
        nib.save(img, str(out_path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return Volume(out_path, "nifti", tuple(int(s) for s in img.shape[:3]), zooms, warnings, Path(item.name).name)


# --------------------------------------------------------------------------------------
# Entry point


def _thin_coverage_warning(n, spacing_mm):
    return (
        f"Only {n} slice(s) covering about {n * spacing_mm:.0f} mm. NV-Reason-CT reads a "
        f"{MODEL_FOV_MM:.0f} mm 3D field; with this little anatomy the output is not meaningful."
    )


def _save(vol, affine, out_path):
    img = nib.Nifti1Image(vol, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(out_path))


def build_volume(files, workdir, *, series_uid=None, jpeg=None):
    """Convert dropped files into ``workdir/volume.nii.gz``.

    Returns ``(Volume, series_list, chosen_series_uid)``; the series list is empty and the
    uid is None unless DICOM was found.
    """
    workdir = Path(workdir)
    out_path = workdir / "volume.nii.gz"
    groups = classify(files)

    if groups["dicom"]:
        series = scan_dicom_series(groups["dicom"])
        if not series:
            raise ConversionError("Found DICOM files but none contained image data.")
        chosen = next((s for s in series if s.uid == series_uid), series[0])
        volume = dicom_to_nifti(chosen, out_path)
        if groups["image"] or groups["nifti"]:
            volume.warnings.append("Non-DICOM files in the drop were ignored.")
        return volume, series, chosen.uid

    if groups["nifti"]:
        nifti = sorted(groups["nifti"], key=lambda f: natural_key(f.name))
        suffix = ".nii.gz" if nifti[0].name.lower().endswith(".gz") else ".nii"
        volume = nifti_passthrough(nifti[0], workdir / f"volume{suffix}")
        if len(nifti) > 1:
            volume.warnings.append(f"{len(nifti)} NIfTI files dropped; using {Path(nifti[0].name).name}.")
        return volume, [], None

    if groups["image"]:
        return images_to_nifti(groups["image"], out_path, jpeg or JpegOptions()), [], None


    raise ConversionError(
        "No usable images found. Drop a DICOM folder or files, a .zip, a .nii/.nii.gz volume, "
        "or a stack of JPEG/PNG slices."
    )
