"""Round-trip tests for convert.py using a synthetic CT phantom.

Run with:  python -m pytest webapp/test_convert.py   (or: python webapp/test_convert.py)
"""

import random
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import InputFile, JpegOptions, build_volume  # noqa: E402


COLS, ROWS, SLICES = 64, 48, 20
ROW_SPACING, COL_SPACING, STEP = 0.8, 0.7, 2.5  # DICOM PixelSpacing = [row, col]
ORIGIN = np.array([-20.0, -15.0, 100.0])  # LPS position of the first (most inferior) slice


def phantom():
    """HU volume indexed [column, row, slice] with an asymmetric marker to catch flips."""
    vol = np.full((COLS, ROWS, SLICES), -1000, dtype=np.int16)
    vol[8:56, 6:42, :] = 40  # soft tissue body
    vol[40:50, 8:14, 14:18] = 700  # dense marker: patient left, anterior, superior
    return vol


def expected_ras_affine():
    lps = np.eye(4)
    lps[:3, 0] = [COL_SPACING, 0, 0]
    lps[:3, 1] = [0, ROW_SPACING, 0]
    lps[:3, 2] = [0, 0, STEP]
    lps[:3, 3] = ORIGIN
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps


def write_slice(path, pixels, ipp, iop, spacing, series_uid, instance, description="Axial"):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 2
    ds.SeriesDescription = description
    ds.StudyDescription = "CT CHEST TEST"
    ds.InstanceNumber = instance
    ds.ImagePositionPatient = [float(v) for v in ipp]
    ds.ImageOrientationPatient = [float(v) for v in iop]
    ds.PixelSpacing = list(spacing)
    ds.SliceThickness = STEP
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)


def write_dicom_series(folder, vol):
    folder.mkdir(parents=True, exist_ok=True)
    uid = generate_uid()
    order = list(range(SLICES))
    random.Random(0).shuffle(order)  # file order must not matter
    for n, k in enumerate(order):
        stored = vol[:, :, k].T.astype(np.int32) + 1024  # (rows, cols)
        ipp = ORIGIN + np.array([0, 0, STEP * k])
        write_slice(folder / f"IM{n:04d}", stored, ipp, [1, 0, 0, 0, 1, 0],
                    [ROW_SPACING, COL_SPACING], uid, SLICES - k)
    # A localizer in the same series (different size/orientation) must be ignored.
    write_slice(folder / "SCOUT", np.zeros((30, 40)), [0, 0, 0], [1, 0, 0, 0, 0, -1],
                [1.0, 1.0], uid, 999)
    # A second, smaller series; the bigger CT series should be picked by default.
    other = generate_uid()
    for k in range(3):
        write_slice(folder / f"OTHER{k}", np.zeros((16, 16)), [0, 0, 5.0 * k], [1, 0, 0, 0, 1, 0],
                    [1.0, 1.0], other, k + 1, description="Dose report")


def inputs_for(folder):
    return [InputFile(p, str(p.relative_to(folder.parent))) for p in sorted(folder.rglob("*")) if p.is_file()]


def check_matches_phantom(nifti_path, vol):
    img = nib.load(str(nifti_path))
    assert img.shape == vol.shape, img.shape
    np.testing.assert_allclose(img.affine, expected_ras_affine(), atol=1e-4)
    np.testing.assert_array_equal(np.asarray(img.dataobj), vol)


def test_dicom_series_roundtrip():
    vol = phantom()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_dicom_series(tmp / "study", vol)
        volume, series, uid = build_volume(inputs_for(tmp / "study"), tmp)
        assert len(series) == 2 and series[0].uid == uid and series[0].images == SLICES + 1
        assert any("localizer" in w for w in volume.warnings), volume.warnings
        assert volume.study == "CT CHEST TEST", volume.study
        check_matches_phantom(volume.path, vol)

        # The other series can be selected explicitly.
        small, _, other_uid = build_volume(inputs_for(tmp / "study"), tmp, series_uid=series[1].uid)
        assert other_uid == series[1].uid and small.shape == (16, 16, 3)


def test_dicom_zip_roundtrip():
    vol = phantom()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_dicom_series(tmp / "study", vol)
        zip_path = tmp / "upload.bin"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in (tmp / "study").iterdir():
                zf.write(p, f"export/DICOM/{p.name}")
            zf.writestr("__MACOSX/._IM0000", b"junk")
        work = tmp / "work"
        work.mkdir()
        from convert import expand_zips
        files = expand_zips([InputFile(zip_path, "study.zip")], work)
        volume, _, _ = build_volume(files, work)
        check_matches_phantom(volume.path, vol)


def test_png_stack_orientation_and_window():
    vol = phantom()
    wl, ww = 40.0, 1000.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        folder = tmp / "pngs"
        folder.mkdir()
        # Export like a PACS: superior slice first, row 0 anterior, column 0 patient right.
        for n, k in enumerate(range(SLICES - 1, -1, -1)):
            display = vol[:, :, k].T  # (rows, cols); LPS column 0 is already patient right
            u8 = np.clip((display - (wl - ww / 2)) / ww * 255.0, 0, 255).round().astype(np.uint8)
            Image.fromarray(u8).save(folder / f"slice_{n + 1}.png")  # natural sort: 2 < 10
        opts = JpegOptions(window_center=wl, window_width=ww, fov_mm=COLS * COL_SPACING,
                           slice_spacing_mm=STEP, first_is_superior=True)
        volume, _, _ = build_volume(inputs_for(folder), tmp, jpeg=opts)
        img = nib.as_closest_canonical(nib.load(str(volume.path)))
        assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
        got = np.asarray(img.dataobj).astype(np.float32)
        # Canonical RAS: x from right to left reversed => index 0 is patient right.
        expected = vol[::-1, ::-1, :].astype(np.float32)  # LPS phantom -> RAS
        expected = np.clip(expected, wl - ww / 2, wl + ww / 2)  # values outside the window are lost
        assert np.abs(got - expected).max() <= ww / 255.0 + 1, np.abs(got - expected).max()
        assert any("Hounsfield" in w for w in volume.warnings)
        assert volume.study == "pngs", volume.study  # folder name when there is no DICOM study


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
