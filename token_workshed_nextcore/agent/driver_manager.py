"""Pinned, app-managed CUA Driver installation and integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

CUA_DRIVER_VERSION = "0.28.1"
CUA_DRIVER_SHA256 = "b3cb3781feb9855dfcdd616bc5c9c8e627453e3e16a073c1446326691a549b24"
CUA_DRIVER_URL = (
    "https://github.com/trycua/cua/releases/download/cua-driver-rs-v0.28.1/"
    "cua-driver-rs-0.28.1-darwin-arm64.tar.gz"
)


class DriverUnavailableError(RuntimeError):
    pass


def managed_driver_path(app_data_dir: Path) -> Path:
    return app_data_dir / "agent" / "bin" / "cua-driver"


def _managed_manifest_path(driver_path: Path) -> Path:
    return driver_path.with_name("cua-driver.manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_verified_managed_driver(driver_path: Path) -> bool:
    if (
        not driver_path.is_file()
        or driver_path.is_symlink()
        or not os.access(driver_path, os.X_OK)
    ):
        return False
    manifest_path = _managed_manifest_path(driver_path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            isinstance(manifest, dict)
            and manifest.get("version") == CUA_DRIVER_VERSION
            and manifest.get("archive_sha256") == CUA_DRIVER_SHA256
            and manifest.get("binary_sha256") == _sha256_file(driver_path)
        )
    except (OSError, ValueError, TypeError):
        return False


def ensure_driver(app_data_dir: Path, *, cancel_event=None) -> Path:
    """Download only the pinned official arm64 release and verify its digest."""
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("CUA Driver download stopped by the user.")
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise DriverUnavailableError("Computer control currently requires Apple Silicon macOS.")
    destination = managed_driver_path(app_data_dir)
    if _is_verified_managed_driver(destination):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nextcore-cua-") as temporary:
        archive = Path(temporary) / "driver.tar.gz"
        request = urllib.request.Request(
            CUA_DRIVER_URL,
            headers={"User-Agent": "Token-Workshed-NextCore"},
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response, archive.open("wb") as out:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("CUA Driver download stopped by the user.")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        except InterruptedError:
            raise
        except Exception as exc:
            raise DriverUnavailableError(f"Could not download CUA Driver {CUA_DRIVER_VERSION}: {exc}") from exc

        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("CUA Driver download stopped by the user.")

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != CUA_DRIVER_SHA256:
            raise DriverUnavailableError("Downloaded CUA Driver checksum did not match the pinned release.")

        extracted: Path | None = None
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle.getmembers():
                    name = Path(member.name)
                    if not member.isfile() or name.name != "cua-driver":
                        continue
                    if name.is_absolute() or ".." in name.parts:
                        continue
                    source = bundle.extractfile(member)
                    if source is None:
                        continue
                    staged = Path(temporary) / "cua-driver"
                    with source, staged.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    extracted = staged
                    break
        except (OSError, tarfile.TarError) as exc:
            raise DriverUnavailableError(f"The pinned CUA Driver archive is invalid: {exc}") from exc
        if extracted is None:
            raise DriverUnavailableError("The pinned CUA Driver archive did not contain its executable.")
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("CUA Driver installation stopped by the user.")
        staged_destination = destination.with_name("cua-driver.new")
        manifest_path = _managed_manifest_path(destination)
        staged_manifest = manifest_path.with_name(f"{manifest_path.name}.new")
        shutil.copyfile(extracted, staged_destination)
        staged_destination.chmod(0o700)
        binary_digest = _sha256_file(staged_destination)
        staged_manifest.write_text(
            json.dumps(
                {
                    "version": CUA_DRIVER_VERSION,
                    "archive_sha256": CUA_DRIVER_SHA256,
                    "binary_sha256": binary_digest,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        staged_manifest.chmod(0o600)
        os.replace(staged_destination, destination)
        os.replace(staged_manifest, manifest_path)
    return destination
