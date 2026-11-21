"""macOS privacy-permission requests owned by the NextCore app process."""

from __future__ import annotations

import ctypes
import sys


class MacOSPermissionRequiredError(RuntimeError):
    """The user must grant a macOS privacy permission before computer use."""


def _request_accessibility_permission() -> bool:
    """Ask macOS to prompt for Accessibility access for the responsible app."""
    application_services = ctypes.CDLL(
        "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
    )
    core_foundation = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )

    create_dictionary = core_foundation.CFDictionaryCreateMutable
    create_dictionary.argtypes = [
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    create_dictionary.restype = ctypes.c_void_p
    set_value = core_foundation.CFDictionarySetValue
    set_value.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    set_value.restype = None
    release = core_foundation.CFRelease
    release.argtypes = [ctypes.c_void_p]
    release.restype = None

    options = create_dictionary(None, 1, None, None)
    if not options:
        raise OSError("Could not create the macOS Accessibility permission request.")
    try:
        prompt_key = ctypes.c_void_p.in_dll(
            application_services, "kAXTrustedCheckOptionPrompt"
        ).value
        true_value = ctypes.c_void_p.in_dll(core_foundation, "kCFBooleanTrue").value
        if not prompt_key or not true_value:
            raise OSError("macOS Accessibility permission symbols are unavailable.")
        set_value(options, prompt_key, true_value)

        check = application_services.AXIsProcessTrustedWithOptions
        check.argtypes = [ctypes.c_void_p]
        check.restype = ctypes.c_bool
        return bool(check(options))
    finally:
        release(options)


def _request_screen_recording_permission() -> bool:
    """Request Screen Recording access for the responsible NextCore app."""
    core_graphics = ctypes.CDLL(
        "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
    )
    preflight = core_graphics.CGPreflightScreenCaptureAccess
    preflight.argtypes = []
    preflight.restype = ctypes.c_bool
    if preflight():
        return True

    request = core_graphics.CGRequestScreenCaptureAccess
    request.argtypes = []
    request.restype = ctypes.c_bool
    return bool(request())


def request_computer_use_permissions() -> None:
    """Request the two macOS grants needed before starting CUA Driver.

    The calls run in NextCore's Python app process before the managed driver
    starts. A false result can mean that the macOS prompt is still waiting for
    the user; the app must stop this action and let them grant access first.
    """
    if sys.platform != "darwin":
        return

    failures: list[str] = []
    try:
        if not _request_accessibility_permission():
            failures.append("Accessibility")
    except Exception:
        failures.append("Accessibility")

    try:
        if not _request_screen_recording_permission():
            failures.append("Screen Recording")
    except Exception:
        failures.append("Screen Recording")

    if failures:
        missing = " and ".join(failures)
        raise MacOSPermissionRequiredError(
            f"NextCore needs {missing} permission for computer control. macOS has "
            "not granted the required access yet. Enable Token Workshed NextCore "
            "in System Settings > Privacy & Security > Accessibility and "
            "Screen & System Audio Recording. After changing access, quit and "
            "reopen NextCore so macOS applies the grant."
        )
