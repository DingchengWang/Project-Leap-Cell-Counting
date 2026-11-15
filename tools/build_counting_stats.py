#!/usr/bin/env python3
"""Compile and inspect the fixed arm64 statistics library shipped with counting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libLeapCountingStats.dylib"
PACKAGE_PATH = "engine/leap_cell_counting/" + LIBRARY_NAME
SOURCE_PATH = "native/counting_stats.c"
EXPECTED_EXPORTS = {"leap_counting_stats_abi_version", "leap_counting_stats_u8", "leap_counting_stats_u16"}
ABI_VERSION = 1
MINIMUM_MACOS = "27.0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library_contract(library):
    """Inspect the actual Mach-O and load its ABI entry point in a clean process."""
    library = Path(library).resolve()
    arch = subprocess.run(["/usr/bin/lipo", "-archs", str(library)], text=True, capture_output=True)
    names = subprocess.run(["/usr/bin/nm", "-gjU", str(library)], text=True, capture_output=True)
    links = subprocess.run(["/usr/bin/otool", "-L", str(library)], text=True, capture_output=True)
    commands = subprocess.run(["/usr/bin/otool", "-l", str(library)], text=True, capture_output=True)
    signature = subprocess.run(["/usr/bin/codesign", "--verify", "--strict", str(library)], text=True, capture_output=True)
    symbols = sorted(line.strip()[1:] for line in names.stdout.splitlines() if line.strip().startswith("_"))
    linked = [line.strip().split(" (", 1)[0] for line in links.stdout.splitlines()[1:] if line.strip()]
    install_name = linked[0] if linked else None
    dependencies = linked[1:]
    minimum_macos = None
    rpaths = []
    command = None
    for line in commands.stdout.splitlines():
        value = line.strip()
        if value.startswith("cmd "):
            command = value.split(None, 1)[1]
        elif command == "LC_BUILD_VERSION" and value.startswith("minos "):
            minimum_macos = value.split(None, 1)[1]
        elif command == "LC_RPATH" and value.startswith("path "):
            rpaths.append(value[5:].split(" (offset", 1)[0])
    missing = sorted(EXPECTED_EXPORTS - set(symbols))
    unexpected = sorted(set(symbols) - EXPECTED_EXPORTS)
    system_dependencies = bool(dependencies and all(path.startswith("/usr/lib/") for path in dependencies))
    static_passed = bool(all(result.returncode == 0 for result in (arch, names, links, commands, signature))
                         and arch.stdout.strip() == "arm64" and minimum_macos == MINIMUM_MACOS
                         and install_name == "@rpath/" + LIBRARY_NAME and system_dependencies
                         and not rpaths and not missing and not unexpected)
    runtime_load = {"performed": False}
    if static_passed:
        env = {key: value for key, value in os.environ.items() if not key.startswith("DYLD_")}
        script = ("import ctypes,sys;lib=ctypes.CDLL(sys.argv[1]);"
                  "fn=lib.leap_counting_stats_abi_version;fn.argtypes=[];fn.restype=ctypes.c_int;"
                  "print(fn())")
        try:
            loaded = subprocess.run([sys.executable, "-c", script, str(library)], cwd="/private/tmp",
                                    env=env, text=True, capture_output=True, timeout=20)
            runtime_load = {"performed": True, "exit_code": loaded.returncode, "abi": loaded.stdout.strip(),
                            "stderr": loaded.stderr, "tested_macos": platform.mac_ver()[0]}
        except subprocess.TimeoutExpired:
            runtime_load = {"performed": True, "timed_out": True, "tested_macos": platform.mac_ver()[0]}
    return {"architecture": arch.stdout.strip(), "minimum_macos": minimum_macos,
            "install_name": install_name, "dependencies": dependencies, "rpaths": rpaths,
            "exports": symbols, "missing_exports": missing, "unexpected_exports": unexpected,
            "codesign_exit_code": signature.returncode, "codesign_stderr": signature.stderr,
            "runtime_load": runtime_load,
            "passed": static_passed and runtime_load.get("exit_code") == 0 and runtime_load.get("abi") == str(ABI_VERSION)}


def build_library(source, output, identity="-"):
    source, output = Path(source).resolve(), Path(output).resolve()
    source_hash = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="Leap-Counting-Stats-build-", dir="/private/tmp") as directory:
        library = Path(directory) / LIBRARY_NAME
        command = ["/usr/bin/xcrun", "clang", "-dynamiclib", "-arch", "arm64", "-mmacosx-version-min=27.0",
                   "-O3", "-Wall", "-Wextra", "-Werror", "-fvisibility=hidden",
                   "-Wl,-install_name,@rpath/" + LIBRARY_NAME, str(source), "-o", str(library)]
        compiled = subprocess.run(command, text=True, capture_output=True)
        if compiled.returncode:
            raise RuntimeError("Counting statistics compilation failed:\n" + compiled.stderr)
        options = ["--options", "runtime", "--timestamp"] if identity != "-" else []
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", identity, *options, str(library)],
                       text=True, capture_output=True, check=True)
        contract = library_contract(library)
        if not contract["passed"]:
            raise RuntimeError("Counting statistics library contract failed: " + json.dumps(contract))
        if sha256(source) != source_hash:
            raise RuntimeError("Counting statistics source changed during compilation")
        with tempfile.NamedTemporaryFile(prefix=".LeapCountingStats-", dir=output.parent, delete=False) as target:
            replacement = Path(target.name)
        try:
            shutil.copy2(library, replacement)
            os.replace(replacement, output)
        finally:
            replacement.unlink(missing_ok=True)
    compiler = subprocess.run(["/usr/bin/xcrun", "clang", "--version"], text=True, capture_output=True, check=True)
    return {"path": PACKAGE_PATH, "sha256": sha256(output), "bytes": output.stat().st_size,
            "source_path": SOURCE_PATH, "source_sha256": source_hash, "abi_version": ABI_VERSION,
            "builder_sha256": sha256(Path(__file__)), "signing_identity": identity,
            "compiler": compiler.stdout.strip(), "command": command, "compiler_stderr": compiled.stderr,
            "verification": contract}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / SOURCE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", default="-")
    args = parser.parse_args()
    print(json.dumps(build_library(args.source, args.output, args.identity), indent=2))
