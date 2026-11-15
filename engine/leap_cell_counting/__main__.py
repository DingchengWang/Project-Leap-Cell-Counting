"""CLI for metadata, analysis and persistent Z frames."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Project Leap cell-counting worker")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inspect")
    modes.add_argument("--request")
    modes.add_argument("--slice")
    modes.add_argument("--slice-server")
    parser.add_argument("--preview-path")
    parser.add_argument("--z", type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    from .worker import component_usage_lock, run_request
    try:
        if args.request:
            if args.preview_path or args.output or args.z is not None:
                parser.error("--request does not accept preview or slice options")
            return run_request(args.request)
        with component_usage_lock():
            # Inspection and slices never import torch or Cellpose.
            from .imaging import create_preview, create_slice, inspect_image, serve_slices
            if args.slice_server:
                if args.preview_path or args.output or args.z is not None:
                    parser.error("--slice-server reads JSON-lines requests from stdin")
                return serve_slices(args.slice_server, sys.stdin, sys.stdout)
            if args.inspect:
                if args.output or args.z is not None:
                    parser.error("--inspect does not accept --output/--z")
                result = inspect_image(args.inspect)
                if args.preview_path:
                    try:
                        result.update(create_preview(args.inspect, args.preview_path))
                    except Exception as exc:
                        result["preview_error"] = {"code": getattr(exc, "code", "preview_failed"), "message": str(exc)}
            else:
                if args.z is None or not args.output or args.preview_path:
                    parser.error("--slice requires --z and --output")
                result = create_slice(args.slice, args.z, args.output)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_code": getattr(exc, "code", "technical_failure"),
                          "message": str(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
