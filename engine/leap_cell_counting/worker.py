"""Standalone worker lifecycle, cancellation, shared-component lease, and artifacts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import traceback

from .errors import CountingCancelled, CountingError


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".counting-", suffix=".json",
                                         dir=str(path.parent), delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def component_usage_lock(*, required=False):
    value = os.environ.get("PROJECT_LEAP_COMPONENT_LOCK")
    if not value and not required:
        yield
        return
    if not value or not Path(value).is_absolute():
        raise CountingError("component_lock_required", "需要应用共享组件锁的绝对路径。")
    try:
        descriptor = os.open(value, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise CountingError("component_lock_unavailable", "无法取得共享组件使用锁。") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CountingError("component_busy", "组件正在维护，请稍后重试。") from exc
        yield
    finally:
        os.close(descriptor)


class Events:
    def __init__(self, request):
        self.run_id = request["run_id"]
        self.path = Path(request["events_path"])
        self.cancel_path = Path(request["cancel_path"])
        self.cancelled = False
        self.sequence = 0
        self.stage = "count_projection"
        self._event_lock = threading.Lock()

    def emit(self, event_type, stage, message, **details):
        with self._event_lock:
            self.sequence += 1
            self.stage = stage
            event = {"schema_version": 1, "run_id": self.run_id, "sequence": self.sequence,
                     "timestamp": datetime.now(timezone.utc).isoformat(), "type": event_type,
                     "stage": stage, "message": message, **details}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def check_cancel(self):
        if self.cancelled or self.cancel_path.exists():
            raise CountingCancelled("细胞计数已取消。")

    def signal_cancel(self, _number, _frame):
        # Raise only at explicit checkpoints: an asynchronous exception can
        # interrupt executor.submit before the returned future is tracked.
        self.cancelled = True


def validate_request(request_path):
    from .paths import output_path
    path = Path(request_path)
    if not path.is_absolute():
        raise CountingError("invalid_request", "请求文件必须为绝对路径。")
    path = path.resolve(strict=True)
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or request.get("schema_version") not in (1, 2):
        raise CountingError("invalid_request", "请求必须为 schema_version=1/2 的 JSON 对象。")
    if not isinstance(request.get("run_id"), str) or not request["run_id"].strip():
        raise CountingError("invalid_request", "请求缺少 run_id。")
    source_value = request.get("source_path")
    if not isinstance(source_value, str) or not Path(source_value).is_absolute():
        raise CountingError("invalid_request", "source_path 必须为绝对路径。")
    source = Path(source_value).resolve(strict=True)
    for key in ("run_dir", "events_path", "outcome_path", "cancel_path"):
        if not isinstance(request.get(key), str):
            raise CountingError("invalid_request", f"缺少 {key}。")
        request[key] = str(output_path(request[key], source))
    run = Path(request["run_dir"])
    if run != path.parent:
        raise CountingError("invalid_request", "请求必须保存在其 run_dir 中。")
    for key in ("events_path", "outcome_path", "cancel_path"):
        if Path(request[key]).parent != run:
            raise CountingError("invalid_request", f"{key} 必须直接位于当前 run_dir 中。")
    paths = [path, *(Path(request[k]) for k in ("events_path", "outcome_path", "cancel_path")), run / "worker.json"]
    if len(set(paths)) != len(paths):
        raise CountingError("invalid_request", "请求、进程登记、事件、取消和结果路径不得重叠。")
    if (run / "counting").exists() or Path(request["outcome_path"]).exists():
        raise CountingError("existing_run", "此任务已有输出，请创建新任务。")
    event_path = Path(request["events_path"])
    if event_path.exists() and event_path.stat().st_size:
        raise CountingError("existing_run", "此任务已有事件，请创建新任务。")
    request["source_path"] = str(source)
    return request


def _runtime_cache(run):
    cache = run / "cache"
    for key, relative in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba"),
                          ("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg"), ("TMPDIR", "tmp")):
        location = cache / relative
        if cache.is_symlink() or location.is_symlink():
            raise CountingError("invalid_cache_path", "任务缓存目录不能指向外部位置。")
        location.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(location)
    tempfile.tempdir = str(cache / "tmp")


def analyse(request, events):
    """One native workflow combines independent nucleus and body observations."""
    from .efficient import analyse as efficient_analysis
    return efficient_analysis(request, events)


def run_request(request_path):
    # Keep the lease through final cleanup, outcome and event publication: a
    # maintenance process must never delete the run while its final writes run.
    with component_usage_lock(required=True):
        return _run_request_locked(request_path)


def _run_request_locked(request_path):
    request = validate_request(request_path)
    events = Events(request)
    run = Path(request["run_dir"])
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        atomic_json(run / "worker.json", {"pid": os.getpid(), "kind": "analysis", "module_id": "cell-counting",
                                         "request_path": str(Path(request_path).resolve())})
        for sig in old_handlers:
            signal.signal(sig, events.signal_cancel)
        events.check_cancel()
        _runtime_cache(run)
        outcome = analyse(request, events)
    except (CountingCancelled, KeyboardInterrupt):
        outcome = {"status": "cancelled", "error_code": "user_cancelled", "message": "细胞计数已取消。"}
    except Exception as exc:
        traceback.print_exc()
        outcome = {"status": "failed", "error_code": getattr(exc, "code", "technical_failure"), "message": str(exc)}
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if not (run / "counting/analysis.json").exists():
            for filename in ("labels.npy", "body-labels.npy"):
                (run / "counting" / filename).unlink(missing_ok=True)
    outcome.update(schema_version=1, run_id=request["run_id"], module_id="cell-counting")
    atomic_json(request["outcome_path"], outcome)
    events.emit(outcome["status"], events.stage, outcome["message"],
                **{k: v for k, v in outcome.items() if k not in ("status", "message", "schema_version", "run_id")})
    print(json.dumps(outcome, ensure_ascii=False, allow_nan=False), flush=True)
    return {"review_ready": 0, "cancelled": 130, "failed": 1}[outcome["status"]]
