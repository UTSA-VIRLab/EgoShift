"""GPU memory accounting shared by the runner and the supervisor.

Every OOM this project has hit came from one of four causes, and this module
exists to make all four checkable *before* a job burns hours:

  * a model was loaded onto cards another job already occupied;
  * `device_map="auto"` filled every card to the brim with weights, leaving
    nothing for activations -- the model loads, then every single generate()
    dies (InternVL3-78B Task B: 1,098 consecutive OOMs, 69 minutes, no data);
  * allocator fragmentation, which `expandable_segments` largely removes;
  * nobody checked whether the weights could fit at all before starting.

Sizes come from the actual files in the HF cache rather than a hand-kept
table, so a re-quantised or re-sharded checkpoint cannot silently invalidate
the estimate.

nvidia-smi is used for the free/total query rather than torch so the
supervisor can call it without paying for a torch import. CUDA_VISIBLE_DEVICES
IS honoured here (2026-08-25): this account is restricted to cards 0-3 and the
rest belong to another user, so every helper below reports only the visible
cards, renumbered exactly as torch renumbers them. Without that, `fits`
measured the wrong card and `max_memory_map` returned physical indices that
device_map could not place -- `Device 2 is not available, available devices are
[0, 1]` -- and, worse, planned weights onto cards this account must not touch.

Pair the mask with CUDA_DEVICE_ORDER=PCI_BUS_ID so nvidia-smi ordering and CUDA
ordering agree; otherwise CUDA's default FASTEST_FIRST may permute them.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

GIB = 1024 ** 3

# Activation, KV-cache and fragmentation headroom to keep free on each card.
# The 78B Task B failure had weights fitting with roughly a gigabyte to spare
# per card and still OOM'd on every item, so this is deliberately generous.
DEFAULT_RESERVE_GIB = 5.0

# Multiplier on raw weight bytes for load-time overhead (buffers, the odd
# non-sharded tensor, dtype upcasts in the vision tower).
WEIGHT_OVERHEAD = 1.08


def set_alloc_conf() -> None:
    """Enable expandable segments unless the caller already chose a policy.

    Must run before torch is imported; every OOM traceback this project has
    produced recommends it, and it costs nothing when memory is plentiful.
    """

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def visible_devices() -> list[int] | None:
    """Physical indices CUDA_VISIBLE_DEVICES exposes, in torch's order.

    None means "unset -- every card is visible". An empty list means the mask
    is set but selects nothing, which is a real state (no GPUs) and must not be
    confused with None. UUID-form masks cannot be mapped to nvidia-smi indices
    here, so they are treated as unset rather than guessed at.
    """

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            return None          # GPU-<uuid> form; leave ordering alone
        out.append(int(part))
    return out


def gpu_memory() -> list[tuple[int, int]]:
    """(free, total) bytes per VISIBLE GPU, in torch's device order.

    When CUDA_VISIBLE_DEVICES is set, element i describes what torch calls
    cuda:i -- not physical card i. That is what makes the indices returned by
    `max_memory_map` placeable.
    """

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pairs = []
    for line in out.strip().splitlines():
        try:
            free, total = (int(x.strip()) for x in line.split(","))
        except ValueError:
            continue
        pairs.append((free * 1024 * 1024, total * 1024 * 1024))
    visible = visible_devices()
    if visible is None:
        return pairs
    return [pairs[i] for i in visible if 0 <= i < len(pairs)]


def usable_bytes(reserve_gib: float = DEFAULT_RESERVE_GIB) -> int:
    """Free VRAM across all cards, minus per-card headroom for activations."""

    reserve = int(reserve_gib * GIB)
    return sum(max(0, free - reserve) for free, _ in gpu_memory())


def _hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def estimate_weight_bytes(repo: str) -> int | None:
    """Total checkpoint bytes for `repo` from the local HF cache, or None.

    None means "not cached / cannot tell", which callers must treat as
    "proceed without a preflight verdict" rather than as zero.
    """

    folder = _hub_dir() / ("models--" + repo.replace("/", "--"))
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return None
    revs = [d for d in snapshots.iterdir() if d.is_dir()]
    if not revs:
        return None
    newest = max(revs, key=lambda d: d.stat().st_mtime)
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for f in newest.glob(pattern):
            try:
                total += f.stat().st_size          # follows the blob symlink
            except OSError:
                pass
    return total or None


def fits(repo: str, reserve_gib: float = DEFAULT_RESERVE_GIB,
         device: str = "auto") -> tuple[bool, str]:
    """Can `repo`'s weights load right now and still leave activation room?

    `device` matters and must not be ignored. A job pinned to `cuda:3` can only
    use card 3, so summing free memory across the whole machine would happily
    green-light a 30 GiB model onto a card with 21 GiB free -- which is the
    precise way a higher job cap turns into an OOM.

    Returns (ok, human-readable reason). Unknown size is treated as ok, since
    refusing to run an uncached model would be worse than trying.
    """

    weights = estimate_weight_bytes(repo)
    mem = gpu_memory()
    pinned = None
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            pinned = int(device.split(":", 1)[1])
        except ValueError:
            pinned = None

    if pinned is not None:
        if pinned >= len(mem):
            return True, f"device {device} not visible; skipping check"
        free, _ = mem[pinned]
        usable = max(0, free - int(reserve_gib * GIB))
        where = f"on {device}"
    else:
        usable = usable_bytes(reserve_gib)
        where = "across all cards"

    if weights is None:
        return True, f"size unknown (not cached); {usable/GIB:.0f} GiB usable {where}"
    need = int(weights * WEIGHT_OVERHEAD)
    ok = need <= usable
    verdict = "fits" if ok else "DOES NOT FIT"
    return ok, (f"{verdict}: needs ~{need/GIB:.0f} GiB "
                f"(weights {weights/GIB:.0f} GiB x {WEIGHT_OVERHEAD}), "
                f"{usable/GIB:.0f} GiB usable {where} "
                f"after {reserve_gib:.0f} GiB/card reserved")


def max_memory_map(reserve_gib: float = DEFAULT_RESERVE_GIB) -> dict[int, int]:
    """Per-device byte caps for device_map="auto".

    Without this, accelerate packs weights into every card up to its full
    capacity and generation has nowhere to put activations.
    """

    reserve = int(reserve_gib * GIB)
    return {i: max(0, free - reserve) for i, (free, _) in enumerate(gpu_memory())}


def summary(reserve_gib: float = DEFAULT_RESERVE_GIB) -> str:
    mem = gpu_memory()
    if not mem:
        return "no GPUs visible"
    parts = [f"{i}:{free/GIB:.0f}/{total/GIB:.0f}" for i, (free, total) in enumerate(mem)]
    return (f"free/total GiB per card {' '.join(parts)} -> "
            f"{usable_bytes(reserve_gib)/GIB:.0f} GiB usable")


def is_oom(exc: BaseException) -> bool:
    """True for both torch.cuda.OutOfMemoryError and the RuntimeError form.

    The 78B failure surfaced as `RuntimeError: CUDA error: out of memory`
    rather than the typed OutOfMemoryError, so matching on the type alone
    misses it.
    """

    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text
