"""Detect this machine's hardware + software stack AND run a full
compatibility audit against faceswap_pro v2's requirements.

Writes (next to this script):
    PROJECT_ENV.md       human-readable hardware snapshot + compat report
    PROJECT_ENV.json     machine-readable full dump

Checks:
    BLOCKER  Python 3.10.x, PyTorch + CUDA, ffmpeg on PATH,
             GPU detected, VRAM >= 8 GB, every package in
             requirements.txt installed at the right version
    WARN     VRAM 8-16 GB (fine-tune will OOM), missing optional
             features (SAM2 pkg, inswapper_128 ONNX, GFPGAN weights)
    INFO     cuDNN version, FP8-capable compute capability
    PASS     everything that's working

Each failing check carries a `fix_hint` with an exact remediation
command. Exit code 0 if no blockers, 1 if any blocker.

    python detect_system.py
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# tiny shell helper that NEVER raises
# ---------------------------------------------------------------------
def _run(cmd, timeout=15):
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="ignore",
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except FileNotFoundError:
        return -1, "", "command not found"
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as exc:
        return -1, "", str(exc)


# ---------------------------------------------------------------------
# Compatibility check primitive
# ---------------------------------------------------------------------
LEVEL_BLOCKER = "BLOCKER"
LEVEL_WARN    = "WARN"
LEVEL_INFO    = "INFO"
LEVEL_PASS    = "PASS"


@dataclass
class CompatCheck:
    level: str
    name: str
    status: str
    detail: str = ""
    fix_hint: str = ""


# ---------------------------------------------------------------------
# OS / Python / CPU / RAM
# ---------------------------------------------------------------------
def detect_os():
    return {
        "system":   platform.system(),
        "release":  platform.release(),
        "version":  platform.version(),
        "platform": platform.platform(),
        "machine":  platform.machine(),
        "node":     platform.node(),
    }


def detect_python():
    return {
        "version":        sys.version.split()[0],
        "version_full":   sys.version,
        "executable":     sys.executable,
        "implementation": platform.python_implementation(),
    }


def detect_cpu():
    info = {
        "name":          platform.processor() or "unknown",
        "cores_logical": os.cpu_count() or 0,
    }
    if platform.system() == "Windows":
        rc, out, _ = _run([
            "wmic", "cpu", "get",
            "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed",
            "/value",
        ])
        if rc == 0:
            for ln in out.splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if not v:
                        continue
                    if k == "Name":
                        info["name"] = v
                    elif k == "NumberOfCores":
                        info["cores_physical"] = int(v)
                    elif k == "NumberOfLogicalProcessors":
                        info["cores_logical"] = int(v)
                    elif k == "MaxClockSpeed":
                        info["max_clock_mhz"] = int(v)
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for ln in f:
                    if ln.startswith("model name"):
                        info["name"] = ln.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return info


def detect_ram():
    info = {}
    if platform.system() == "Windows":
        rc, out, _ = _run([
            "wmic", "ComputerSystem", "get",
            "TotalPhysicalMemory", "/value",
        ])
        if rc == 0:
            for ln in out.splitlines():
                if "TotalPhysicalMemory=" in ln:
                    try:
                        b = int(ln.split("=", 1)[1].strip())
                        info["total_gb"]    = round(b / 1024**3, 1)
                        info["total_bytes"] = b
                    except Exception:
                        pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for ln in f:
                    if ln.startswith("MemTotal:"):
                        kb = int(ln.split()[1])
                        info["total_gb"]    = round(kb / 1024**2, 1)
                        info["total_bytes"] = kb * 1024
                        break
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------
# GPU detection: nvidia-smi -> wmic fallback (Windows)
# ---------------------------------------------------------------------
def detect_gpu_via_nvidia_smi():
    rc, out, err = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,"
        "compute_cap,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return [], (err or "nvidia-smi not found").strip()
    gpus = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7:
            continue
        idx, name, vram_mib, driver, cc, uuid, pci = parts
        try:
            vram_gb = round(int(vram_mib) / 1024, 1)
        except Exception:
            vram_gb = vram_mib
        gpus.append({
            "index":              int(idx),
            "name":               name,
            "vram_gb":            vram_gb,
            "driver_version":     driver,
            "compute_capability": cc,
            "uuid":               uuid,
            "pci_bus_id":         pci,
        })
    return gpus, ""


def detect_gpu_via_wmic():
    if platform.system() != "Windows":
        return []
    rc, out, _ = _run([
        "wmic", "path", "win32_VideoController", "get",
        "Name,AdapterRAM,DriverVersion,VideoProcessor", "/value",
    ])
    if rc != 0:
        return []
    current, gpus = {}, []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            if current.get("name"):
                gpus.append(current)
                current = {}
            continue
        if "=" in ln:
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip()
            if not v:
                continue
            if k == "Name":
                current["name"] = v
            elif k == "AdapterRAM":
                try:
                    current["vram_gb_approx"] = round(int(v) / 1024**3, 1)
                except Exception:
                    pass
            elif k == "DriverVersion":
                current["driver_version"] = v
            elif k == "VideoProcessor":
                current["video_processor"] = v
    if current.get("name"):
        gpus.append(current)
    # Drop integrated-only entries when a discrete GPU is present
    return gpus


# ---------------------------------------------------------------------
# CUDA / cuDNN / PyTorch
# ---------------------------------------------------------------------
def detect_cuda_via_nvcc():
    rc, out, _ = _run(["nvcc", "--version"])
    if rc != 0:
        return {"installed": False}
    m = re.search(r"release\s+(\d+\.\d+)", out)
    return {
        "installed": True,
        "version":   m.group(1) if m else "unknown",
        "raw":       out.strip().splitlines()[-1] if out else "",
    }


def detect_torch():
    info = {"installed": False}
    try:
        import torch
        info["installed"]     = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_version"]  = torch.version.cuda
            try:
                info["cudnn_version"] = torch.backends.cudnn.version()
            except Exception:
                info["cudnn_version"] = None
            info["device_count"]  = torch.cuda.device_count()
            info["devices"]       = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index":              i,
                    "name":               p.name,
                    "vram_gb":            round(p.total_memory / 1024**3, 2),
                    "compute_capability": f"{p.major}.{p.minor}",
                    "sm_count":           p.multi_processor_count,
                })
    except ImportError:
        info["torch_version"] = "not installed"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def detect_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        return {"installed": False}
    rc, out, _ = _run([exe, "-version"])
    first = (out.splitlines() or [""])[0] if rc == 0 else ""
    return {"installed": True, "path": exe, "version_line": first}


# ---------------------------------------------------------------------
# requirements.txt parser + spec matcher (stdlib only)
# ---------------------------------------------------------------------
_PKG_IMPORT_MAP = {
    "opencv-python":          "cv2",
    "opencv-contrib-python":  "cv2",
    "huggingface-hub":        "huggingface_hub",
    "PyYAML":                 "yaml",
    "Pillow":                 "PIL",
    "onnxruntime-gpu":        "onnxruntime",
    "antlr4-python3-runtime": "antlr4",
    "hydra-core":             "hydra",
    "uvicorn[standard]":      "uvicorn",
}


def _parse_requirement_line(ln):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        return None
    if "#" in ln:
        ln = ln.split("#", 1)[0].strip()
    m = re.match(r"^([A-Za-z0-9_.\-]+(?:\[[^\]]+\])?)(.*)$", ln)
    if not m:
        return None
    return {"name": m.group(1).strip(), "spec": m.group(2).strip()}


def _get_installed_version(name):
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    for candidate in (name, name.replace("_", "-")):
        try:
            return version(candidate)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _parse_version(v):
    parts = re.findall(r"\d+", v or "")
    return tuple(int(p) for p in parts) if parts else (0,)


def _spec_satisfied(installed, spec):
    if not spec:
        return True, ""
    iv = _parse_version(installed)
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"^(==|>=|<=|>|<|~=|!=)\s*([0-9A-Za-z.\-+]+)$", raw)
        if not m:
            continue
        op, ref = m.group(1), m.group(2)
        rv = _parse_version(ref)
        L = max(len(iv), len(rv))
        a = iv + (0,) * (L - len(iv))
        b = rv + (0,) * (L - len(rv))
        if op == "==" and a != b:
            return False, f"need =={ref}, got {installed}"
        if op == "!=" and a == b:
            return False, f"need !={ref}, got {installed}"
        if op == ">=" and a <  b:
            return False, f"need >={ref}, got {installed}"
        if op == "<=" and a >  b:
            return False, f"need <={ref}, got {installed}"
        if op == ">"  and a <= b:
            return False, f"need >{ref}, got {installed}"
        if op == "<"  and a >= b:
            return False, f"need <{ref}, got {installed}"
        if op == "~=" and a <  b:
            return False, f"need ~={ref}, got {installed}"
    return True, ""


def parse_requirements_txt():
    here = Path(__file__).resolve().parent
    rp = here / "requirements.txt"
    if not rp.is_file():
        return []
    out = []
    for ln in rp.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_requirement_line(ln)
        if parsed:
            out.append(parsed)
    return out


# ---------------------------------------------------------------------
# Compatibility checks
# ---------------------------------------------------------------------
def check_python(env):
    ver = env["python"]["version"]
    parts = ver.split(".")
    major = int(parts[0]) if parts else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    if (major, minor) == (3, 10):
        return CompatCheck(LEVEL_PASS, "python_version",
            f"Python {ver} OK")
    return CompatCheck(LEVEL_BLOCKER, "python_version",
        f"Python {ver} -- need 3.10.x",
        detail="LatentSync pins Python 3.10.",
        fix_hint="Install Python 3.10 and recreate your venv "
                 "(`py -3.10 -m venv venv_new`).")


def check_torch(env):
    out = []
    t = env["torch"]
    if not t.get("installed"):
        out.append(CompatCheck(LEVEL_BLOCKER, "torch_installed",
            "PyTorch NOT installed",
            fix_hint="pip install torch==2.4.1 torchvision==0.19.1 "
                     "torchaudio==2.4.1 --index-url "
                     "https://download.pytorch.org/whl/cu121"))
        return out
    out.append(CompatCheck(LEVEL_PASS, "torch_installed",
        f"PyTorch {t.get('torch_version', '?')} installed"))
    if not t.get("cuda_available"):
        out.append(CompatCheck(LEVEL_BLOCKER, "torch_cuda",
            "torch.cuda.is_available() = False",
            detail="Inference will fall back to CPU -- unusable.",
            fix_hint="Reinstall PyTorch with a CUDA wheel: "
                     "pip install torch==2.4.1 --index-url "
                     "https://download.pytorch.org/whl/cu121"))
    else:
        out.append(CompatCheck(LEVEL_PASS, "torch_cuda",
            f"CUDA {t.get('cuda_version')} available, "
            f"cuDNN {t.get('cudnn_version')}"))
    return out


def check_gpu(env):
    out = []
    gpus = env.get("gpus") or []
    if not gpus:
        out.append(CompatCheck(LEVEL_BLOCKER, "gpu_detected",
            "No GPU detected",
            detail="nvidia-smi returned no device.",
            fix_hint="Install an NVIDIA GPU + driver, or use a "
                     "cloud GPU instance."))
        return out
    g = gpus[0]
    vram = g.get("vram_gb") or g.get("vram_gb_approx") or 0
    try:
        vram = float(vram)
    except Exception:
        vram = 0.0
    out.append(CompatCheck(LEVEL_PASS, "gpu_detected",
        f"GPU: {g.get('name', '?')} ({vram} GB VRAM, "
        f"CC {g.get('compute_capability', '?')})"))
    if vram < 8:
        out.append(CompatCheck(LEVEL_BLOCKER, "gpu_vram",
            f"{vram} GB VRAM -- below 8 GB minimum",
            detail="LatentSync 512 will OOM.",
            fix_hint="A 12+ GB card is the realistic floor."))
    elif vram < 16:
        out.append(CompatCheck(LEVEL_WARN, "gpu_vram",
            f"{vram} GB VRAM -- inference OK, fine-tune will OOM",
            detail="Per-clip fine-tune needs ~16-24 GB."))
    else:
        out.append(CompatCheck(LEVEL_PASS, "gpu_vram",
            f"{vram} GB VRAM -- inference + fine-tune both fit"))
    cc = str(g.get("compute_capability", ""))
    try:
        if float(cc) >= 8.9:
            out.append(CompatCheck(LEVEL_INFO, "fp8_support",
                f"Compute capability {cc} has native FP8 tensor cores"))
    except Exception:
        pass
    return out


def check_ffmpeg(env):
    if env["ffmpeg"].get("installed"):
        return CompatCheck(LEVEL_PASS, "ffmpeg",
            f"ffmpeg on PATH at {env['ffmpeg']['path']}")
    return CompatCheck(LEVEL_BLOCKER, "ffmpeg",
        "ffmpeg NOT on PATH",
        detail="Every render stage shells out to ffmpeg.",
        fix_hint="Windows: install from https://www.gyan.dev/ffmpeg/"
                 "builds/ and add bin/ to PATH. "
                 "Linux: apt install ffmpeg. macOS: brew install ffmpeg.")


def check_requirements_packages():
    out = []
    reqs = parse_requirements_txt()
    if not reqs:
        out.append(CompatCheck(LEVEL_WARN, "requirements_file",
            "requirements.txt not found -- skipping pip checks"))
        return out
    for req in reqs:
        name, spec = req["name"], req["spec"]
        installed = _get_installed_version(name)
        if installed is None:
            mapped = _PKG_IMPORT_MAP.get(name)
            if mapped:
                installed = _get_installed_version(mapped)
        if installed is None:
            out.append(CompatCheck(LEVEL_BLOCKER, f"pkg:{name}",
                f"{name}{spec} NOT installed",
                fix_hint=f'pip install "{name}{spec}"'))
            continue
        ok, why = _spec_satisfied(installed, spec)
        if ok:
            out.append(CompatCheck(LEVEL_PASS, f"pkg:{name}",
                f"{name} {installed} satisfies {spec or '(any)'}"))
        else:
            out.append(CompatCheck(LEVEL_BLOCKER, f"pkg:{name}",
                f"{name} {installed} -- {why}",
                fix_hint=f'pip install --upgrade "{name}{spec}"'))
    return out


def _safe_is_file(p):
    """Path.is_file() raises OSError on broken symlinks / stale
    junctions / offline network drives. Return False instead so a
    bad path never crashes the whole audit."""
    try:
        return p.is_file()
    except OSError:
        return False


def check_optional_features():
    out = []
    here = Path(__file__).resolve().parent
    try:
        import sam2  # noqa: F401
        out.append(CompatCheck(LEVEL_PASS, "sam2_pkg",
            "SAM 2 importable -- mask-out feature available"))
    except Exception:
        out.append(CompatCheck(LEVEL_WARN, "sam2_pkg",
            "SAM 2 not importable -- mask-out feature disabled",
            detail="Lip-Sync tab's mask-out accordion won't work.",
            fix_hint="git clone https://github.com/facebookresearch/"
                     "sam2.git && pip install -e ./sam2"))
    inswap_paths = [
        here / "checkpoints" / "inswapper_128.onnx",
        here / "models" / "face_swap" / "inswapper_128.onnx",
    ]
    found = next((p for p in inswap_paths if _safe_is_file(p)), None)
    if found:
        out.append(CompatCheck(LEVEL_PASS, "inswapper_128",
            f"inswapper_128.onnx found at {found}"))
    else:
        out.append(CompatCheck(LEVEL_WARN, "inswapper_128",
            "inswapper_128.onnx NOT placed -- face-swap tab will fail",
            detail="Insightface-licensed weight; not auto-downloaded.",
            fix_hint="Acquire inswapper_128.onnx and place at "
                     "checkpoints/inswapper_128.onnx."))
    gfp_paths = [
        here / "models" / "GFPGANv1.4.pth",
        here / "gfpgan" / "weights" / "GFPGANv1.4.pth",
    ]
    if any(_safe_is_file(p) for p in gfp_paths):
        out.append(CompatCheck(LEVEL_PASS, "gfpgan_weights",
            "GFPGAN v1.4 weights found"))
    else:
        out.append(CompatCheck(LEVEL_INFO, "gfpgan_weights",
            "GFPGAN v1.4 weights not present yet "
            "(auto-downloaded on first Enhance faces render)"))
    sam2_ckpt = here / "checkpoints" / "sam2" / "sam2.1_hiera_base_plus.pt"
    if _safe_is_file(sam2_ckpt):
        out.append(CompatCheck(LEVEL_PASS, "sam2_weights",
            f"SAM 2 weights at {sam2_ckpt.name}"))
    else:
        out.append(CompatCheck(LEVEL_INFO, "sam2_weights",
            "SAM 2 weights not present yet "
            "(auto-downloaded on first mask-out render)"))
    return out


def run_all_checks(env):
    checks = []
    checks.append(check_python(env))
    checks.extend(check_torch(env))
    checks.extend(check_gpu(env))
    checks.append(check_ffmpeg(env))
    checks.extend(check_requirements_packages())
    checks.extend(check_optional_features())
    return checks


def _level_order(c):
    return {LEVEL_BLOCKER: 0, LEVEL_WARN: 1,
            LEVEL_INFO: 2, LEVEL_PASS: 3}.get(c.level, 4)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    print("=" * 70)
    print(f"detect_system.py -- writing snapshots into {here}")
    print("=" * 70)

    env = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "os":           detect_os(),
        "python":       detect_python(),
        "cpu":          detect_cpu(),
        "ram":          detect_ram(),
        "ffmpeg":       detect_ffmpeg(),
        "cuda_nvcc":    detect_cuda_via_nvcc(),
        "torch":        detect_torch(),
    }

    gpus_smi, smi_err = detect_gpu_via_nvidia_smi()
    if gpus_smi:
        env["gpu_source"] = "nvidia-smi"
        env["gpus"]       = gpus_smi
    else:
        gpus_wmic = detect_gpu_via_wmic()
        if gpus_wmic:
            env["gpu_source"] = "wmic"
            env["gpus"]       = gpus_wmic
            env["gpu_detection_note_smi"] = smi_err
        else:
            env["gpu_source"] = "none"
            env["gpus"]       = []
            env["gpu_detection_error"] = smi_err

    # ---- compatibility audit ----
    checks = run_all_checks(env)
    env["compatibility"] = [asdict(c) for c in checks]
    n_blocker = sum(1 for c in checks if c.level == LEVEL_BLOCKER)
    n_warn    = sum(1 for c in checks if c.level == LEVEL_WARN)
    n_info    = sum(1 for c in checks if c.level == LEVEL_INFO)
    n_pass    = sum(1 for c in checks if c.level == LEVEL_PASS)
    env["compatibility_summary"] = {
        "blockers": n_blocker, "warnings": n_warn,
        "info":     n_info,    "passes":   n_pass,
    }

    # ---- JSON ----
    json_p = here / "PROJECT_ENV.json"
    json_p.write_text(
        json.dumps(env, indent=2, ensure_ascii=False),
        encoding="utf-8", newline="\n",
    )

    # ---- Markdown ----
    md = []
    md.append("# Project environment")
    md.append("")
    md.append(f"_Generated by `detect_system.py` at {env['generated_at']}._")
    md.append("")
    md.append("_Future sessions: **read this file first** before assuming "
              "anything about the host. Re-run `python detect_system.py` "
              "to refresh._")
    md.append("")
    md.append("## GPU")
    if env["gpus"]:
        for g in env["gpus"]:
            md.append(f"- **{g.get('name', 'unknown')}**")
            for k in ("vram_gb", "vram_gb_approx", "driver_version",
                      "compute_capability", "pci_bus_id",
                      "video_processor"):
                v = g.get(k)
                if v not in (None, "", "unknown"):
                    md.append(f"  - {k}: `{v}`")
        md.append(f"- detection source: `{env['gpu_source']}`")
    else:
        md.append(
            f"- **no GPU detected** "
            f"(error: `{env.get('gpu_detection_error', 'unknown')}`)"
        )
    md.append("")

    md.append("## CUDA + PyTorch")
    nvcc = env["cuda_nvcc"]
    if nvcc.get("installed"):
        md.append(f"- nvcc: `{nvcc.get('version')}` "
                  f"-- `{nvcc.get('raw', '')}`")
    else:
        md.append("- nvcc: not installed / not on PATH")
    t = env["torch"]
    if t.get("installed"):
        md.append(f"- torch: `{t.get('torch_version')}`")
        md.append(f"- torch.cuda.is_available: `{t.get('cuda_available')}`")
        if t.get("cuda_available"):
            md.append(f"- torch CUDA: `{t.get('cuda_version')}`, "
                      f"cuDNN: `{t.get('cudnn_version')}`")
            for d in t.get("devices", []):
                cc = d["compute_capability"].replace(".", "")
                md.append(
                    f"- torch device {d['index']}: `{d['name']}` "
                    f"({d['vram_gb']} GB VRAM, sm_{cc}, "
                    f"{d['sm_count']} SMs)"
                )
    else:
        md.append(f"- torch: `{t.get('torch_version', 'not installed')}`")
    md.append("")

    md.append("## CPU + RAM")
    cpu = env["cpu"]
    md.append(
        f"- CPU: `{cpu.get('name', 'unknown')}` "
        f"({cpu.get('cores_physical', '?')} physical / "
        f"{cpu.get('cores_logical', '?')} logical cores"
        + (f", {cpu['max_clock_mhz']} MHz max"
           if 'max_clock_mhz' in cpu else "")
        + ")"
    )
    md.append(f"- RAM: `{env['ram'].get('total_gb', 'unknown')} GB`")
    md.append("")

    md.append("## OS + Python")
    md.append(f"- OS: `{env['os']['platform']}`")
    md.append(f"- Hostname: `{env['os']['node']}`")
    md.append(f"- Python: `{env['python']['version']}` "
              f"at `{env['python']['executable']}`")
    md.append("")

    md.append("## ffmpeg")
    f = env["ffmpeg"]
    if f.get("installed"):
        md.append(f"- path: `{f['path']}`")
        if f.get("version_line"):
            md.append(f"- version: `{f['version_line']}`")
    else:
        md.append("- not on PATH")
    md.append("")

    md.append("## Compatibility")
    md.append("")
    md.append(f"**Summary**: {n_blocker} BLOCKER, {n_warn} WARN, "
              f"{n_info} INFO, {n_pass} PASS")
    md.append("")
    if n_blocker == 0:
        md.append("All hard requirements satisfied. Safe to launch.")
    else:
        md.append(f"**{n_blocker} blocker(s) below must be fixed "
                  "before the app will run.**")
    md.append("")
    sorted_checks = sorted(checks, key=_level_order)
    for level in (LEVEL_BLOCKER, LEVEL_WARN, LEVEL_INFO, LEVEL_PASS):
        bucket = [c for c in sorted_checks if c.level == level]
        if not bucket:
            continue
        md.append(f"### {level} ({len(bucket)})")
        md.append("")
        for c in bucket:
            md.append(f"- **{c.name}** -- {c.status}")
            if c.detail:
                md.append(f"  - {c.detail}")
            if c.fix_hint:
                md.append(f"  - **fix:** `{c.fix_hint}`")
        md.append("")
    md.append("---")
    md.append("")
    md.append("## Full machine-readable dump")
    md.append("See `PROJECT_ENV.json` next to this file.")
    md.append("")

    md_p = here / "PROJECT_ENV.md"
    md_p.write_text("\n".join(md), encoding="utf-8", newline="\n")

    # ---- console summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if env["gpus"]:
        for g in env["gpus"]:
            print(
                f"GPU: {g.get('name', '?')} | "
                f"VRAM: {g.get('vram_gb', g.get('vram_gb_approx', '?'))} GB | "
                f"Driver: {g.get('driver_version', '?')} | "
                f"CC: {g.get('compute_capability', '?')}"
            )
    else:
        print(f"GPU: NONE DETECTED "
              f"({env.get('gpu_detection_error', '')})")
    print(f"CPU: {cpu.get('name', '?')} "
          f"({cpu.get('cores_physical', '?')}c / "
          f"{cpu.get('cores_logical', '?')}t)")
    print(f"RAM: {env['ram'].get('total_gb', '?')} GB")
    print(f"OS:  {env['os']['platform']}")
    print(f"Python: {env['python']['version']}")
    print(f"torch: {t.get('torch_version', '?')}, "
          f"cuda_avail: {t.get('cuda_available', False)}")
    if nvcc.get("installed"):
        print(f"nvcc: {nvcc.get('version')}")

    print()
    print("=" * 70)
    print(f"COMPATIBILITY: {n_blocker} BLOCKER | {n_warn} WARN | "
          f"{n_info} INFO | {n_pass} PASS")
    print("=" * 70)
    for level, marker in ((LEVEL_BLOCKER, "[X]"), (LEVEL_WARN, "[!]")):
        for c in [x for x in checks if x.level == level]:
            print(f"{marker} {c.name}: {c.status}")
            if c.fix_hint:
                print(f"    fix: {c.fix_hint}")
    print()
    print("Wrote:")
    print(f"  {md_p}")
    print(f"  {json_p}")
    if n_blocker > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
