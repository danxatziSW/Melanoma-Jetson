"""Converts deployment ONNX models to TensorRT engines (FP16 + FP32).

Usage: python3 scripts/deployedTensorrt/convert.py [--precision fp16 fp32]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.config import load_config

_ROOT        = Path(__file__).resolve().parents[3]
_DEPLOY_DIR  = _ROOT / "outputs" / "ablation_noseg" / "meta" / "deployment"
_TRT_DIR     = _DEPLOY_DIR / "tensorrt"
_MODELS      = ["resnet50_none_sens", "medfusionnet_none_sens"]
_WORKSPACE   = 2   # GB

_R="\033[0m"; _B="\033[1m"; _G="\033[32m"; _Y="\033[33m"
_RE="\033[31m"; _C="\033[36m"; _D="\033[2m"; _W="\033[97m"; _BL="\033[34m"

def _c(t, *codes): return "".join(codes) + t + _R
_SEP  = _c("═"*64, _BL)
_SEP2 = _c("─"*64, _D)


def convert_onnx_to_trt(onnx_path: Path, engine_path: Path,
                        fp16: bool, workspace_gb: int) -> dict:
    import tensorrt as trt

    t0     = time.perf_counter()
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_gb * (1 << 30))
    if fp16:
        if not builder.platform_has_fast_fp16:
            print(f"    {_c('WARN: platform has no fast FP16 — building anyway', _Y)}")
        config.set_flag(trt.BuilderFlag.FP16)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = builder.build_serialized_network(network, config)
    if serialised is None:
        raise RuntimeError("TRT build_serialized_network returned None")

    engine_path.write_bytes(serialised)
    elapsed = time.perf_counter() - t0
    size_mb = engine_path.stat().st_size / 1e6
    return {"elapsed": elapsed, "size_mb": size_mb}


def _build_one(onnx_path: str, engine_path: str, fp16: bool, workspace: int):
    """Entry point for subprocess build — isolates CUDA memory per model."""
    import json
    result = convert_onnx_to_trt(Path(onnx_path), Path(engine_path), fp16, workspace)
    print(json.dumps(result))


def main():
    import json, subprocess

    parser = argparse.ArgumentParser(
        description="Convert deployment ONNX models to TensorRT engines.")
    parser.add_argument("--precision", nargs="+", default=["fp16", "fp32"],
                        choices=["fp16", "fp32"])
    parser.add_argument("--workspace", type=int, default=_WORKSPACE,
                        help="TRT workspace GB (default: 2)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--_build-one", nargs=4, metavar=("ONNX","ENGINE","FP16","WS"),
                        help=argparse.SUPPRESS)  # internal subprocess flag
    args = parser.parse_args()

    # subprocess worker path
    if args._build_one:
        onnx_p, engine_p, fp16_s, ws_s = args._build_one
        _build_one(onnx_p, engine_p, fp16_s == "1", int(ws_s))
        return

    print()
    print(_SEP)
    print(_c("  TRT Conversion — Deployment Models", _B, _W))
    print(_SEP2)
    print(f"  Models     : {_c(', '.join(_MODELS), _C)}")
    print(f"  Precisions : {_c(', '.join(p.upper() for p in args.precision), _B)}")
    print(f"  Source     : {_c(str(_DEPLOY_DIR.relative_to(_ROOT)), _D)}")
    print(f"  Output     : {_c(str(_TRT_DIR.relative_to(_ROOT)), _D)}")
    print(_c("  Note: each build runs in a fresh subprocess to avoid CUDA OOM", _D))
    print(_SEP)
    print()

    results = []
    for model_name in _MODELS:
        onnx_path = _DEPLOY_DIR / f"{model_name}.onnx"
        if not onnx_path.exists():
            print(f"  {_c('✗', _RE)} {model_name}.onnx not found — skipped")
            continue

        sz_onnx = onnx_path.stat().st_size / 1e6
        print(f"  {_c(model_name, _B)}  ({sz_onnx:.1f} MB ONNX)")

        for prec in args.precision:
            fp16        = (prec == "fp16")
            engine_path = _TRT_DIR / f"{model_name}_{prec}.engine"

            if args.skip_existing and engine_path.exists():
                sz = engine_path.stat().st_size / 1e6
                print(f"    {_c('⟳', _Y)} {prec.upper()}  already exists ({sz:.1f} MB) — skipped")
                results.append({"model": model_name, "prec": prec,
                                 "status": "skip", "size_mb": sz})
                continue

            print(f"    {_c('⠋', _C)} {prec.upper()}  building ...", end="", flush=True)
            try:
                # run in subprocess so CUDA context is fresh (avoids OOM from prior builds)
                proc = subprocess.run(
                    [sys.executable, __file__,
                     "--_build-one", str(onnx_path), str(engine_path),
                     "1" if fp16 else "0", str(args.workspace)],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip().split("\n")[-1])
                # last line of stdout is the JSON result
                r = json.loads(proc.stdout.strip().split("\n")[-1])
                sz_str = f'{r["size_mb"]:.1f} MB'
                el_str = f'{r["elapsed"]:.0f}s'
                print(f"\r    {_c('✓', _G)} {prec.upper()}  "
                      f"{_c(sz_str, _B)}  {_c(el_str, _D)}")
                results.append({"model": model_name, "prec": prec,
                                 "status": "ok", **r})
            except Exception as e:
                print(f"\r    {_c('✗', _RE)} {prec.upper()}  {e}")
                results.append({"model": model_name, "prec": prec,
                                 "status": "error", "error": str(e)})
        print()

    ok  = [r for r in results if r["status"] == "ok"]
    skp = [r for r in results if r["status"] == "skip"]
    err = [r for r in results if r["status"] == "error"]
    print(_SEP)
    print(f"  {_c(f'✓ Built   : {len(ok)}', _G)}   "
          f"{_c(f'⟳ Skipped : {len(skp)}', _Y)}   "
          f"{_c(f'✗ Errors  : {len(err)}', _RE)}")
    if ok:
        print()
        for r in ok:
            print(f"  {_c('✓', _G)}  {r['model']}_{r['prec']}.engine  "
                  f"({r['size_mb']:.1f} MB,  {r['elapsed']:.0f}s)")
    if err:
        print()
        for r in err:
            print(f"  {_c('✗', _RE)}  {r['model']}_{r['prec']}  — {r.get('error','')}")
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
