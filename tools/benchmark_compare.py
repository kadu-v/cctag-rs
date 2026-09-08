"""Alternate baseline/current binaries without competing benchmark processes.

Build both detect examples with the same CLI instrumentation and release flags.
uv run --no-project python tools/benchmark_compare.py BASELINE CURRENT --output REPORT
"""
import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--modes", nargs="+", choices=["fast", "parity"], default=["fast", "parity"])
    p.add_argument("--sets", type=int, default=3)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    assert a.sets > 0 and a.iters > 0 and a.warmup >= 0 and all(t > 0 for t in a.threads)
    report = {"platform": platform.platform(), "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
              "baseline": str(a.baseline.resolve()), "current": str(a.current.resolve()),
              "sha256": {v: hashlib.sha256(getattr(a, v).read_bytes()).hexdigest() for v in ["baseline", "current"]},
              "iterations": a.iters, "warmup": a.warmup, "records": []}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for threads in a.threads:
        for mode in a.modes:
            for name in ["01", "02"]:
                image = f"CCTag/sample/{name}.png"
                for repeat in range(a.sets):
                    order = ["baseline", "current"] if repeat % 2 == 0 else ["current", "baseline"]
                    results = {}
                    for label in order:
                        cmd = [str(getattr(a, label).resolve()), image, "--threads", str(threads),
                               "--warmup", str(a.warmup), "--iters", str(a.iters), "--timings"]
                        if mode == "parity":
                            cmd.append("--parity")
                        if platform.system() == "Darwin":
                            cmd = ["/usr/bin/time", "-l", *cmd]
                        result = subprocess.run(cmd, text=True, capture_output=True)
                        if result.returncode:
                            raise RuntimeError(f"Command failed: {cmd}\n{result.stdout}\n{result.stderr}")
                        out = result.stdout
                        med = float(re.search(r"time: median ([\d.]+) ms", out)[1])
                        p90 = float(re.search(r"p90: ([\d.]+) ms", out)[1])
                        rss = re.search(r"(\d+)\s+maximum resident set size", result.stderr)
                        record = {"image": name, "threads": threads, "mode": mode, "set": repeat + 1,
                                  "version": label, "median_ms": med, "p90_ms": p90,
                                  "peak_rss_bytes": int(rss[1]) if rss else None,
                                  "stages_ms": {s: float(t) for s, t in re.findall(r"^(\S+)\s+([\d.]+) ms \(median\)", out, re.M)}}
                        report["records"].append(record)
                        results[label] = out.split("time: median", 1)[0]
                        a.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(f"{name} {mode} {threads}T set={repeat+1} {label}: {med:.3f}ms p90={p90:.3f}", flush=True)
                    if results["baseline"] != results["current"]:
                        raise RuntimeError(f"Detection output mismatch: {name} {mode} {threads}T")
    for name in ["01", "02"]:
        for mode in a.modes:
            for threads in a.threads:
                values = {v: statistics.median(r["median_ms"] for r in report["records"]
                          if (r["image"], r["mode"], r["threads"], r["version"]) == (name, mode, threads, v)) for v in ["baseline", "current"]}
                print(f"{name} {mode} {threads}T: {values['baseline']:.3f} -> {values['current']:.3f}ms ({(values['current']/values['baseline']-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
