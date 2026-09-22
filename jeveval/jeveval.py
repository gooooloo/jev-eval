#!/usr/bin/env python3
"""Jev 评测框架 —— 用你自己的标注数据测准确率、校准和阈值。

数据全程留在本机；除了调用 TypeSafe API（你本来就在调）外不外传。

用法：
    python jeveval.py run    --spec spec.json --data data.jsonl --out raw.jsonl
    python jeveval.py report --raw raw.jsonl [--target 0.99]
    python jeveval.py run    --spec ... --data ... --out raw.jsonl --repeat 5   # 稳定性
    python jeveval.py selftest                                                  # 验证框架本身
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

import metrics as M

# ------------------------------------------------------------------ 数据加载

def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "questions" not in spec:
        sys.exit(f"{path}: 缺少顶层 'questions' 字段")
    for key, q in spec["questions"].items():
        if q.get("type") not in ("choice", "noul", "score"):
            sys.exit(f"问题 '{key}': type 必须是 choice / noul / score")
        if q["type"] in ("choice", "score") and not q.get("criteria"):
            sys.exit(f"问题 '{key}': {q['type']} 必须提供 criteria")
    return spec


def load_data(path: Path) -> list[dict]:
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path}:{ln} JSON 解析失败: {e}")
        if "state" not in r or "gold" not in r:
            sys.exit(f"{path}:{ln} 每行必须有 'state' 和 'gold'")
        r.setdefault("id", str(ln))
        rows.append(r)
    if not rows:
        sys.exit(f"{path}: 没有数据")
    return rows


def build_question(q: dict):
    t = q["type"]
    if t == "noul":
        return Noul(instructions=q["instructions"])
    if t == "choice":
        return Choice(instructions=q["instructions"], criteria=q["criteria"])
    return Score(instructions=q["instructions"], criteria=q["criteria"])


# ------------------------------------------------------------------ 判分

def grade(qtype: str, answer, gold) -> dict:
    """把 SDK 的答案和标注对齐成 {pred, p, conf, ok}。

    p    = 模型给"它选中的那个答案"的概率（用于校准指标）
    conf = 用来当开关的分数
    """
    if qtype == "noul":
        p_yes = answer.noul
        pred = p_yes >= 0.5
        gold_b = bool(gold)
        return {"pred": pred, "p": max(p_yes, 1 - p_yes),
                "conf": max(p_yes, 1 - p_yes), "ok": pred == gold_b,
                "raw": {"noul": p_yes}}

    if qtype == "choice":
        pred = answer.choice
        ok = pred in gold if isinstance(gold, list) else pred == gold
        return {"pred": pred, "p": max(answer.probabilities.values()),
                "conf": answer.confidence, "ok": ok,
                "raw": {"choice": pred, "probabilities": answer.probabilities}}

    # score：gold 可以是档位标签，也可以是数值
    legend = answer.legend or {}
    idx = int(round(answer.score))
    pred_label = legend.get(idx, legend.get(str(idx), idx))
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        ok = abs(answer.score - float(gold)) <= 0.5
    else:
        ok = str(pred_label) == str(gold)
    probs = {int(k) if str(k).lstrip("-").isdigit() else k: v
             for k, v in (answer.probabilities or {}).items()}
    return {"pred": pred_label, "p": probs.get(idx, max(probs.values()) if probs else 0.0),
            "conf": answer.confidence, "ok": ok,
            "raw": {"score": answer.score, "legend": legend, "probabilities": answer.probabilities}}


# ------------------------------------------------------------------ 执行

async def run_one(client, sem, spec, row, model, retries=3):
    questions = row.get("questions") or spec["questions"]
    payload = {k: build_question(q) for k, q in questions.items() if k in row["gold"]}
    if not payload:
        return None

    state = row["state"]
    kwargs = {"model": model} if model else {}

    async with sem:
        t0 = time.perf_counter()
        last = None
        for attempt in range(retries):
            try:
                resp = await client.system_one(state, payload, **kwargs)
                break
            except Exception as e:                      # noqa: BLE001
                last = e
                if attempt == retries - 1:
                    return {"id": row["id"], "error": f"{type(e).__name__}: {e}"}
                await asyncio.sleep(1.0 * 2 ** attempt)
        ms = (time.perf_counter() - t0) * 1000

    buckets = {"choice": resp.choices, "noul": resp.nouls, "score": resp.scores}
    out = {"id": row["id"], "ms": ms, "model": resp.model,
           "input_tokens": resp.usage.input_tokens,
           "output_tokens": resp.usage.output_tokens, "results": {}}
    for k in payload:
        qtype = questions[k]["type"]
        out["results"][k] = {"qtype": qtype, "gold": row["gold"][k],
                             **grade(qtype, buckets[qtype][k], row["gold"][k])}
    return out


async def run(args) -> None:
    spec = load_spec(Path(args.spec))
    data = load_data(Path(args.data))
    if args.limit:
        data = data[: args.limit]
    sem = asyncio.Semaphore(args.concurrency)

    print(f"样本 {len(data)} 条 · 问题 {len(spec['questions'])} 个 · "
          f"并发 {args.concurrency} · 重复 {args.repeat} 轮", file=sys.stderr)

    all_runs = []
    async with AsyncTypeSafeClient() as client:
        for rep in range(args.repeat):
            t0 = time.perf_counter()
            res = await asyncio.gather(*(run_one(client, sem, spec, r, args.model) for r in data))
            res = [r for r in res if r]
            wall = time.perf_counter() - t0
            errs = sum("error" in r for r in res)
            print(f"  第 {rep+1} 轮：{len(res)-errs}/{len(res)} 成功，{wall:.1f}s，"
                  f"{len(res)/wall:.1f} req/s" + (f"，{errs} 个失败" if errs else ""),
                  file=sys.stderr)
            for r in res:
                r["run"] = rep
            all_runs.extend(res)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"原始结果已写入 {args.out}（可用 report 反复分析，不再调 API）", file=sys.stderr)

    report(argparse.Namespace(raw=args.out, target=args.target, bins=args.bins,
                              show_failures=args.show_failures))


# ------------------------------------------------------------------ 报告

def _fmt_pct(x): return f"{x:.1%}" if x == x else "n/a"


def report(args) -> None:
    raw = [json.loads(l) for l in Path(args.raw).read_text(encoding="utf-8").splitlines() if l.strip()]
    ok_rows = [r for r in raw if "error" not in r]
    errs = [r for r in raw if "error" in r]
    if not ok_rows:
        sys.exit("没有成功的结果")

    runs = sorted({r["run"] for r in ok_rows})
    base = [r for r in ok_rows if r["run"] == runs[0]]
    keys = sorted({k for r in base for k in r["results"]})

    print("=" * 74)
    print(f"样本 {len(base)} 条 · 问题 {len(keys)} 个 · 轮次 {len(runs)}"
          + (f" · 请求失败 {len(errs)}" if errs else ""))
    tok = sum(r["input_tokens"] for r in ok_rows)
    lat = sorted(r["ms"] for r in ok_rows)
    print(f"input tokens {tok:,}（约 ${tok*42/1e9:.4f}） · "
          f"延迟 p50 {lat[len(lat)//2]:.0f}ms p95 {lat[int(len(lat)*.95)]:.0f}ms")
    if errs:
        print(f"\n失败样本（前 3）：")
        for e in errs[:3]:
            print(f"  id={e['id']}  {e['error']}")

    for key in keys:
        rows = [r["results"][key] for r in base if key in r["results"]]
        if not rows:
            continue
        qtype = rows[0]["qtype"]
        print("\n" + "=" * 74)
        print(f"问题「{key}」  type={qtype}  n={len(rows)}")
        print(f"  准确率 {_fmt_pct(M.accuracy(rows))}   "
              f"ECE {M.ece(rows, nbins=args.bins):.4f}   "
              f"MCE {M.mce(rows, nbins=args.bins):.4f}   "
              f"Brier {M.brier(rows):.4f}")
        if M.ece(rows, nbins=args.bins) > 0.10:
            print("  ⚠ ECE > 0.10：概率不可信，别拿它当阈值")

        print("\n  可靠性曲线（模型说的概率 vs 实际准确率）")
        print(f"    {'区间':<14}{'n':>5}{'实际':>9}{'模型说':>9}{'偏差':>9}")
        for b in M.reliability(rows, nbins=args.bins):
            flag = "  虚高" if b["gap"] > 0.10 else ("  偏保守" if b["gap"] < -0.10 else "")
            print(f"    [{b['lo']:.2f},{b['hi']:.2f})  {b['n']:>5}{b['acc']:>9.1%}"
                  f"{b['mean_p']:>9.3f}{b['gap']:>+9.3f}{flag}")

        print("\n  阈值扫描（卡住 conf 以上自动处理）")
        print(f"    {'阈值':<8}{'自动化率':>10}{'自动部分准确':>14}{'漏网错误':>10}{'转人工':>9}")
        for s in M.sweep(rows):
            print(f"    >={s['threshold']:<6.2f}{s['coverage']:>9.1%}{s['accuracy']:>13.1%}"
                  f"{s['errors']:>10}{s['deferred']:>9}")

        rec = M.recommend(rows, target_acc=args.target)
        print(f"\n  ▶ 达到 {args.target:.0%} 准确率的最低阈值：", end="")
        if rec:
            print(f"conf >= {rec['threshold']:.2f}")
            print(f"    可自动处理 {rec['coverage']:.1%} 的量（{rec['n_auto']} 条），"
                  f"实测准确率 {rec['accuracy']:.1%}，转人工 {rec['deferred']} 条")
            near = M.near_threshold(rows, rec["threshold"])
            if near:
                print(f"    注意：有 {len(near)} 条 conf 落在阈值 ±0.05 内，"
                      f"这批会受概率抖动影响而反复横跳")
        else:
            print(f"不存在 —— 任何阈值都达不到 {args.target:.0%}")
            print("    说明：要么模型在这个任务上不够好，要么 state/criteria 需要改进")

        if qtype == "choice":
            conf = M.confusion(rows)
            wrong = {k: v for k, v in conf.items() if k[0] != k[1]}
            if wrong:
                print("\n  主要混淆（gold -> pred，次数）")
                for (g, p), n in sorted(wrong.items(), key=lambda kv: -kv[1])[:6]:
                    print(f"    {g} -> {p}   {n}")

        if args.show_failures:
            bad = sorted((r for r in rows if not r["ok"]), key=lambda r: -r["conf"])
            if bad:
                print(f"\n  判错的样本（按 conf 降序，前 {min(8,len(bad))}）")
                for r in bad[:8]:
                    print(f"    conf={r['conf']:.3f} p={r['p']:.3f} "
                          f"gold={r['gold']} pred={r['pred']}")

        if len(runs) > 1:
            per_run = []
            for rn in runs:
                rr = {r["id"]: r["results"].get(key) for r in ok_rows if r["run"] == rn}
                ids = sorted(rr)
                if all(rr[i] for i in ids):
                    per_run.append([rr[i] for i in ids])
            if len(per_run) > 1:
                st = M.stability(per_run)
                print(f"\n  稳定性（{st['runs']} 轮重复）")
                print(f"    概率标准差 平均 {st['mean_p_std']:.4f}  最大 {st['max_p_std']:.4f}")
                print(f"    结论翻转率 {st['argmax_flip_rate']:.1%}"
                      f"（{st['flipped_items']}/{st['n']} 条在不同轮次给出不同答案）")
                if st["argmax_flip_rate"] > 0.05:
                    print("    ⚠ 翻转率偏高：这批数据里有不少样本处于模型的决策边界上")


# ------------------------------------------------------------------ 自测

def selftest(args) -> None:
    """用合成数据端到端验证框架本身（会真实调用 API）。"""
    import random
    rng = random.Random(42)
    fruits = {"苹果": None, "香蕉": None, "西瓜": None}
    spec = {"questions": {
        "heaviest": {"type": "choice", "instructions": "记录里哪一项的总重量最大？",
                     "criteria": fruits},
        "over1000": {"type": "noul", "instructions": "记录里是否存在总重量超过 1000 克的项目？"},
    }}
    data = []
    for i in range(30):
        w = {f: rng.randint(50, 900) * rng.randint(1, 3) for f in fruits}
        state = "库存记录：\n" + "\n".join(f"{f}：共 {v} 克。" for f, v in w.items())
        data.append({"id": f"s{i}", "state": state,
                     "gold": {"heaviest": max(w, key=w.get),
                              "over1000": any(v > 1000 for v in w.values())}})

    d = Path(args.dir); d.mkdir(parents=True, exist_ok=True)
    (d / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(d / "data.jsonl", "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"合成样例已写入 {d}/spec.json 和 {d}/data.jsonl —— 照这个格式放你自己的数据",
          file=sys.stderr)

    asyncio.run(run(argparse.Namespace(
        spec=str(d / "spec.json"), data=str(d / "data.jsonl"), out=str(d / "raw.jsonl"),
        concurrency=10, repeat=args.repeat, model=None, limit=None,
        target=0.99, bins=10, show_failures=True)))


# ------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Jev 评测框架")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="调 API 跑评测")
    r.add_argument("--spec", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", default="raw.jsonl")
    r.add_argument("--concurrency", type=int, default=15)
    r.add_argument("--repeat", type=int, default=1, help="重复轮数，>1 时输出稳定性")
    r.add_argument("--model", default=None, help="如 jev-1.13.0，建议固定版本")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--target", type=float, default=0.99)
    r.add_argument("--bins", type=int, default=10)
    r.add_argument("--show-failures", action="store_true", default=True)
    r.set_defaults(func=lambda a: asyncio.run(run(a)))

    p = sub.add_parser("report", help="分析已有结果，不调 API")
    p.add_argument("--raw", required=True)
    p.add_argument("--target", type=float, default=0.99)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--show-failures", action="store_true", default=True)
    p.set_defaults(func=report)

    s = sub.add_parser("selftest", help="用合成数据验证框架")
    s.add_argument("--dir", default="./_selftest")
    s.add_argument("--repeat", type=int, default=2)
    s.set_defaults(func=selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
