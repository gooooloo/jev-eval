"""同一请求打 10 次，看返回是否完全一致。"""
import asyncio, collections
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul
from genset import build

async def main():
    items = build(20, seed=7)
    ch = [i for i in items if i["kind"] == "choice" and i["tier"] == 4][:3]
    nl = [i for i in items if i["kind"] == "noul" and i["tier"] == 4][:3]

    async with AsyncTypeSafeClient() as c:
        for it in ch + nl:
            q = (Choice(instructions=it["instructions"], criteria=it["criteria"])
                 if it["kind"] == "choice" else Noul(instructions=it["instructions"]))
            rs = await asyncio.gather(*(c.system_one(it["state"], {"q": q}) for _ in range(10)))
            if it["kind"] == "choice":
                vals = [(r.choices["q"].choice, round(r.choices["q"].confidence, 4),
                         tuple(sorted(r.choices["q"].probabilities.items()))) for r in rs]
            else:
                vals = [round(r.nouls["q"].noul, 6) for r in rs]
            uniq = collections.Counter(vals)
            tag = "完全确定" if len(uniq) == 1 else f"有波动({len(uniq)} 种)"
            print(f"[tier4 {it['kind']:<6}] {tag}")
            for v, n in uniq.most_common():
                shown = v if it["kind"] == "noul" else f"{v[0]} conf={v[1]} {dict(v[2])}"
                print(f"    {n:>2}/10  {shown}")

asyncio.run(main())
