"""最小 Jev 冒烟测试：一次调用里混用 Noul / Choice / Score 三种 primitive。"""

import os
import time

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

STATE = """\
主题：又被扣了两次钱
我上个月已经取消了订阅，但这个月还是被扣了 99 元，而且扣了两次。
我已经发过三封邮件没人回，请今天之内给我退款，否则我就去投诉了。
"""

def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("请先设置 TYPESAFE_API_KEY")

    client = TypeSafeClient()
    t0 = time.perf_counter()
    result = client.system_one(
        STATE,
        {
            "billing": Noul(instructions="这条工单是关于计费或退款问题吗？"),
            "needs_human": Noul(instructions="这条工单需要人工客服介入吗？"),
            "tone": Choice(
                instructions="用户的语气是怎样的？",
                criteria={"calm": None, "frustrated": None, "angry": None},
            ),
            "team": Choice(
                instructions="应该路由给哪个团队？",
                criteria={
                    "billing": "扣款、退款、发票",
                    "technical": "产品无法使用、报错",
                    "sales": "购买咨询、升级套餐",
                },
            ),
            "urgency": Score(
                instructions="这条工单的紧急程度？",
                criteria=["low", "medium", "high", "critical"],
            ),
        },
    )
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"耗时 {elapsed:.0f} ms\n")
    for key, ans in result.nouls.items():
        print(f"[noul]   {key:12} = {ans.noul:.4f}")
    for key, ans in result.choices.items():
        print(f"[choice] {key:12} = {ans.choice}  (confidence {ans.confidence:.4f})")
        print(f"         {ans.probabilities}")
    for key, ans in result.scores.items():
        print(f"[score]  {key:12} = {ans.score}  (confidence {ans.confidence:.4f})")
        print(f"         legend={ans.legend}")
        print(f"         {ans.probabilities}")

    print(f"\nusage: {result.usage}")


if __name__ == "__main__":
    main()
