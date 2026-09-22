# jev-eval

在**你自己的数据**上评测 [TypeSafe Jev](https://typesafe.ai) 的准确率、概率校准和可用阈值。

Jev 是 TypeSafe 的 System One 模型：不生成文本，一次前向传播返回带类型的判断
（Choice / Noul / Score）和概率分布。它的核心卖点是**校准**——自报的 confidence
应当反映真实准确率。这个仓库就是用来验证这件事在你的任务上成不成立，并算出
**你应该把自动化阈值卡在哪里**。

## 快速开始

```bash
git clone <this repo> && cd jev-eval
python3 -m venv .venv && . .venv/bin/activate     # 需要 Python >= 3.10
pip install -r requirements.txt
export TYPESAFE_API_KEY=...                        # console.typesafe.ai/keys

cd jeveval
python metrics.py                                  # 指标实现自检，不联网
python jeveval.py selftest                         # 端到端验证，约 $0.001
```

`selftest` 跑通就说明环境没问题，然后照 `jeveval/examples/` 的格式放你自己的数据：

```bash
python jeveval.py run --spec spec.json --data data.jsonl --out raw.jsonl
python jeveval.py report --raw raw.jsonl --target 0.99   # 换阈值重算，不再调 API
```

数据全程留在本机，除调用 TypeSafe API 外不外传。

## 目录

| 路径 | 内容 |
|---|---|
| `jeveval/` | **主要交付物**：评测框架。详见 [jeveval/README.md](jeveval/README.md) |
| `jeveval/jeveval.py` | CLI：`run` / `report` / `selftest` |
| `jeveval/metrics.py` | ECE / Brier / 可靠性曲线 / 阈值推荐，纯函数带自检 |
| `jeveval/examples/` | 数据格式样例 |
| `experiments/` | 探索期脚本和结论，见 [FINDINGS.md](FINDINGS.md) |

## 结论速览

在合成的中文任务上实测（详见 [FINDINGS.md](FINDINGS.md)）：

- confidence **不是摆设**：ECE 0.035，且偏差方向系统性偏保守（实际准确率 ≥ 自报概率）
- 但**题目无解时它会自信地错**——校准不能替你检查 state 里有没有漏字段
- **按问题形状选 primitive**：互斥选一个用 Choice，独立标签用 Noul。
  把互斥问题拆成多个 Noul 实测更差（丢掉了归一化约束）
- **重复采样基本没用**：概率会抖但 argmax 不抖；该做的是留一条不自动决策的带

这些数字来自合成数据，**不要直接搬到你的业务上**——这正是这个框架存在的意义。
