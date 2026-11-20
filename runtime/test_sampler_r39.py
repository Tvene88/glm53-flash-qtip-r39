"""[R3.9SMP] Unit tests of glm53_fold_sampler_r39 (no model needed). Run: python test_sampler_r39.py [--cuda]"""
import math
import sys

import torch

from glm53_fold_sampler_r39 import Sampler, FinalReplay, parse_request_sampling, K_MAX

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def run(dev):
    torch.manual_seed(0)
    V = 5000
    s = Sampler(dev)

    # 1. greedy (flag off) == argmax, ties included, any dtype
    ok = True
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for _ in range(200):
            row = torch.randn(V, device=dev).to(dt)
            row[torch.randint(0, V, (3,))] = row.max()             # ties at the max
            if int(s.pick(row)) != int(torch.argmax(row)):
                ok = False
    check(f"{dev}: greedy == argmax (600 rows, 3 dtypes, ties)", ok)

    # 2. distribution at T=1, top_p=1: frequencies of the top candidates match the softmax over the top 64
    row = torch.zeros(V, device=dev)
    row[:8] = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5], device=dev)
    row[8:] = -20.0
    s.set(1.0, 1.0, 0, seed=1)
    N = 20000
    cnt = torch.zeros(V, dtype=torch.long)
    for _ in range(N):
        s.refresh()
        cnt[int(s.pick(row))] += 1
    v, _ = torch.topk(row.float().cpu(), K_MAX)
    p = torch.softmax(v, 0)[:8]
    emp = cnt[:8].double() / N
    err = float((emp - p.double()).abs().max())
    check(f"{dev}: T=1 frequencies match softmax", err < 0.012, f"(max abs error {err:.4f})")

    # 3. temperature sharpens: T=0.5 -> frequencies follow softmax(2*logits)
    s.set(0.5, 1.0, 0, seed=2)
    cnt.zero_()
    for _ in range(N):
        s.refresh()
        cnt[int(s.pick(row))] += 1
    p2 = torch.softmax(2 * v, 0)[:8]
    err2 = float((cnt[:8].double() / N - p2.double()).abs().max())
    check(f"{dev}: T=0.5 frequencies match softmax(2x)", err2 < 0.012, f"(max abs error {err2:.4f})")

    # 4. top_p: nucleus of 0.5 at T=1 -> p = [.386 .234 .142 ...] -> tokens 0,1 only (0.386 < 0.5, 0.62 >= 0.5)
    s.set(1.0, 0.5, 0, seed=3)
    seen = set()
    for _ in range(4000):
        s.refresh()
        seen.add(int(s.pick(row)))
    check(f"{dev}: top_p=0.5 keeps only the nucleus", seen == {0, 1}, f"(seen {sorted(seen)})")

    # 5. top_k = 3
    s.set(1.0, 1.0, 3, seed=4)
    seen = set()
    for _ in range(4000):
        s.refresh()
        seen.add(int(s.pick(row)))
    check(f"{dev}: top_k=3", seen <= {0, 1, 2} and len(seen) == 3, f"(seen {sorted(seen)})")

    # 6. seed reproducibility
    s.set(1.0, 0.95, 0, seed=7)
    a = []
    for _ in range(50):
        s.refresh(); a.append(int(s.pick(row)))
    s.set(1.0, 0.95, 0, seed=7)
    b = []
    for _ in range(50):
        s.refresh(); b.append(int(s.pick(row)))
    check(f"{dev}: same seed -> same draws", a == b)

    # 7. back to greedy
    s.set(0.0)
    check(f"{dev}: temperature 0 -> greedy again", int(s.pick(row)) == 0 and not s.active)

    # 8. request parsing
    d = {"temperature": 1.0, "top_p": 0.95, "top_k": 0}
    ok = parse_request_sampling({}, d) == d and parse_request_sampling({"temperature": 0.2}, d)["temperature"] == 0.2
    bad = 0
    for body in ({"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": 1.5}, {"temperature": "hot"},
                 {"seed": True}):
        try:
            parse_request_sampling(body, d)
        except ValueError:
            bad += 1
    check("request parsing (defaults, override, 6 invalid bodies rejected)", ok and bad == 6, f"(rejected {bad})")


def run_graph():
    dev = torch.device("cuda")
    V = 154000
    s = Sampler(dev)
    logits = torch.zeros(1, 1, V, device=dev)
    logits[0, 0, :4] = torch.tensor([2.0, 1.9, 1.8, 1.7], device=dev)
    logits[0, 0, 4:] = -30.0
    out = {}

    def seg():
        out["nt"] = s.pick(logits[0, -1])

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        seg()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        seg()
    fr = FinalReplay(g, s)
    s.set(0.0)
    greedy = []
    for _ in range(20):
        fr.replay(); greedy.append(int(out["nt"]))
    check("cuda graph: greedy replays == argmax", set(greedy) == {0})
    s.set(1.0, 1.0, 0, seed=11)
    draws = []
    for _ in range(400):
        fr.replay(); draws.append(int(out["nt"]))
    seen = set(draws)
    check("cuda graph: sampled replays vary and stay in the top 4", seen == {0, 1, 2, 3}, f"(seen {sorted(seen)})")
    # timing of the pick inside a graph (sampler cost per token)
    gt = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gt):
        for _ in range(100):
            s.pick(logits[0, -1])
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); gt.replay(); e1.record(); torch.cuda.synchronize()
    print(f"[INFO] pick inside a graph, vocab {V}: {e0.elapsed_time(e1) / 100 * 1000:.1f} us per token")


if __name__ == "__main__":
    run("cpu")
    if "--cuda" in sys.argv and torch.cuda.is_available():
        run("cuda")
        run_graph()
    print("RESULT:", "ALL PASS" if not FAILS else f"FAILED {FAILS}")
    sys.exit(1 if FAILS else 0)
