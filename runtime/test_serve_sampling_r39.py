"""[R3.9SMP] Server-side sampling plumbing with the FakeRunner (no model, no GPU). Run: python test_serve_sampling_r39.py"""
import sys

import glm53_fold_serve as S

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


class RecordingFake(S.FakeRunner):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sampling_seen = []

    def generate(self, req):
        self.sampling_seen.append(getattr(req, "sampling", "MISSING"))
        return super().generate(req)


tpl = S.ChatTemplate.from_file("C:/models/GLM53-backbone-r34h/chat_template.jinja")
fake = RecordingFake()
eng = S.ServeEngine(fake, tpl, default_thinking=True, max_context=100000,
                    default_sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 0})
msgs = [{"role": "user", "content": "hello"}]

p = eng.prepare({"messages": msgs})
check("server defaults applied", p.sampling == {"temperature": 1.0, "top_p": 0.95, "top_k": 0}, str(p.sampling))
p = eng.prepare({"messages": msgs, "temperature": 0, "seed": 5})
check("request overrides (temperature 0 = greedy, seed)", p.sampling["temperature"] == 0.0 and p.sampling["seed"] == 5)
check("temperature no longer listed as ignored", "temperature" not in p.ignored and "seed" not in p.ignored)
p = eng.prepare({"messages": msgs, "frequency_penalty": 0.3})
check("penalties still reported as ignored", p.ignored.get("frequency_penalty") == 0.3)
for bad in ({"temperature": -0.1}, {"top_p": 0}, {"top_k": 2.5}, {"temperature": "x"}):
    try:
        eng.prepare({"messages": msgs, **bad})
        check(f"invalid {bad} rejected", False)
    except S.RequestError as e:
        check(f"invalid {bad} rejected", True, f"(param {e.param})")

res = eng.execute(eng.prepare({"messages": msgs, "temperature": 0.7}))
check("generation hands the request's sampling to the runner",
      bool(fake.sampling_seen) and fake.sampling_seen[-1] and fake.sampling_seen[-1].get("temperature") == 0.7,
      str(fake.sampling_seen[-1:] if fake.sampling_seen else "no call"))
h = eng.health()
check("health reports the sampling defaults", isinstance(h["sampling"], dict) and h["sampling"]["temperature"] == 1.0)
eng0 = S.ServeEngine(RecordingFake(), tpl)
check("default engine stays greedy", eng0.health()["sampling"] == "greedy" and eng0.prepare({"messages": msgs}).sampling["temperature"] == 0.0)

print("RESULT:", "ALL PASS" if not FAILS else f"FAILED {FAILS}")
sys.exit(1 if FAILS else 0)
