"""
quick_check.py — fast, dependency-light proof that the upgrades work.

Needs ONLY numpy. It mocks torch / torchvision / cv2 / skimage / PIL so it can
import the modules without a GPU, the dataset, MongoDB, or model weights.

It verifies:
  * 8-class subtype list (Change 4)
  * adaptive compression genuinely shrinks the latent as quality drops (Change 7)
  * protocol benchmark is fair: TCP fails on a drop, DTN delivers 100% at
    lowest overhead (Changes 2 & 3)

Run:  python quick_check.py
"""
import sys
from unittest.mock import MagicMock

# Mock heavy deps so imports succeed; keep numpy REAL.
for m in ["torch", "torch.nn", "torchvision", "torchvision.models",
          "torchvision.transforms", "cv2", "PIL", "skimage.metrics",
          "motor", "motor.motor_asyncio", "database", "dotenv"]:
    sys.modules[m] = MagicMock()

import numpy as np  # real
import ai_engine
import transfer_manager as tm

ok = True

def check(label, cond):
    global ok
    print(("  PASS " if cond else "  FAIL ") + label)
    ok = ok and cond

print("\n[1] 8-class subtypes (Change 4)")
check("SUBTYPES has 8 entries", len(ai_engine.SUBTYPES) == 8)
check("includes Adenosis & Papillary Carcinoma",
      "Adenosis" in ai_engine.SUBTYPES and "Papillary Carcinoma" in ai_engine.SUBTYPES)

print("\n[2] Adaptive compression shrinks with quality (Change 7)")
latent = np.random.randn(32, 28, 28).astype(np.float32)
sizes = {}
for q in (90, 50, 25):
    mode = ai_engine._quality_to_mode(q)
    packed = ai_engine._pack_latent(latent, mode)
    sizes[q] = len(packed)
    print(f"      quality={q:>2} mode={mode} bytes={len(packed)}")
check("size decreases as bandwidth/quality drops", sizes[90] > sizes[50] > sizes[25])

print("\n[3] Fair protocol benchmark over one drop trace (Changes 2 & 3)")
trace = [True, True, False, True, False, False, True, True]
res = tm.benchmark_protocols(trace, num_bundles=4, total_size_kb=100)
for k, v in res.items():
    print(f"      {k:11s} delivery={v['delivery_ratio']:>4} overhead={v['overhead']:>9} {v['efficiency']}")
check("TCP/IP fails on a mid-transfer drop", res["TCP_IP"]["status"].startswith("Failed"))
check("Codec-Aware DTN delivers 100%", res["Neural_DTN"]["delivery_ratio"] == "100%")
check("DTN overhead < Epidemic",
      float(res["Neural_DTN"]["overhead"].split()[0]) < float(res["Epidemic"]["overhead"].split()[0]))

print("\n" + ("ALL CHECKS PASSED ✅" if ok else "SOME CHECKS FAILED ❌"))
sys.exit(0 if ok else 1)
