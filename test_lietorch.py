import torch
import lietorch_backends
from lietorch import SE3

print(f"PyTorch: {torch.__version__}")
print(f"lietorch_backends: {lietorch_backends.__file__}")

results = []
def run(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
        results.append(True)
    except Exception as e:
        print(f"FAIL: {name} -> {e}")
        results.append(False)

run("no autocast - inv",      lambda: SE3.Identity(1, device='cuda').inv())
run("no autocast - mul+inv",  lambda: SE3.Identity(2, device='cuda') * SE3.Identity(2, device='cuda').inv())

def _with_autocast():
    with torch.cuda.amp.autocast(enabled=True):
        SE3.Identity(2, device='cuda') * SE3.Identity(2, device='cuda').inv()

run("autocast(True) - mul+inv", _with_autocast)

print(f"\n{'All PASS' if all(results) else 'Some FAILED'} ({sum(results)}/{len(results)})")
