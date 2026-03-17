"""Quick beam search test on the best checkpoint from the latest run."""
import sys, os, torch, glob
sys.path.insert(0, os.path.dirname(__file__))

from playground import load_checkpoint
from prepare import Tokenizer, load_dataset, _beam_search_solve
from rubiks import Cube, Episode

# Find latest model_best.pt
runs_dir = os.path.join(os.path.dirname(__file__), "runs")
best_files = sorted(glob.glob(os.path.join(runs_dir, "*/model_best.pt")), reverse=True)
if not best_files:
    best_files = sorted(glob.glob(os.path.join(runs_dir, "*/model.pt")), reverse=True)
checkpoint = best_files[0]
print(f"Using checkpoint: {checkpoint}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_checkpoint(checkpoint, device=device)
tokenizer = Tokenizer.from_directory()

payload = load_dataset()
all_eps = [Episode.from_dict(e) for e in payload["eval_episodes"]["id"]]
size2 = [e for e in all_eps if e.size == 2][:64]
size3 = [e for e in all_eps if e.size == 3][:64]

for width in [8, 32]:
    for label, eps in [("2x2", size2), ("3x3", size3)]:
        solved = 0
        for ep in eps:
            cube = Cube(ep.size)
            cube.apply_moves(ep.scramble)
            ok = _beam_search_solve(model, tokenizer, cube, beam_width=width, max_steps=100)
            solved += int(ok)
        print(f"Beam w={width} {label}: {solved}/{len(eps)} ({solved/len(eps):.0%})")
