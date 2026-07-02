#!/usr/bin/env bash
# Build the SHIFT Python 3.9 env (external/shift/.venv-shift).
# NOTE: repo pins torch==2.0.1, which CANNOT run on Blackwell (sm_120) GPUs.
# We install a cu128 torch instead (>=2.7 supports sm_120). Documented deviation.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-shift/bin/python
step(){ echo; echo "===== $* ====="; }

step "1/8 build tools (gym 0.21 needs old setuptools/wheel)"
uv pip install --python "$PY" "setuptools==65.5.0" "wheel==0.38.4" || { echo "FAIL:buildtools"; exit 11; }

step "2/8 torch + torchvision (cu128, Blackwell-capable; resolver picks newest cp39 wheel)"
uv pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cu128 || { echo "FAIL:torch"; exit 12; }

step "3/8 gym 0.21 (no build isolation so it sees the pinned setuptools)"
uv pip install --python "$PY" --no-build-isolation "gym==0.21.0" || echo "WARN:gym021 (only 1 import in repo; will patch to gymnasium if needed)"

step "4/8 gymnasium[atari] + ale-py (env backend) + ROMs"
uv pip install --python "$PY" "gymnasium[atari]==0.29.1" "ale-py" "autorom[accept-rom-license]" || { echo "FAIL:gymnasium"; exit 14; }
.venv-shift/bin/AutoROM --accept-license || echo "WARN:AutoROM (gymnasium[atari] may already bundle ROMs)"

step "5/8 denoising_diffusion_pytorch (no-deps to protect torch) + its real deps"
uv pip install --python "$PY" --no-deps "denoising_diffusion_pytorch==1.5.4" || echo "WARN:ddp"
uv pip install --python "$PY" einops ema-pytorch accelerate || echo "WARN:ddp-deps"

step "6/8 remaining requirements"
uv pip install --python "$PY" \
  appdirs==1.4.4 hydra-core==1.3.2 matplotlib omegaconf==2.3.0 opencv-contrib-python \
  Pillow pygame pygit2 pytest pytorch_pretrained_bert==0.6.2 query scikit-learn scipy \
  torcheval tqdm ultralytics_thop wandb || echo "WARN:rest (inspect log)"

step "7/8 numpy<2 (repo era; modern enough for newer torch)"
uv pip install --python "$PY" "numpy<2" || echo "WARN:numpy"

step "8/8 SANITY: torch CUDA on Blackwell + gymnasium Atari env"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(64, 64, device="cuda")
    print("cuda matmul ok, sum=", float((x @ x).sum()))
try:
    import gymnasium, ale_py
    gymnasium.register_envs(ale_py)
    env = gymnasium.make("FreewayNoFrameskip-v4")
    obs, _ = env.reset(seed=0)
    obs2, r, term, trunc, info = env.step(env.action_space.sample())
    print("gymnasium Freeway OK | obs shape:", obs.shape, "| action_space:", env.action_space)
    env.close()
except Exception as e:
    import traceback; print("ENV SANITY FAILED:"); traceback.print_exc()
try:
    import gym; print("gym (legacy):", gym.__version__)
except Exception as e:
    print("gym (legacy) not installed:", e)
PY
echo; echo "===== SETUP DONE ====="
