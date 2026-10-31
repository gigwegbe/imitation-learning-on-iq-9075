# ACT on the Dragonwing IQ-9075 Hexagon NPU, end to end

Pull a trained ACT policy from Hugging Face, compile it through Qualcomm AI Hub
for the IQ-9075's Hexagon NPU, and run it on the SO-101 arm through ONNX
Runtime's QNN execution provider. This is the ONNX-Runtime + QNN path.

Written for **LeRobot 0.6.1** on both machines. On 0.6.1 there is no launcher
and no monkey-patch: `lerobot-rollout` auto-discovers the plugin and resolves
the `act_onnx_qnn` policy type by naming convention. (The old 0.3.x launcher,
`run_record_act_npu.py`, is kept in this package for reference only -- you do
not need it on 0.6.1.)

Two machines:

- a dev machine (x86/Mac) with network + Qualcomm AI Hub, for the compile
- the IQ-9075 board (aarch64), for the arm and inference

What runs where: the whole ACT network (ResNet vision + the transformer) runs
on the Hexagon as one AI-Hub-compiled ONNX graph, executed in fp16 on the HTP.
LeRobot handles the
camera reads, normalization, the action-chunk queue, and sending joint commands.
Normalization in 0.6.x lives entirely in the processor pipeline saved inside the
checkpoint (`policy_preprocessor.json` / `policy_postprocessor.json`), so the
ONNX graph takes already-normalized inputs and returns a normalized action
chunk that LeRobot de-normalizes. Nothing to export separately.

---

## 0. Prerequisites

Dev machine:

```bash
pip install qai-hub lerobot torch onnx onnxruntime huggingface_hub
qai-hub configure --api_token <token from https://aihub.qualcomm.com/account>
```

Board (aarch64):

```bash
# ONNX Runtime with the QNN execution provider (board only, aarch64).
pip install onnxruntime==1.29.0 onnxruntime-qnn==2.7.0
# plus your existing LeRobot 0.6.1 install
```

The CPU baseline in section 2 needs neither `onnxruntime-qnn` nor the plugin,
only LeRobot 0.6.1 and a CPU PyTorch. The QNN packages above are for the NPU
path (sections 3-5).

---

## 1. Pull the ACT checkpoint from Hugging Face

Download the checkpoint's `pretrained_model` directory. It must contain
`config.json` (`"type": "act"`), `model.safetensors`, and the saved processor
pipeline (`policy_preprocessor.json`, `policy_postprocessor.json` and their
`*_processor.safetensors`):

```bash
hf download gigwegbe/act-pick-and-place-cube-guided \
  --include "checkpoints/009000/pretrained_model/*" \
  --local-dir ./act_ckpt
# -> ./act_ckpt/checkpoints/009000/pretrained_model/{config.json, model.safetensors,
#    policy_preprocessor.json, policy_postprocessor.json, ...}
```

---

## 2. Baseline: run ACT on CPU (stock LeRobot, no compile)

Before compiling anything, run the trained checkpoint as-is on the board's CPU.
This is the honest baseline the NPU path is measured against: same arm, same
cameras, same normalization, just the plain PyTorch ACT policy on the CPU.
Expect roughly 1 second per action chunk -- smooth but slow. The NPU path in the
sections below runs the identical policy at about 66 ms, roughly 16x faster.

Nothing to compile or install here. The checkpoint's `config.json` already says
`"type": "act"`, so `lerobot-rollout` runs LeRobot's native ACT policy on the
CPU with no plugin, no `onnxruntime-qnn`, and no AI Hub. Run this on the pristine
checkpoint before section 4 patches `config.json` to `act_onnx_qnn`, or keep a
separate copy of the `pretrained_model` dir for the baseline.

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=./act_ckpt/checkpoints/009000/pretrained_model \
  --policy.device=cpu \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras='{
    front: { type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30 },
    wrist: { type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30 }
  }' \
  --task="pick the cube and place it in the box" \
  --duration=60 \
  --display_data=false
```

Here `--policy.device=cpu` governs the whole network, since there is no ONNX
graph yet, so every ResNet and transformer op runs on the CPU. That is the
difference from the NPU run in section 5, where the same flag only governs the
small pre/post tensors while the graph runs on the Hexagon.

Camera keys must be `front` and `wrist` to match the `observation.images.front`
/ `observation.images.wrist` keys in your config. The indices (`/dev/video0`,
`/dev/video2`) shift between boots and USB ports, so confirm them first with
`lerobot-find-cameras opencv` and swap if front/wrist are reversed.

Safety: clear the workspace and keep a hand on the e-stop for the first run.

Time one chunk here so you have a real before-number for the writeup. The NPU
path below is the same policy and the same checkpoint, compiled to the Hexagon.

---

## 3. Install the plugin (once)

```bash
# from this package root (the directory that holds pyproject.toml)
pip install -e .
mkdir -p /home/ubuntu/act_qnn_ctx_cache   # QNN HTP context cache
```

On 0.6.1 that is all the registration needed: `lerobot-rollout` calls
`register_third_party_plugins()` at startup, which imports every installed
`lerobot_policy_*` distribution, so `act_onnx_qnn` registers itself.

---

## 4. Compile through AI Hub and install into the checkpoint (one command)

This can run on the board itself; only the AI Hub step needs the network (the
board pushes the model to AI Hub's cloud and gets the compiled result back).

`compile_act_onnx_qnn.py` traces ACT (eval mode, VAE skipped), compiles to ONNX
on AI Hub, downloads and MERGES the two-file result (graph + weights) into one
self-contained `.onnx`, verifies it is complete, then installs it straight into
the checkpoint's `pretrained_model` directory, removes any stray `model.data`,
and patches `config.json` to the `act_onnx_qnn` policy. After it finishes,
there is nothing to copy by hand.

```bash
python compile_act_onnx_qnn.py \
  --checkpoint ./act_ckpt/checkpoints/009000/pretrained_model \
  --device "Dragonwing IQ-9075 EVK" \
  --state-dim 6 \
  --image-height 480 --image-width 640 \
  --output act_so101_onnx_qnn.onnx
```

The defaults match your config.json (cameras `front` + `wrist`, 640x480,
state/action dim 6, chunk 100), so nothing needs editing. By default
`--install-dir` is the `--checkpoint` dir, so the finished model lands exactly
where `lerobot-rollout` loads it.

What AI Hub returns and why the merge matters: the compiled model comes back as
a tiny `model.onnx` (~200 KB, just the graph wiring) plus a big `model.data`
(~177 MB, the weights). The `.onnx` only points at the `.data`; move one without
the other and loading fails with "Cannot parse data from external tensors". The
script merges the weights into the `.onnx` so there is a single self-contained
file, and verifies 0 external tensors remain before installing it.

Notes:

- Precision: the compile option is just `--target_runtime onnx` (the old
  `--quantize_full_type int8` flag is no longer accepted on the ONNX target).
  The plain ONNX runs in fp16 on the HTP via the plugin's
  `enable_htp_fp16_precision=1`, no calibration, recommended for ACT. The graph
  has no embedded QNN context, so the QNN EP builds it at first load (~20 s),
  cached in `qnn_context_cache_dir`.
- Optional int8: pass `--int8` to run AI Hub's `submit_quantize_job` (then a
  second compile to the ONNX target). It needs calibration data; the script
  uses random normalized tensors by default (`--calib-samples`), which runs but
  can cost accuracy, so edit `build_calibration_data()` to feed real normalized
  frames for a real int8 build. Start with fp16.
- Flags for the install behavior: `--install-dir <dir>` (default: the
  checkpoint dir), `--no-install` (just write `--output` locally, no config
  patch), `--no-patch-config` (install the `.onnx` but leave `config.json`
  alone), `--qnn-cache-dir <dir>` (written into the config as
  `qnn_context_cache_dir`).
- Camera inputs are named `state` / `front_image` / `wrist_image`. If you ever
  retrain with different camera keys, change them in BOTH
  `compile_act_onnx_qnn.py` and `modeling_act_onnx_qnn.py`.

---

## 5. Run on the arm, on the NPU (board)

LeRobot 0.6.1 has `lerobot-rollout`. The `base` strategy with `--task` +
`--duration` runs the policy on the arm with no dataset recording, so there is
no dataset dir to collide (the old `FileExistsError` dance is gone):

```bash
unset ADSP_LIBRARY_PATH
lerobot-rollout \
  --strategy.type=base \
  --policy.path=./act_ckpt/checkpoints/009000/pretrained_model \
  --policy.device=cpu \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras='{
    front: { type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30 },
    wrist: { type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30 }
  }' \
  --task="pick the cube and place it in the box" \
  --duration=60 \
  --display_data=false
```

Camera keys MUST be `front` and `wrist` to match the `observation.images.front`
/ `observation.images.wrist` keys in your config. The indices
(`/dev/video0`, `/dev/video2`) are a guess -- Linux video indices shift between
boots and USB ports, so confirm them first with `lerobot-find-cameras opencv`
and swap if front/wrist are reversed.

What to check at startup:

- the policy's log line listing active providers; `QNNExecutionProvider` present
  means the ACT graph is on the Hexagon. `--policy.device=cpu` only governs the
  small PyTorch-side pre/post tensors, not the ONNX graph.
- first inference is slow (HTP prepare, ~20 s); cached afterward in
  `qnn_context_cache_dir`.

Safety: clear the workspace and keep a hand on the e-stop for the first run.

---

## Strict-NPU check (recommended once)

Set `"strict_npu": true` in the config for one run. That omits the CPU provider,
so ONNX Runtime raises if any op can't run on the Hexagon, proving the whole
graph lands on the NPU. Then set it back to `false` for a deployment that
tolerates CPU fallback.

---

## Files

- `compile_act_onnx_qnn.py` — dev machine: ACT checkpoint -> AI Hub int8 ONNX, inlined + verified (+ optional AI Hub profile/inference).
- `inline_onnx.py` — fold a two-file external-data ONNX into one self-contained file (the compile script already does this; standalone for fixing an existing model).
- `lerobot_policy_act_onnx_qnn/` — the LeRobot policy plugin (board). Registers `act_onnx_qnn`, loads the `.onnx` on the Hexagon via ORT's QNN EP, keeps the action-queue + background prefetch, and reuses the checkpoint's saved processors for normalization.
- `run_record_act_npu.py` — **legacy**, for LeRobot 0.3.x only (registers the type, patches `get_policy_class`, runs `lerobot-record`). Not needed on 0.6.1; use `lerobot-rollout` as in step 5.

---

## Troubleshooting

- `Couldn't find a choice class for 'act_onnx_qnn'` / `NotImplementedError: Policy with name act_onnx_qnn`: the plugin isn't installed in the same env as `lerobot-rollout`. From this package root (where `pyproject.toml` is), `pip install -e .`, and make sure no empty same-named folder shadows it.
- `AttributeError: 'ACTONNXQNNConfig' object has no attribute 'prefetch_threshold'`: you're on an older copy of the plugin. This field is in the current `configuration_act_onnx_qnn.py`; reinstall the plugin.
- `QNN_DEVICE_ERROR_INVALID_CONFIG`: `unset ADSP_LIBRARY_PATH` before launching.
- `QNNExecutionProvider` not in the active-providers log: the graph fell back to CPU; check the onnxruntime-qnn install and the board's QAIRT/QNN SDK version.
- Camera `TimeoutError` on `/dev/videoN`: index shifted; re-run `lerobot-find-cameras opencv`, check nothing else holds the device (`fuser /dev/video*`), and try another USB port.
- `Cannot parse data from external tensors`: a two-file model slipped through; run `inline_onnx.py` on it.
- Many "Unexpected key(s)" weight-load warnings at load: expected and harmless — the ONNX wrapper carries no torch weights; the compiled `.onnx` holds the real ones.
