#!/usr/bin/env python3
"""
Compile an ACT policy for the Dragonwing IQ-9075 Hexagon NPU, via Qualcomm AI
Hub and ONNX Runtime's QNN EP.

Can run ON THE BOARD itself: the only thing that needs the network is the AI
Hub step (the board pushes the traced model to AI Hub's cloud, the compile
happens there, and the result comes back). No separate dev machine required.

End to end, this script:
  1. loads + traces the ACT inference path (eval mode, VAE skipped),
  2. submits an AI Hub compile job targeting ONNX,
  3. downloads the result and MERGES the two-file form (model.onnx graph +
     model.data weights) into ONE self-contained .onnx,
  4. verifies the merged file is complete (loads under CPU ORT, zero external
     tensors),
  5. by default INSTALLS it straight into the checkpoint's pretrained_model
     directory, removes any stray model.data there, and patches config.json to
     the act_onnx_qnn policy type.
So after this finishes, lerobot-rollout just works, with no manual download,
unzip, or copy.

Why the merge matters: AI Hub hands the model back as a tiny model.onnx (~200
KB, just the wiring) plus a big model.data (~177 MB, the weights). The .onnx
only POINTS at the .data; move one without the other and it breaks with
"Cannot parse data from external tensors". Merging bakes the weights into the
.onnx so there is a single file to carry around.

Prerequisites:
    pip install qai-hub lerobot torch onnx onnxruntime
    qai-hub configure --api_token <token from https://aihub.qualcomm.com/account>

Precision: by default this produces a plain (un-quantized) ONNX graph. The
board's QNN EP runs it in fp16 on the HTP (the plugin sets
enable_htp_fp16_precision=1), so no calibration is needed and fp16 holds ACT's
accuracy. The graph carries no embedded QNN context, so the QNN EP builds the
HTP context at first load on the board (~20 s); the plugin's
qnn_context_cache_dir caches it so later loads skip the prepare. Pass --int8
for the optional quantize-job path.

Camera keys are "front"/"wrist"; the ONNX input names are
state/front_image/wrist_image.
"""

import argparse
import glob
import os
import shutil
import tempfile
import zipfile

import numpy as np
import torch

import qai_hub as hub
from lerobot.policies.act.modeling_act import ACTPolicy


def inline_external_data(src: str, dst: str) -> str:
    """Repack an external-data ONNX (.onnx + .data) into ONE self-contained
    file. AI Hub's profiler loads the model host-side with ORT shape
    inference, which cannot resolve EXTERNAL tensors and fails with
    'Cannot parse data from external tensors'. Inlining pulls the weights
    into the protobuf; the embedded QNN context binary is preserved, so the
    result is still the compiled NPU model. Returns dst."""
    import onnx
    from onnx.external_data_helper import (
        convert_model_from_external_data,
        _get_all_tensors,
    )

    model = onnx.load(src, load_external_data=True)
    convert_model_from_external_data(model)
    onnx.save_model(model, dst, save_as_external_data=False)

    # Check ALL tensors, not just graph.initializer: AI Hub emits some weights
    # as Constant-node attribute tensors, which an initializer-only check would
    # miss (that is how a half-external file slipped through before). Reload
    # WITHOUT external data so any still-external tensor stays flagged.
    check = onnx.load(dst, load_external_data=False)
    remaining = [
        t.name for t in _get_all_tensors(check)
        if t.data_location == onnx.TensorProto.EXTERNAL
    ]
    if remaining:
        raise RuntimeError(
            f"Inline failed: {len(remaining)} external tensors remain in {dst} "
            f"(e.g. {remaining[:3]})"
        )
    onnx.checker.check_model(dst)
    return dst


def verify_onnx_cpu(path: str) -> None:
    """Load the inline model under CPU ORT to prove it resolves before we
    hand it back to AI Hub or copy it to the board."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("  Inline model loads under CPU ORT.")
    print("  Inputs: ", [(i.name, i.shape) for i in sess.get_inputs()])
    print("  Outputs:", [(o.name, o.shape) for o in sess.get_outputs()])


def assert_self_contained(path: str) -> None:
    """Fail loudly if `path` still references external data, so a broken,
    half-downloaded, or graph-only file can never reach the checkpoint dir.
    This is the 'verify the download actually completed' check."""
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    size_mb = os.path.getsize(path) / 1e6
    model = onnx.load(path, load_external_data=False)
    ext = [t.name for t in _get_all_tensors(model)
           if t.data_location == onnx.TensorProto.EXTERNAL]
    if ext:
        raise RuntimeError(
            f"{path} is NOT self-contained: {len(ext)} external tensors "
            f"(e.g. {ext[:3]}), size {size_mb:.1f} MB. The weights (model.data) "
            f"did not get merged in."
        )
    if size_mb < 10:
        raise RuntimeError(
            f"{path} is only {size_mb:.1f} MB with no external tensors; that is "
            f"too small to hold the ACT weights. The merge produced a "
            f"graph-only file."
        )
    print(f"  Verified self-contained: {path} ({size_mb:.1f} MB, 0 external tensors).")


def patch_config_json(config_path: str, onnx_filename: str, qnn_cache_dir: str) -> None:
    """Flip a checkpoint's config.json to the act_onnx_qnn policy and set the
    plugin fields, preserving every other field. Idempotent."""
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["type"] = "act_onnx_qnn"
    cfg["onnx_filename"] = onnx_filename
    cfg["qnn_context_cache_dir"] = qnn_cache_dir
    cfg.setdefault("htp_performance_mode", "burst")
    cfg.setdefault("qnn_rpc_control_latency", 100)
    cfg.setdefault("strict_npu", False)
    cfg.setdefault("prefetch_threshold", 5)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  Patched {config_path}: type=act_onnx_qnn, onnx_filename={onnx_filename}")


# ---------------------------------------------------------------------------
# Inference wrapper: exposes only the deployable ACT forward (no VAE, no
# action-queue), taking explicit tensors so the tracer produces a clean,
# fixed-shape graph. EDIT the camera args to match YOUR policy's cameras.
# ---------------------------------------------------------------------------
class ACTInferenceWrapper(torch.nn.Module):
    def __init__(self, policy: ACTPolicy):
        super().__init__()
        policy.eval()
        self.policy = policy

    @torch.no_grad()
    def forward(self, state: torch.Tensor, front_image: torch.Tensor, wrist_image: torch.Tensor):
        # predict_action_chunk runs the ACT model in eval mode, which skips
        # the training-only VAE encoder entirely (confirmed from LeRobot
        # source: the VAE path is gated on self.training).
        batch = {
            "observation.state": state,
            "observation.images.front": front_image,
            "observation.images.wrist": wrist_image,
        }
        actions = self.policy.predict_action_chunk(batch)
        return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to your trained ACT checkpoint's pretrained_model directory")
    p.add_argument("--device", default="Dragonwing IQ-9075 EVK",
                   help="AI Hub device name to compile/profile against")
    p.add_argument("--state-dim", type=int, default=6,
                   help="Robot state vector size (6 for SO-101: 5 joints + gripper)")
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--output", default="act_so101_onnx_qnn.onnx",
                   help="Filename for the downloaded compiled ONNX model")
    p.add_argument("--skip-aihub-profile", action="store_true",
                   help="Skip the optional AI Hub profile/inference jobs. The "
                        "board-ready .onnx is still produced. Use this if you "
                        "only want the deployable model and not the AI Hub "
                        "latency sanity-check.")
    p.add_argument("--int8", action="store_true",
                   help="Optional: int8-quantize the compiled model through AI "
                        "Hub's quantize job (the old `--quantize_full_type int8` "
                        "compile flag is no longer accepted for the ONNX target). "
                        "Default is OFF: a plain ONNX graph that the board's QNN "
                        "EP runs in fp16 on the HTP (enable_htp_fp16_precision), "
                        "which needs no calibration. int8 needs calibration data; "
                        "see --calib-samples.")
    p.add_argument("--calib-samples", type=int, default=16,
                   help="Number of calibration samples for --int8. These are "
                        "RANDOM normalized tensors by default, which is enough to "
                        "run but can cost accuracy; for best int8 accuracy, edit "
                        "build_calibration_data() to feed real normalized frames "
                        "from your dataset.")
    p.add_argument("--install-dir", default=None,
                   help="Directory to install the finished self-contained .onnx "
                        "into (the checkpoint's pretrained_model dir, where "
                        "config.json lives). Defaults to --checkpoint, so the "
                        "model lands right where lerobot-rollout loads it. The "
                        "file is also left in the working dir as --output.")
    p.add_argument("--no-install", action="store_true",
                   help="Do not install into the checkpoint dir / patch config; "
                        "just write --output in the working dir.")
    p.add_argument("--no-patch-config", action="store_true",
                   help="Install the .onnx but leave config.json untouched (you "
                        "will set type=act_onnx_qnn and onnx_filename yourself).")
    p.add_argument("--qnn-cache-dir", default="/home/ubuntu/act_qnn_ctx_cache",
                   help="Writable dir on the board where the QNN EP caches the "
                        "HTP context; written into config.json as "
                        "qnn_context_cache_dir.")
    args = p.parse_args()

    print(f"Loading ACT policy from {args.checkpoint} ...")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy = policy.to("cpu")
    wrapper = ACTInferenceWrapper(policy)
    wrapper.eval()

    # Fixed-shape example inputs (the NPU wants static shapes). Batch 1.
    dummy_state = torch.randn(1, args.state_dim)
    dummy_front = torch.randn(1, 3, args.image_height, args.image_width)
    dummy_wrist = torch.randn(1, 3, args.image_height, args.image_width)

    print("Tracing model ...")
    traced = torch.jit.trace(wrapper, (dummy_state, dummy_front, dummy_wrist))

    input_specs = {
        "state": tuple(dummy_state.shape),
        "front_image": tuple(dummy_front.shape),
        "wrist_image": tuple(dummy_wrist.shape),
    }

    # -----------------------------------------------------------------------
    # Compile options.
    #   --target_runtime onnx  -> produce a plain .onnx target model.
    #
    # The old `--quantize_full_type int8` flag is NO LONGER accepted for the
    # ONNX target (AI Hub now rejects it: "The --quantize_full_type option is
    # not supported for target_runtime='ONNX'"). Quantization moved to a
    # separate quantize job (see --int8 below).
    #
    # Default (no --int8): a plain fp32 ONNX graph, no embedded QNN context.
    # The board's QNN EP builds the HTP context at first load (~20 s; cached
    # via the plugin's qnn_context_cache_dir) and runs the graph in fp16 on
    # the HTP (the plugin sets enable_htp_fp16_precision=1). ACT is a
    # CNN+transformer without extreme activation ranges, so fp16 holds
    # accuracy and needs no calibration.
    # -----------------------------------------------------------------------
    compile_options = "--target_runtime onnx"

    print(f"Submitting compile job for device: {args.device} ...")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=hub.Device(args.device),
        input_specs=input_specs,
        options=compile_options,
    )
    compile_job.wait()
    print(f"Compile job finished: {compile_job.url}")
    compiled_model = compile_job.get_target_model()

    # -----------------------------------------------------------------------
    # Optional int8 path: run AI Hub's quantize job on the compiled model.
    # This is the current replacement for the removed --quantize_full_type
    # flag. It needs calibration data; by default we use random normalized
    # tensors (enough to produce a model, but swap in real frames for
    # accuracy). Leave --int8 off to deploy the fp16 model, which is the
    # recommended path for ACT.
    # -----------------------------------------------------------------------
    if args.int8:
        from qai_hub.client import QuantizeDtype

        def build_calibration_data(n):
            # NOTE: the ONNX graph expects ALREADY-NORMALIZED inputs (0.6.x
            # keeps normalization in the processor pipeline). For best int8
            # accuracy, replace these random tensors with real observations run
            # through your checkpoint's policy_preprocessor so the value ranges
            # match what the arm feeds at run time.
            return {
                "state": [np.random.randn(1, args.state_dim).astype(np.float32)
                          for _ in range(n)],
                "front_image": [np.random.randn(1, 3, args.image_height, args.image_width).astype(np.float32)
                                for _ in range(n)],
                "wrist_image": [np.random.randn(1, 3, args.image_height, args.image_width).astype(np.float32)
                                for _ in range(n)],
            }

        print(f"Submitting int8 quantize job ({args.calib_samples} RANDOM "
              f"calibration samples; edit build_calibration_data for real "
              f"frames) ...")
        quantize_job = hub.submit_quantize_job(
            model=compiled_model,
            calibration_data=build_calibration_data(args.calib_samples),
            weights_dtype=QuantizeDtype.INT8,
            activations_dtype=QuantizeDtype.INT8,
        )
        quantize_job.wait()
        print(f"Quantize job finished: {quantize_job.url}")

        # Re-target the quantized model to the ONNX runtime so the result is a
        # deployable .onnx for the board's QNN EP.
        print("Compiling the quantized model to the ONNX target ...")
        q_compile_job = hub.submit_compile_job(
            model=quantize_job.get_target_model(),
            device=hub.Device(args.device),
            options="--target_runtime onnx",
        )
        q_compile_job.wait()
        print(f"Quantized compile job finished: {q_compile_job.url}")
        compiled_model = q_compile_job.get_target_model()

    # -----------------------------------------------------------------------
    # The compiled ONNX comes back in EXTERNAL-DATA form (a small .onnx graph
    # plus a large .data). AI Hub's profiler and inference jobs load it
    # host-side with ORT shape inference, which cannot resolve external
    # tensors -> "Cannot parse data from external tensors ... /model/
    # Constant_5_output_0", and the job fails before running anything.
    #
    # Fix: download the compiled model, repack it into ONE self-contained
    # file, verify it under CPU ORT, re-upload, and profile/infer on THAT.
    # Inlining only moves weights into the protobuf; the embedded QNN context
    # binary is preserved, so it is still the compiled NPU model.
    # -----------------------------------------------------------------------
    workdir = tempfile.mkdtemp(prefix="act_onnx_")
    raw_dl = os.path.join(workdir, "compiled.onnx")
    returned = compiled_model.download(raw_dl)

    # download() is inconsistent across qai-hub versions: it may return a
    # bare .onnx, a dir with a sibling .data, or (commonly for ONNX targets)
    # a .zip like "compiled.onnx.onnx.zip". Normalize all of these: unzip any
    # archive, then find the real .onnx anywhere under workdir.
    returned = returned if isinstance(returned, str) else raw_dl
    for z in glob.glob(os.path.join(workdir, "**", "*.zip"), recursive=True):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(workdir)

    onnx_hits = glob.glob(os.path.join(workdir, "**", "*.onnx"), recursive=True)
    if not onnx_hits:
        raise RuntimeError(
            f"No .onnx found in downloaded model at {workdir}. "
            f"Contents: {os.listdir(workdir)}"
        )

    # If several .onnx exist, prefer the one with a sibling .data (the real
    # external-data graph), then the largest.
    def _score(path):
        d = os.path.dirname(path)
        has_data = any(f.endswith(".data") for f in os.listdir(d))
        return (has_data, os.path.getsize(path))

    src_onnx = max(onnx_hits, key=_score)
    print(f"Using compiled graph: {src_onnx}")

    # === ESSENTIAL PATH: this must succeed and produces the board file. ===
    print("Inlining external data into a single self-contained ONNX ...")
    inline_path = os.path.join(workdir, "compiled_inline.onnx")
    inline_external_data(src_onnx, inline_path)
    verify_onnx_cpu(inline_path)

    # Save the self-contained model NOW, before any optional AI Hub step, so
    # the deployable artifact exists even if the profile/inference bonus
    # below hiccups. No .data to carry to the board.
    shutil.copy(inline_path, args.output)
    print(f"\n==> Board-ready self-contained ONNX written to: {args.output}")
    assert_self_contained(args.output)

    # === INSTALL: drop the finished file straight into the checkpoint dir, ===
    # remove any stray model.data, and patch config.json. After this,
    # lerobot-rollout just works with no manual steps.
    if not args.no_install:
        install_dir = args.install_dir or args.checkpoint
        onnx_basename = os.path.basename(args.output)
        installed = os.path.join(install_dir, onnx_basename)
        print(f"\nInstalling into checkpoint dir: {install_dir}")
        if os.path.abspath(installed) != os.path.abspath(args.output):
            shutil.copy(args.output, installed)
        assert_self_contained(installed)

        # Remove any stray two-file leftovers next to it (they would only
        # confuse a future load; the installed .onnx is self-contained).
        for stray in ("model.data", onnx_basename + ".data"):
            sp = os.path.join(install_dir, stray)
            if os.path.exists(sp):
                os.remove(sp)
                print(f"  Removed stray weights file: {sp}")

        cfg_path = os.path.join(install_dir, "config.json")
        if args.no_patch_config:
            print("  Left config.json untouched (--no-patch-config).")
        elif os.path.exists(cfg_path):
            patch_config_json(cfg_path, onnx_basename, args.qnn_cache_dir)
        else:
            print(f"  [WARN] No config.json at {cfg_path}; skipped patching.")
    print("    (Everything below is an optional AI Hub sanity check.)\n")

    # === OPTIONAL BONUS: AI Hub profile + inference verification. ===
    # Wrapped so an AI-Hub-side issue can NEVER fail the run or lose the model
    # above. Your real latency number comes from lerobot-rollout on the board.
    if args.skip_aihub_profile:
        print("Skipping AI Hub profile/inference (--skip-aihub-profile).")
    else:
        try:
            print("Re-uploading the inline model to AI Hub ...")
            inline_uploaded = hub.upload_model(inline_path)

            print("Submitting profile job (AI Hub latency sanity-check) ...")
            profile_job = hub.submit_profile_job(
                model=inline_uploaded,
                device=hub.Device(args.device),
            )
            profile_job.wait()
            print(f"Profile job finished: {profile_job.url}")
            print("  --> AI Hub latency is a sanity check; the headline number")
            print("      comes from lerobot-rollout on the board.")

            print("Submitting inference job to verify numerical correctness ...")
            inference_job = hub.submit_inference_job(
                model=inline_uploaded,
                device=hub.Device(args.device),
                inputs={
                    "state": [dummy_state.numpy()],
                    "front_image": [dummy_front.numpy()],
                    "wrist_image": [dummy_wrist.numpy()],
                },
            )
            inference_job.wait()
            on_device_output = inference_job.download_output_data()

            with torch.no_grad():
                local_output = wrapper(dummy_state, dummy_front, dummy_wrist).cpu().numpy()

            # Random inputs make absolute values meaningless, but matching
            # shapes confirm the graph is intact. (int8 adds some drift.)
            od = list(on_device_output.values())[0][0]
            print("Local (PyTorch) output shape:", tuple(local_output.shape))
            print("On-device output shape:      ", tuple(np.asarray(od).shape))
            print("Max abs diff:                ", float(np.max(np.abs(local_output - od))))
        except Exception as e:
            print(f"\n[WARN] AI Hub profile/inference step failed: {e}")
            print("[WARN] This does NOT affect the board-ready model already")
            print(f"[WARN] written to {args.output}. Re-run with")
            print("[WARN] --skip-aihub-profile to suppress this optional step.")

    print("")
    if args.no_install:
        print("NEXT: copy the .onnx into the checkpoint's pretrained_model dir,")
        print("set type=act_onnx_qnn + onnx_filename in config.json, then run")
        print("lerobot-rollout.")
    else:
        print("DONE: model installed and config.json patched. Just run")
        print("lerobot-rollout with --policy.path pointing at this checkpoint.")
        print("(Make sure the board's QNN cache dir exists:")
        print(f"   mkdir -p {args.qnn_cache_dir})")


if __name__ == "__main__":
    main()
