from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_onnx_qnn")
@dataclass
class ACTONNXQNNConfig(ACTConfig):
    """
    Config for running a Qualcomm-AI-Hub-compiled ACT .onnx (with an
    embedded QNN context binary) on the IQ-9075 Hexagon NPU via
    ONNX Runtime's QNN execution provider.

    Subclasses the real ACTConfig and registers under "act_onnx_qnn", so
    changing "type": "act" -> "type": "act_onnx_qnn" in a checkpoint's
    config.json makes lerobot-rollout route here instead of the PyTorch
    ACTPolicy.
    """
    type: str = "act_onnx_qnn"

    # Filename of the Qualcomm-AI-Hub-compiled ONNX model, sitting in the
    # same checkpoint directory as this config.json.
    onnx_filename: str = "act_so101_onnx_qnn.onnx"

    # ---- ONNX Runtime QNN execution provider options ----
    # HTP performance mode, passed at run() time as the run-config entry
    # "qnn.perf_mode" (onnxruntime-qnn >= 2.7): "burst" | "high_performance"
    # | "balanced" | "power_saver" | ... "burst" = max HTP clocks.
    htp_performance_mode: str = "burst"

    # Passed at run() time as "qnn.rpc_control_latency" (microseconds). Lower
    # trims the RPC wake latency on each inference call. 100 is a sane start.
    qnn_rpc_control_latency: int = 100

    # Where ORT/QNN caches the compiled context on the board. Optional but
    # speeds up repeated loads a lot. Set to a writable dir on the board.
    qnn_context_cache_dir: str | None = None

    # If True, fail loudly when any op cannot run on the NPU instead of
    # silently falling back to CPU. This is the strict-verification trick:
    # turn it ON once to confirm the whole graph lands on the HTP, then you
    # may turn it OFF for a pragmatic deployment that allows CPU fallback.
    strict_npu: bool = False

    # When the action queue drops to this many remaining actions, start a
    # background prefetch of the next chunk so a fresh chunk is ready before
    # the queue empties. Must exceed the inference time measured in control
    # ticks (one NPU inference ~66-100 ms; at 30 fps that is ~2-3 ticks),
    # otherwise the queue empties at the chunk boundary and the arm stalls
    # there. Default 5 gives a comfortable margin; raise it if you still see a
    # hitch at chunk seams. ACTConfig has no such field -- it is specific to
    # this NPU-prefetch policy, so select_action reads it from here.
    prefetch_threshold: int = 5
