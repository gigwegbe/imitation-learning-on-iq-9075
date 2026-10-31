"""
Drop-in replacement for ACTPolicy where the ResNet+Transformer forward
pass runs on a Qualcomm Hexagon NPU via ONNX Runtime's QNN execution
provider, loading a Qualcomm-AI-Hub-compiled .onnx (with an embedded QNN
context binary), instead of PyTorch.

This is the ROUTE B (ONNX Runtime + QNN EP) sibling of Naveen's
act_htp policy, which used LiteRT + the QNN TFLite delegate. The
registration decorator, the prefetch threading, the queue logic, and the
reuse of LeRobot's own pre/post processors for normalization are kept
IDENTICAL to his design. Only the inference core (_ensure_session and
predict_action_chunk) is swapped from LiteRT to onnxruntime.
"""

from collections import deque
from pathlib import Path
from threading import Lock, Thread
import logging
import time
import numpy as np
import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_act_onnx_qnn import ACTONNXQNNConfig


class ACTONNXQNNPolicy(PreTrainedPolicy, nn.Module):
    config_class = ACTONNXQNNConfig
    name = "act_onnx_qnn"

    def __init__(self, config: ACTONNXQNNConfig, dataset_stats=None):
        PreTrainedPolicy.__init__(self, config)
        nn.Module.__init__(self)
        self.config = config

        # Dummy param so standard nn.Module machinery in lerobot-rollout has
        # at least one parameter to operate on without special-casing.
        self._unused_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._session = None
        self._input_names = None          # cached ORT input names
        self._output_name = None          # first ORT output name
        self._run_options = None          # per-run QNN perf config
        self._ep_registered = False       # EP library registered once
        self._ep_lib_name = "QNNExecutionProvider"
        self._action_queue = deque([])
        self._session_lock = Lock()
        self._queue_lock = Lock()
        self._prefetch_start_lock = Lock()
        self._prefetch_thread: Thread | None = None
        self._warmed_up = False

    # -----------------------------------------------------------------
    # Inference core, ONNX Runtime + QNN EP (this is the Route B swap)
    #
    # Uses the onnxruntime-qnn >= 2.7 API: register the EP library once,
    # pick the QNN EP device via get_ep_devices(), and bind it to the
    # session with add_provider_for_devices(). HTP perf settings live on
    # a RunOptions passed at run() time (not in the provider options),
    # which is the 2.7 way to drive qnn.perf_mode / qnn.rpc_control_latency.
    # -----------------------------------------------------------------
    def _ensure_session(self):
        if self._session is not None:
            return

        import onnxruntime as ort
        import onnxruntime_qnn as qnn_ep

        log = logging.getLogger(__name__)

        model_path = Path(self.config.pretrained_path) / self.config.onnx_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Compiled ONNX model not found at {model_path}. Check "
                f"onnx_filename in your config matches the actual filename "
                f"in this checkpoint directory."
            )

        # 1. Register the QNN EP shared library once per process. ORT keeps
        #    this registration global, so guard against a double-register.
        if not self._ep_registered:
            ep_lib_path = qnn_ep.get_library_path()
            try:
                ort.register_execution_provider_library(
                    self._ep_lib_name, ep_lib_path
                )
            except Exception as e:
                # Already registered (e.g. a previous session in the same
                # process) is fine; anything else is real.
                if "already" not in str(e).lower():
                    raise
            self._ep_registered = True
            log.info(
                f"ACTONNXQNNPolicy: registered QNN EP library "
                f"'{self._ep_lib_name}' (onnxruntime_qnn {qnn_ep.__version__}) "
                f"from {ep_lib_path}"
            )

        # 2. Find the QNN EP device(s) ORT now knows about.
        all_ep_devices = ort.get_ep_devices()
        selected = [d for d in all_ep_devices if d.ep_name == self._ep_lib_name]
        if not selected:
            msg = (
                "ACTONNXQNNPolicy: no QNNExecutionProvider device found via "
                "get_ep_devices(). onnxruntime-qnn is installed but the QNN "
                "backend did not initialize, check the QAIRT/QNN SDK on the "
                "board and that this is the aarch64 board build."
            )
            if self.config.strict_npu:
                raise RuntimeError(msg)
            log.warning(msg + " Falling back to CPU.")

        # 3. Bind the QNN EP (backed by the HTP backend) to the session.
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if selected:
            ep_options = {"backend_path": qnn_ep.get_qnn_htp_path()}
            # Run a non-quantized (fp32) ONNX graph in fp16 on the HTP. This is
            # what makes the default (un-quantized) compile fast on the NPU: the
            # Hexagon executes the matmuls/convs in fp16 instead of falling back
            # to slow fp32. Harmless for an already-int8 model. Defaults to "1"
            # in recent ORT, set explicitly so behavior is pinned.
            ep_options["enable_htp_fp16_precision"] = "1"
            if self.config.qnn_context_cache_dir is not None:
                # Cache the consumed/generated QNN context so repeated loads
                # skip the HTP prepare.
                ep_options["qnn_context_cache_enable"] = "1"
                ep_options["qnn_context_cache_path"] = str(
                    Path(self.config.qnn_context_cache_dir) / "act_qnn_ctx.bin"
                )
            sess_options.add_provider_for_devices(selected, ep_options)

        self._session = ort.InferenceSession(
            str(model_path), sess_options=sess_options
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name

        # 4. Per-run QNN config. perf_mode = "burst" for max HTP clocks;
        #    rpc_control_latency trims the RPC wake latency on each call.
        self._run_options = ort.RunOptions()
        self._run_options.add_run_config_entry(
            "qnn.perf_mode", self.config.htp_performance_mode
        )
        self._run_options.add_run_config_entry(
            "qnn.rpc_control_latency", str(self.config.qnn_rpc_control_latency)
        )

        active = self._session.get_providers()
        log.info(
            f"ACTONNXQNNPolicy: ONNX model loaded. Active providers: {active}. "
            f"Inputs: {self._input_names}. "
            f"If QNNExecutionProvider is present, the ACT network runs on the "
            f"Hexagon NPU; the 'device=cpu' shown at policy load time refers "
            f"only to PyTorch-side pre/post-processing tensors."
        )
        if "QNNExecutionProvider" not in active:
            log.warning(
                "ACTONNXQNNPolicy: QNNExecutionProvider is NOT active, the "
                "model is running on CPU. Check onnxruntime-qnn install and "
                "the QAIRT/QNN SDK version on the board."
            )

    def _run(self, feeds):
        """Single-point session.run with the QNN RunOptions applied."""
        return self._session.run([self._output_name], feeds, self._run_options)

    def _warm_up(self):
        if self._warmed_up:
            return
        self._ensure_session()
        feeds = {
            i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape],
                             dtype=np.float32)
            for i in self._session.get_inputs()
        }
        t0 = time.perf_counter()
        with self._session_lock:
            self._run(feeds)
        logging.getLogger(__name__).info(
            f"ACTONNXQNNPolicy: warm-up inference took "
            f"{(time.perf_counter() - t0) * 1000:.1f} ms"
        )
        self._warmed_up = True

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path=None, *args,
                        config=None, dataset_stats=None, **kwargs):
        # This policy carries NO torch weights (compute is the ONNX graph on
        # the NPU, loaded lazily). lerobot's make_policy calls from_pretrained,
        # which would otherwise try to load the checkpoint's ACT safetensors
        # into this weightless wrapper. Just construct it; the .onnx holds the
        # weights. make_policy passes config=, dataset_stats=, pretrained_name_or_path=.
        return cls(config, dataset_stats=dataset_stats)

    def get_optim_params(self):
        return []

    def reset(self):
        with self._queue_lock:
            self._action_queue.clear()
        try:
            self._warm_up()
        except ImportError:
            pass

    def forward(self, batch):
        raise NotImplementedError("ACTONNXQNNPolicy is inference-only.")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self._ensure_session()

        with self._session_lock:
            state_np = batch["observation.state"].cpu().numpy().astype(np.float32)
            front_np = batch["observation.images.front"].cpu().numpy().astype(np.float32)
            wrist_np = batch["observation.images.wrist"].cpu().numpy().astype(np.float32)

            # Match each observation to an ONNX input by substring. The AI
            # Hub compile named these "state"/"front_image"/"wrist_image"
            # (see compile_act_onnx_qnn.py); match on "state"/"front"/"wrist".
            feeds = {}
            matched = {"state": False, "front": False, "wrist": False}
            for name in self._input_names:
                low = name.lower()
                if "state" in low:
                    feeds[name] = state_np
                    matched["state"] = True
                elif "front" in low:
                    feeds[name] = front_np
                    matched["front"] = True
                elif "wrist" in low:
                    feeds[name] = wrist_np
                    matched["wrist"] = True

            if not all(matched.values()):
                raise RuntimeError(
                    f"Could not match all three inputs by name -- found ONNX "
                    f"input names {self._input_names}. Fix the name-matching "
                    f"logic above to your model's actual tensor names."
                )

            outputs = self._run(feeds)
            action_chunk = outputs[0]
            return torch.from_numpy(np.asarray(action_chunk))

    # -----------------------------------------------------------------
    # Prefetch + queue logic below is IDENTICAL to Naveen's act_htp.
    # -----------------------------------------------------------------
    def _prefetch_worker(self, batch: dict[str, Tensor]):
        t0 = time.perf_counter()
        try:
            action_chunk = self.predict_action_chunk(batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logging.getLogger(__name__).info(
                f"ACTONNXQNNPolicy: prefetch inference took {elapsed_ms:.1f} ms"
            )
            with self._queue_lock:
                n = self.config.n_action_steps
                self._action_queue.extend(action_chunk[0, i] for i in range(n))
        finally:
            with self._prefetch_start_lock:
                self._prefetch_thread = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        with self._queue_lock:
            queue_len = len(self._action_queue)

        with self._prefetch_start_lock:
            in_flight = self._prefetch_thread
            if queue_len <= self.config.prefetch_threshold and in_flight is None:
                in_flight = Thread(target=self._prefetch_worker, args=(batch,), daemon=True)
                self._prefetch_thread = in_flight
                in_flight.start()

        with self._queue_lock:
            if len(self._action_queue) > 0:
                return self._action_queue.popleft()

        if in_flight is not None:
            logging.getLogger(__name__).warning("ACTONNXQNNPolicy: queue empty")
            t0 = time.perf_counter()
            in_flight.join()
            logging.getLogger(__name__).info(
                f"ACTONNXQNNPolicy: waited {(time.perf_counter() - t0) * 1000:.1f} ms "
                f"for in-flight prefetch"
            )
            with self._queue_lock:
                if len(self._action_queue) > 0:
                    return self._action_queue.popleft()

        logging.getLogger(__name__).warning(
            "ACTONNXQNNPolicy: no prefetch in flight and queue empty"
        )
        action_chunk = self.predict_action_chunk(batch)
        with self._queue_lock:
            n = self.config.n_action_steps
            self._action_queue.extend(action_chunk[0, i] for i in range(n))
            return self._action_queue.popleft()
