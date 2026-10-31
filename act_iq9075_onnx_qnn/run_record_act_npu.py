"""
Launch lerobot-record with the ACT NPU policy registered, on lerobot 0.3.4.

0.3.4 has no `lerobot-rollout`, and its `get_policy_class` is a hardcoded
if/elif with no plugin fallback (the convention-based resolver landed later).
So we:
  1. import the installed plugin so @PreTrainedConfig.register_subclass runs
     (registers the "act_onnx_qnn" CONFIG), and
  2. monkeypatch get_policy_class so the POLICY class resolves too.
Then hand off to lerobot-record's own main, which parses your CLI args.

Run from the directory that holds your checkpoint, with the usual
lerobot-record args (robot/cameras/policy/dataset). Relative --policy.path
still works because we only change sys.path, not the working directory.
"""

import os
import sys

# Drop cwd / this script's dir from sys.path so the pip-INSTALLED package wins
# over a same-named extracted folder sitting next to us (otherwise Python may
# import an empty namespace package and nothing registers).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path
               if p not in ("", ".") and os.path.abspath(p or ".") != _here]

import lerobot_policy_act_onnx_qnn  # noqa: F401  registers act_onnx_qnn (config)

_f = getattr(lerobot_policy_act_onnx_qnn, "__file__", None)
if not _f:
    raise RuntimeError(
        "Imported an empty namespace package for 'lerobot_policy_act_onnx_qnn'. "
        "Install it: pip install -e ./lerobot_policy_act_onnx_qnn"
    )

from lerobot.configs.policies import PreTrainedConfig
if "act_onnx_qnn" not in getattr(PreTrainedConfig, "_choice_registry", {}):
    raise RuntimeError("act_onnx_qnn did not register; check the plugin import.")

# Teach the hardcoded factory about our policy class.
import lerobot.policies.factory as _factory
_orig_get_policy_class = _factory.get_policy_class


def _get_policy_class(name):
    if name == "act_onnx_qnn":
        from lerobot_policy_act_onnx_qnn import ACTONNXQNNPolicy
        return ACTONNXQNNPolicy
    return _orig_get_policy_class(name)


_factory.get_policy_class = _get_policy_class
print(f"[run_record_act_npu] registered + patched act_onnx_qnn from {_f}")

from lerobot.scripts.lerobot_record import main

if __name__ == "__main__":
    main()
