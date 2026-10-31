#!/usr/bin/env python3
"""
Repack an external-data ONNX (model.onnx + model.data) into ONE self-contained
.onnx file, then verify it loads under ONNX Runtime (CPU).

WHY: Qualcomm AI Hub's profiler (and some ORT shape-inference paths) load the
model host-side and cannot resolve EXTERNAL tensors. A two-file model therefore
fails with:

    [ShapeInferenceError] Cannot parse data from external tensors.
    Please load external data into raw data for tensor: /model/Constant_5_output_0

Inlining pulls the weights from the .data file back into the protobuf, giving a
single file AI Hub can load. The embedded QNN context binary is preserved, so
the inlined model is still the compiled NPU model.

Usage:
    python inline_onnx.py act_so101_onnx_qnn.onnx act_so101_onnx_qnn_inline.onnx

The source .onnx and its sibling .data file must be in the same directory
(that is how ONNX references external data).

NOTE on size: this single-file trick works because ACT inlines to ~170 MB,
well under the 2 GB protobuf limit. A full SmolVLA will likely exceed 2 GB when
inlined, which is one of the reasons to export it in pieces instead.
"""

import argparse
import sys

import onnx
from onnx.external_data_helper import convert_model_from_external_data


def inline_external_data(src: str, dst: str) -> None:
    print(f"Loading (with external data): {src}")
    model = onnx.load(src, load_external_data=True)

    before = sum(
        1 for t in model.graph.initializer
        if t.data_location == onnx.TensorProto.EXTERNAL
    )
    print(f"External initializers before: {before}")

    convert_model_from_external_data(model)

    # save_as_external_data=False forces all weights into the single protobuf.
    onnx.save_model(model, dst, save_as_external_data=False)

    # Re-load and confirm nothing is still external.
    check = onnx.load(dst, load_external_data=True)
    after = sum(
        1 for t in check.graph.initializer
        if t.data_location == onnx.TensorProto.EXTERNAL
    )
    print(f"External initializers after:  {after}")
    if after != 0:
        print("FAILED: external tensors remain after inlining.", file=sys.stderr)
        sys.exit(1)

    onnx.checker.check_model(dst)
    print(f"ONNX checker: PASS")
    print(f"Wrote self-contained model: {dst}")


def verify_cpu(path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("(onnxruntime not installed here; skipping CPU load check)")
        return
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("\nCPU ORT load: OK")
    print("Inputs:")
    for i in sess.get_inputs():
        print(f"  {i.name:14s} {i.shape} {i.type}")
    print("Outputs:")
    for o in sess.get_outputs():
        print(f"  {o.name:14s} {o.shape} {o.type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="external-data .onnx (its .data sibling must be alongside)")
    ap.add_argument("dst", nargs="?", default=None,
                    help="output inline .onnx (default: <src stem>_inline.onnx)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the CPU ONNX Runtime load check")
    args = ap.parse_args()

    dst = args.dst
    if dst is None:
        dst = args.src[:-5] + "_inline.onnx" if args.src.endswith(".onnx") else args.src + "_inline.onnx"

    inline_external_data(args.src, dst)
    if not args.no_verify:
        verify_cpu(dst)


if __name__ == "__main__":
    main()
