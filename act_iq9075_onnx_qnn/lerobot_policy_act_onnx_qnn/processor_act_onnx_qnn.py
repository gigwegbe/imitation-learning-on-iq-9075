from lerobot.policies.factory import make_pre_post_processors
from .configuration_act_onnx_qnn import ACTONNXQNNConfig


def make_act_onnx_qnn_pre_post_processors(
    config: ACTONNXQNNConfig,
    pretrained_path: str | None = None,
    dataset_stats: dict | None = None,
    preprocessor_overrides: dict | None = None,
    postprocessor_overrides: dict | None = None,
):
    # Reuse LeRobot's standard pre/post processors. This is what pulls the
    # normalization stats from the checkpoint and de-normalizes the action
    # chunk, so you do NOT need to export separate normalization files.
    return make_pre_post_processors(
        config,
        pretrained_path=pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
