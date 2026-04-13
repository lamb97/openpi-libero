"""
Convert a padded local LIBERO dataset into LeRobot format for OpenPI fine-tuning.

Expected input directory layout:

    libero_raw_delta_xyz_base_224/
      metadata.json
      states.npy          # shape: [num_episodes, max_seq_len, 7]
      actions.npy         # shape: [num_episodes, max_seq_len, 7]
      seq_lengths.npy     # shape: [num_episodes]
      obses_npy/
        episode_000000.npy  # shape: [seq_len, H, W, 3]
        ...

Usage:
    uv run examples/libero/convert_libero_raw_delta_xyz_to_lerobot.py \
        --data-dir /home/yang/dataset/libero_raw_delta_xyz_base_224 \
        --repo-id your_hf_username/libero_raw_delta_xyz_base_224
"""

import json
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def _load_tasks(metadata_path: Path) -> list[str]:
    metadata = json.loads(metadata_path.read_text())
    episodes = metadata["converted_index_map"]
    tasks = [""] * len(episodes)
    for episode in episodes:
        tasks[episode["converted_index"]] = episode["task"]
    return tasks


def main(
    data_dir: str,
    *,
    repo_id: str = "your_hf_username/libero_raw_delta_xyz_base_224",
    fps: int = 10,
    robot_type: str = "libero",
    push_to_hub: bool = False,
):
    data_path = Path(data_dir)
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    states = np.load(data_path / "states.npy", mmap_mode="r")
    actions = np.load(data_path / "actions.npy", mmap_mode="r")
    seq_lengths = np.load(data_path / "seq_lengths.npy")
    tasks = _load_tasks(data_path / "metadata.json")

    image_example = np.load(data_path / "obses_npy" / "episode_000000.npy", mmap_mode="r")
    image_shape = tuple(image_example.shape[1:])

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (states.shape[-1],),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (actions.shape[-1],),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for episode_idx, seq_len in enumerate(seq_lengths):
        frames = np.load(data_path / "obses_npy" / f"episode_{episode_idx:06d}.npy", mmap_mode="r")
        task = tasks[episode_idx]
        for step_idx in range(int(seq_len)):
            dataset.add_frame(
                {
                    "image": frames[step_idx],
                    "state": states[episode_idx, step_idx],
                    "actions": actions[episode_idx, step_idx],
                    "task": task,
                }
            )
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "openpi", "single-camera"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
