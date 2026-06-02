"""Convert diffusion_policy-style UR5 zarr replay buffers into OpenPI's DinoWmNpyDataset format.

Each input zarr is expected to have the layout produced by the diffusion_policy
training pipeline:

    data/img       (T, H, W, 3) uint8
    data/state     (T, 7)       float32   ee_xyz(3) + ee_rpy(3) + gripper(1)
    data/action    (T, 7)       float32   delta_xyz(3) + delta_rpy(3) + gripper(1)
    meta/episode_ends                     int array of cumulative episode end indices

Multiple sources are concatenated into a single output directory; each source
contributes its own prompt (language instruction) which is preserved on a
per-episode basis via ``episode_meta.jsonl``.

Output layout (consumable by ``DinoWmNpyDataset``):
    states.npy                 (total_steps, state_dim) float32
    actions.npy                (total_steps, action_dim) float32
    seq_lengths.npy            (num_episodes,) int64
    obses_npy/episode_NNNNNN.npy  (seq_len, H, W, 3) uint8
    episode_meta.jsonl         one JSON line per episode (converted_index, task, source)
    metadata.json              summary metadata

Example:
    python scripts/convert_ur5_dp_zarr_to_dinowm_npy.py \\
        --output-dir /home/riftuser/datasets/ur5_dp_combined_dinowm_npy \\
        --source ur5_dp_0501_bowl    /home/riftuser/datasets/ur5_dp_0501_bowl/replay_buffer.zarr    "pick up the bowl" \\
        --source ur5_dp_0519_penCase /home/riftuser/datasets/ur5_dp_0519_penCase/replay_buffer.zarr "pick up the pen case" \\
        --source ur5_dp_0521_bowlBox /home/riftuser/datasets/ur5_dp_0521_bowlBox/replay_buffer.zarr "put the bowl in the box" \\
        --image-size 224
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Output directory for the merged DinoWmNpyDataset.",
    )
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "ZARR_PATH", "PROMPT"),
        required=True,
        help="Repeatable: name + path to replay_buffer.zarr + language prompt.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square resize for all images (e.g. 224). Default: keep source resolution.",
    )
    parser.add_argument(
        "--max-episodes-per-source",
        type=int,
        default=None,
        help="If set, only convert the first N episodes from each source (useful for dry runs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def _maybe_resize(image: np.ndarray, size: int | None) -> np.ndarray:
    if size is None or image.shape[:2] == (size, size):
        return image
    return np.asarray(
        Image.fromarray(image).resize((size, size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dir = output_dir / "obses_npy"
    obs_dir.mkdir(parents=True, exist_ok=True)

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    seq_lengths: list[int] = []
    meta_lines: list[dict] = []
    img_shape: tuple[int, int, int] | None = None
    state_dim: int | None = None
    action_dim: int | None = None

    converted_idx = 0
    for name, zarr_path, prompt in args.source:
        zpath = pathlib.Path(zarr_path).expanduser().resolve()
        if not zpath.exists():
            raise FileNotFoundError(f"Zarr path does not exist: {zpath}")

        store = zarr.open(str(zpath), mode="r")
        states = store["data/state"]
        actions = store["data/action"]
        imgs = store["data/img"]
        ep_ends = np.asarray(store["meta/episode_ends"][:], dtype=np.int64)

        if state_dim is None:
            state_dim = int(states.shape[1])
            action_dim = int(actions.shape[1])
        elif states.shape[1] != state_dim or actions.shape[1] != action_dim:
            raise ValueError(
                f"Inconsistent dims in {zpath}: state {states.shape}, action {actions.shape}, "
                f"expected state_dim={state_dim} action_dim={action_dim}"
            )

        starts = np.concatenate([[0], ep_ends[:-1]])
        episode_pairs = list(zip(starts, ep_ends))
        if args.max_episodes_per_source is not None:
            episode_pairs = episode_pairs[: args.max_episodes_per_source]
        for ep_local_idx, (s, e) in enumerate(
            tqdm(episode_pairs, desc=f"converting {name}", unit="ep")
        ):
            s_int, e_int = int(s), int(e)
            ep_states = np.asarray(states[s_int:e_int], dtype=np.float32)
            ep_actions = np.asarray(actions[s_int:e_int], dtype=np.float32)
            ep_imgs_raw = np.asarray(imgs[s_int:e_int])
            if ep_imgs_raw.dtype != np.uint8:
                ep_imgs_raw = ep_imgs_raw.astype(np.uint8)
            if args.image_size is not None:
                ep_imgs = np.stack(
                    [_maybe_resize(frame, args.image_size) for frame in ep_imgs_raw],
                    axis=0,
                )
            else:
                ep_imgs = ep_imgs_raw
            if img_shape is None:
                img_shape = tuple(ep_imgs.shape[1:])
            elif tuple(ep_imgs.shape[1:]) != img_shape:
                raise ValueError(
                    f"Inconsistent image shape: got {ep_imgs.shape[1:]}, expected {img_shape}"
                )

            seq_len = len(ep_states)
            if seq_len < 2:
                continue

            np.save(obs_dir / f"episode_{converted_idx:06d}.npy", ep_imgs)
            all_states.append(ep_states)
            all_actions.append(ep_actions)
            seq_lengths.append(seq_len)
            meta_lines.append(
                {
                    "converted_index": converted_idx,
                    "source": name,
                    "source_episode": ep_local_idx,
                    "task": prompt,
                    "converted_steps": seq_len,
                }
            )
            converted_idx += 1

    if not meta_lines:
        raise RuntimeError("No episodes were converted.")

    states_flat = np.concatenate(all_states, axis=0)
    actions_flat = np.concatenate(all_actions, axis=0)
    seq_lengths_arr = np.asarray(seq_lengths, dtype=np.int64)

    np.save(output_dir / "states.npy", states_flat)
    np.save(output_dir / "actions.npy", actions_flat)
    np.save(output_dir / "seq_lengths.npy", seq_lengths_arr)

    with (output_dir / "episode_meta.jsonl").open("w", encoding="utf-8") as f:
        for item in meta_lines:
            f.write(json.dumps(item) + "\n")

    metadata = {
        "format": "openpi_dinowm_npy",
        "sources": [
            {"name": name, "zarr": str(pathlib.Path(p).resolve()), "prompt": prompt}
            for name, p, prompt in args.source
        ],
        "max_episodes_per_source": args.max_episodes_per_source,
        "num_episodes": len(seq_lengths_arr),
        "total_steps": int(seq_lengths_arr.sum()),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_shape": list(img_shape) if img_shape is not None else None,
        "state_definition": "ee_xyz(3) + ee_rpy(3) + gripper(1) (UR5 ur5_dp zarr replay buffer)",
        "action_definition": "delta_xyz(3) + delta_rpy(3) + gripper(1) (UR5 ur5_dp zarr replay buffer)",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Converted {len(seq_lengths_arr)} episodes / {int(seq_lengths_arr.sum())} frames to {output_dir}")


if __name__ == "__main__":
    main()
