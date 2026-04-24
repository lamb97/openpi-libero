"""Convert diffusion_policy demo_xyz_image demos into OpenPI's local DinoWmNpyDataset format.

This mirrors the current diffusion_policy training logic for
`train_diffusion_unet_demo_xyz_image_workspace`:

- state: `ee_pose_mm_rad[:3]`
- action: forward delta `eef_xyz[t + 1] - eef_xyz[t]`
- image: the step PNG aligned with the state at time `t`

Because delta actions require a next state, the final raw step in each demo is
dropped from the converted dataset.

Example:
    python scripts/convert_demo_xyz_to_dinowm_npy.py \
        --input-dir /home/yang/demos \
        --output-dir /home/yang/demo_xyz_dinowm_npy \
        --exclude-first-n-demos 4 \
        --prompt pick_place
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from dataclasses import dataclass

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class EpisodeData:
    source_demo: str
    states: np.ndarray
    actions: np.ndarray
    images: np.ndarray
    raw_steps: int


@dataclass(frozen=True)
class EpisodeSpec:
    demo_dir: pathlib.Path
    source_demo: str
    raw_steps: int
    converted_steps: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing demo_* folders with steps/*.npz and steps/*.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Output directory for DinoWmNpyDataset files.",
    )
    parser.add_argument(
        "--exclude-first-n-demos",
        type=int,
        default=4,
        help="Skip the first N sorted demo_* folders to mirror the diffusion_policy config.",
    )
    parser.add_argument(
        "--demo-prefix",
        type=str,
        default="demo_",
        help="Only convert folders whose names start with this prefix.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="pick_place",
        help="Task/prompt string written to episode metadata.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=224,
        help="Output image height.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=224,
        help="Output image width.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def _load_image(path: pathlib.Path, *, out_size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(iio.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim != 3:
        raise ValueError(f"Unsupported image rank for {path}: {image.shape}")

    channels = image.shape[-1]
    if channels == 1:
        image = np.repeat(image, 3, axis=-1)
    elif channels == 4:
        image = image[..., :3]
    elif channels != 3:
        raise ValueError(f"Unsupported channel count for {path}: {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[:2] != out_size:
        image = np.asarray(
            Image.fromarray(image).resize((out_size[1], out_size[0]), resample=Image.BILINEAR),
            dtype=np.uint8,
        )

    return image


def _convert_demo(demo_dir: pathlib.Path, *, out_size: tuple[int, int]) -> EpisodeData | None:
    steps_dir = demo_dir / "steps"
    step_npz_paths = sorted(steps_dir.glob("*.npz"))
    if not step_npz_paths:
        return None

    eef_xyz = []
    image_paths = []
    for npz_path in step_npz_paths:
        png_path = npz_path.with_suffix(".png")
        if not png_path.is_file():
            raise FileNotFoundError(f"Missing image for step file: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as step_data:
            if "ee_pose_mm_rad" not in step_data:
                raise KeyError(f"Missing `ee_pose_mm_rad` in {npz_path}")
            eef_xyz.append(np.asarray(step_data["ee_pose_mm_rad"][:3], dtype=np.float32))
        image_paths.append(png_path)

    if len(eef_xyz) < 2:
        return None

    eef_xyz = np.stack(eef_xyz, axis=0)
    states = eef_xyz[:-1]
    actions = np.diff(eef_xyz, axis=0)

    aligned_image_paths = image_paths[:-1]
    images = np.stack([_load_image(path, out_size=out_size) for path in aligned_image_paths], axis=0)

    if not (len(states) == len(actions) == len(images)):
        raise RuntimeError(
            f"Length mismatch for {demo_dir.name}: "
            f"{len(states)=}, {len(actions)=}, {len(images)=}"
        )

    return EpisodeData(
        source_demo=demo_dir.name,
        states=states.astype(np.float32),
        actions=actions.astype(np.float32),
        images=images,
        raw_steps=len(step_npz_paths),
    )


def _scan_demo(demo_dir: pathlib.Path) -> EpisodeSpec | None:
    steps_dir = demo_dir / "steps"
    step_npz_paths = sorted(steps_dir.glob("*.npz"))
    if not step_npz_paths:
        return None

    valid_steps = 0
    for npz_path in step_npz_paths:
        png_path = npz_path.with_suffix(".png")
        if not png_path.is_file():
            raise FileNotFoundError(f"Missing image for step file: {npz_path}")
        valid_steps += 1

    if valid_steps < 2:
        return None

    return EpisodeSpec(
        demo_dir=demo_dir,
        source_demo=demo_dir.name,
        raw_steps=valid_steps,
        converted_steps=valid_steps - 1,
    )


def main() -> None:
    args = _parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_dirs = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_dir() and path.name.startswith(args.demo_prefix)
    )
    demo_dirs = demo_dirs[int(args.exclude_first_n_demos) :]
    if not demo_dirs:
        raise RuntimeError("No demo directories left after exclusion.")

    episode_specs = []
    skipped = 0
    for demo_dir in demo_dirs:
        episode_spec = _scan_demo(demo_dir)
        if episode_spec is None:
            skipped += 1
            continue
        episode_specs.append(episode_spec)

    if not episode_specs:
        raise RuntimeError("No valid demos were converted.")

    seq_lengths = np.asarray([spec.converted_steps for spec in episode_specs], dtype=np.int64)
    max_seq_len = int(seq_lengths.max())
    state_dim = 3
    action_dim = 3
    states = np.zeros((len(episode_specs), max_seq_len, state_dim), dtype=np.float32)
    actions = np.zeros((len(episode_specs), max_seq_len, action_dim), dtype=np.float32)

    obs_dir = output_dir / "obses_npy"
    obs_dir.mkdir(parents=True, exist_ok=True)

    converted_index_map = []
    for episode_idx, episode_spec in enumerate(tqdm(episode_specs, desc="Converting demos", unit="demo")):
        episode = _convert_demo(
            episode_spec.demo_dir,
            out_size=(args.image_height, args.image_width),
        )
        if episode is None:
            raise RuntimeError(f"Demo became invalid during conversion: {episode_spec.demo_dir}")

        seq_len = len(episode.states)
        states[episode_idx, :seq_len] = episode.states
        actions[episode_idx, :seq_len] = episode.actions
        np.save(obs_dir / f"episode_{episode_idx:06d}.npy", episode.images)
        converted_index_map.append(
            {
                "converted_index": episode_idx,
                "source_demo": episode.source_demo,
                "task": args.prompt,
                "raw_steps": episode.raw_steps,
                "converted_steps": seq_len,
            }
        )

    np.save(output_dir / "states.npy", states)
    np.save(output_dir / "actions.npy", actions)
    np.save(output_dir / "seq_lengths.npy", seq_lengths)

    with (output_dir / "episode_meta.jsonl").open("w", encoding="utf-8") as f:
        for item in converted_index_map:
            f.write(json.dumps(item) + "\n")

    metadata = {
        "format": "openpi_dinowm_npy",
        "input_dir": str(input_dir.resolve()),
        "num_episodes": len(episode_specs),
        "exclude_first_n_demos": args.exclude_first_n_demos,
        "state_definition": "ee_pose_mm_rad[:3]",
        "action_definition": "np.diff(ee_pose_mm_rad[:3], axis=0)",
        "image_definition": "steps/*.png aligned with states; resized during conversion; final raw frame dropped",
        "image_shape": [args.image_height, args.image_width, 3],
        "prompt": args.prompt,
        "converted_index_map": converted_index_map,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_steps = int(seq_lengths.sum())
    print(f"Converted {len(episode_specs)} episodes to {output_dir}")
    print(f"Skipped {skipped} demos with fewer than 2 valid steps")
    print(f"Total converted frames: {total_steps}")


if __name__ == "__main__":
    main()
