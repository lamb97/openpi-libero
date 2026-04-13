import dataclasses
import pathlib

import imageio.v2 as imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tyro


@dataclasses.dataclass
class Args:
    train_episode_path: str = "/home/yang/dataset/libero_raw_delta_xyz_base_224/obses_npy/episode_000000.npy"
    train_frame_idx: int = 0

    task_suite_name: str = "libero_10"
    task_id: int = 0
    init_state_idx: int = 0
    seed: int = 7
    env_resolution: int = 256

    output_dir: str = "data/libero/frame_compare"


def main(args: Args) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frames = np.load(args.train_episode_path, mmap_mode="r")
    train_frame = np.asarray(train_frames[args.train_frame_idx])

    eval_frame = _get_eval_frame(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        init_state_idx=args.init_state_idx,
        resolution=args.env_resolution,
        seed=args.seed,
    )

    eval_frame_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(eval_frame, train_frame.shape[0], train_frame.shape[1])
    )
    eval_frame_rot180_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(eval_frame[::-1, ::-1], train_frame.shape[0], train_frame.shape[1])
    )

    train_out = output_dir / "train_frame.png"
    eval_out = output_dir / "eval_frame.png"
    eval_rot_out = output_dir / "eval_frame_rot180.png"
    strip_out = output_dir / "compare_strip.png"

    imageio.imwrite(train_out, train_frame)
    imageio.imwrite(eval_out, eval_frame_resized)
    imageio.imwrite(eval_rot_out, eval_frame_rot180_resized)
    imageio.imwrite(strip_out, np.concatenate([train_frame, eval_frame_resized, eval_frame_rot180_resized], axis=1))

    print(f"Saved training frame to: {train_out}")
    print(f"Saved eval frame to: {eval_out}")
    print(f"Saved rotated eval frame to: {eval_rot_out}")
    print(f"Saved side-by-side strip to: {strip_out}")


def _get_eval_frame(*, task_suite_name: str, task_id: int, init_state_idx: int, resolution: int, seed: int) -> np.ndarray:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(initial_states[init_state_idx])
    frame = np.ascontiguousarray(obs["agentview_image"])
    env.close()
    return frame


if __name__ == "__main__":
    tyro.cli(main)
