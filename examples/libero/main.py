import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
LIBERO_ACTION_CLIP = 1.0
LIBERO_POSITION_GAIN = 4.0
LIBERO_POSE_INPUT_MIN = -np.ones(6, dtype=np.float32)
LIBERO_POSE_INPUT_MAX = np.ones(6, dtype=np.float32)
LIBERO_POSE_OUTPUT_MIN = -np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32)
LIBERO_POSE_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float32)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image.
                    # Rotation is disabled here to match the raw single-camera training data.
                    # img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    # wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = np.ascontiguousarray(obs["agentview_image"])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            # Temporary workaround for zero-state checkpoints: keep the incoming
                            # state shape consistent with the 7D training stats.
                            "observation/state": np.zeros(7, dtype=np.float32),
                            "prompt": str(task_description),
                        }

                        # Query model to get action and log policy inference time.
                        infer_result = client.infer(element)
                        infer_ms = infer_result.get("policy_timing", {}).get("infer_ms")
                        if infer_ms is not None:
                            logging.info(f"pi0 policy infer time: {infer_ms:.2f} ms")
                        action_chunk = infer_result["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = _action_to_libero(action_plan.popleft())

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _inverse_scale_pose_delta(
    pose_delta,
    *,
    input_min=LIBERO_POSE_INPUT_MIN,
    input_max=LIBERO_POSE_INPUT_MAX,
    output_min=LIBERO_POSE_OUTPUT_MIN,
    output_max=LIBERO_POSE_OUTPUT_MAX,
    action_clip=LIBERO_ACTION_CLIP,
):
    pose_delta = np.asarray(pose_delta, dtype=np.float32)
    denom = output_max - output_min
    safe = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    scaled = (pose_delta - output_min) / safe
    cmd = scaled * (input_max - input_min) + input_min
    cmd = np.clip(cmd, input_min, input_max)
    cmd = np.clip(cmd, -action_clip, action_clip)
    return cmd.astype(np.float32)


def _action_to_libero(action):
    """Convert model-predicted state deltas into LIBERO OSC_POSE controller commands."""
    action = np.asarray(action, dtype=np.float32).copy()
    dpos = action[:3] * LIBERO_POSITION_GAIN
    deuler = action[3:6]
    dwidth = float(action[6])

    # The dataset stores relative rotation as Euler XYZ deltas; robosuite OSC_POSE expects a rotvec.
    drotvec = R.from_euler("xyz", deuler).as_rotvec().astype(np.float32)
    pose_delta = np.concatenate([dpos, drotvec], axis=0).astype(np.float32)
    cmd_pose = _inverse_scale_pose_delta(pose_delta)

    # Dataset gripper action is delta finger width: positive opens, negative closes.
    if abs(dwidth) < 1e-6:
        cmd_gripper = 0.0
    else:
        cmd_gripper = float(-np.sign(dwidth))

    cmd = np.concatenate([cmd_pose, np.array([cmd_gripper], dtype=np.float32)], axis=0)
    if np.any(np.isnan(cmd)):
        raise ValueError(f"NaN in converted action. input={action}, converted={cmd}")
    return cmd


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
