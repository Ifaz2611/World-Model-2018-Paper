import numpy as np
import PIL.Image as Image
import gymnasium as gym

# gymnasium 1.x only registers v3; the v0 id from the paper lives in the old gym package.
ENV_ID = "CarRacing-v3"

FRAME_SIZE = 64  # the env renders 96x96, the V model takes 64x64


def make_env(render_mode="rgb_array"):
    return gym.make(ENV_ID, render_mode=render_mode)


def preprocess(frame):
    # frame shape: (96, 96, 3) uint8 -> (64, 64, 3) uint8. It stays uint8 so the saved
    # rollouts are a quarter the size; the /255 happens in the dataset.
    resized = Image.fromarray(frame).resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)

    return np.asarray(resized)


def random_policy(obs=None):
    # A uniform sample over the action box brakes about as often as it accelerates, so
    # the car mostly sits still and the frames never show any track. Gas is biased up
    # and the brake only fires occasionally.
    steer = np.random.uniform(-1.0, 1.0)
    gas = np.random.uniform(0.0, 1.0)
    brake = np.random.uniform(0.0, 0.5) if (np.random.rand() < 0.1) else 0.0

    return np.array([steer, gas, brake], dtype=np.float32)


def rollout(env, num_steps, seed=None, policy=None, skip=0, action_repeat=1):
    obs, info = env.reset(seed=seed)

    # obs shape: (96, 96, 3) uint8 -- frames come back raw and whoever wants the V model
    # input calls preprocess on them.
    frames = []
    actions = []
    action = None

    for step in range(skip + num_steps):
        # Holding an action for a few frames keeps the noise temporally correlated, which
        # is what actually drives the car down the track instead of jittering in place.
        if (step % action_repeat == 0):
            action = env.action_space.sample() if (policy is None) else policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if (step >= skip):
            frames.append(obs)
            actions.append(action)

        if (terminated or truncated):
            break

    return frames, actions
