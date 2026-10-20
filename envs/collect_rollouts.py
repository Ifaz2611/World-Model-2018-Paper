from envs.carracing import make_env, rollout, random_policy, preprocess
import os
import time
import numpy as np

NUM_ROLLOUTS = 500
# A random policy wanders off the track and never finds its way back -- past ~100 steps
# the mean road content of a frame drops by a third and the rest is plain grass. Short
# rollouts spend the frame budget on more track layouts instead.
ROLLOUT_STEPS = 100
SKIP_STEPS = 50  # the camera zooms in over the first ~50 steps, so those frames are junk
ACTION_REPEAT = 5
OUT_DIR = "data/rollouts"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    start = time.time()

    for i in range(NUM_ROLLOUTS):
        # Each rollout gets its own seed so the track layout differs; the V model needs
        # to see more than one track to learn anything general.
        frames, actions = rollout(
            env,
            num_steps=ROLLOUT_STEPS,
            seed=i,
            policy=random_policy,
            skip=SKIP_STEPS,
            action_repeat=ACTION_REPEAT
        )

        frames = np.stack([preprocess(frame) for frame in frames])
        actions = np.stack(actions).astype(np.float32)

        np.savez_compressed(f"{OUT_DIR}/rollout_{i:04d}.npz", frames=frames, actions=actions)

        if (i % 10 == 0):
            elapsed = time.time() - start
            print(f"rollout {i}/{NUM_ROLLOUTS}  frames {frames.shape}  actions {actions.shape}  {elapsed:.0f}s")

    env.close()
    print(f"done  {NUM_ROLLOUTS} rollouts in {time.time() - start:.0f}s")

if (__name__ == "__main__"):
    main()
