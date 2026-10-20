from envs.carracing import make_env, rollout
import matplotlib.pyplot as plt

NUM_FRAMES = 6
FRAME_EVERY = 20  # the camera zooms in over the first ~50 steps, so space the frames out
SEED = 0


def main():
    env = make_env(render_mode="rgb_array")

    frames, actions = rollout(env, num_steps=NUM_FRAMES * FRAME_EVERY, seed=SEED)
    env.close()

    print(f"{len(frames)} frames  shape {frames[0].shape}  dtype {frames[0].dtype}")
    print(f"action space {env.action_space}")

    shown = frames[::FRAME_EVERY][:NUM_FRAMES]

    fig, axes = plt.subplots(1, len(shown), figsize=(2 * len(shown), 2.5))
    for i, (ax, frame) in enumerate(zip(axes, shown)):
        ax.imshow(frame)
        ax.set_title(f"step {i * FRAME_EVERY}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if (__name__ == "__main__"):
    main()
