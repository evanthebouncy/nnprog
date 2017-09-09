import numpy as np
import gym
import random

# Play pong.
#
# animate: bool (whether to render the environment)
def play_pong(animate):
    # Step 1: Set up game
    env = gym.make("Pong-v0")
    obs = env.reset()

    # Step 2: Play game
    our_paddle = None
    sys_paddle = None
    ball = None
    prev_our_paddle = None
    prev_sys_paddle = None
    prev_ball = None
    reward = 0
    while True:
        # Step 2a: Render environment and take a step
        if animate:
            env.render()

        # Step 2b: Get the action
        if our_paddle is None or sys_paddle is None or ball is None or prev_our_paddle is None or prev_sys_paddle is None or prev_ball is None:
            action = env.action_space.sample()
        else:
            # move towards center if ball moving in opposite direction
            if ball[0] - prev_ball[0] <= 0:
                target = 40
            else:
                target = float(ball[1] - prev_ball[1]) / float(ball[0] - prev_ball[0]) * float(80 - ball[0] - 9) + ball[1]
                # if target > 80 or target < 0: target = 40
                # target = ball[1]
            delta = our_paddle[1] - target
            if abs(our_paddle[0] - ball[0]) > 2:
              if delta >= 2:
                  action = 2
              elif delta <= -2:
                  action = 3
            else:
              action = 2 if ball[1] - prev_ball[1] > 0 else 3
              
            #action = 2 if our_paddle[1] - ball[1] >= 0 else 3

        # Step 2c: Update previous state
        prev_our_paddle = our_paddle
        prev_sys_paddle = sys_paddle
        prev_ball = ball

        r = 0#raw_input()
        if r == '2':
            action = 2
        elif r == '3':
            action = 3
        else:
            action = action

        # Step 2d: Take a step
        obs, rew, done, info = env.step(action)
        reward += rew

        # Step 2e: Get new state
        obs = preprocess(obs)
        our_paddle = get_our_paddle(obs)
        sys_paddle = get_sys_paddle(obs)
        ball = get_ball(obs)

        # Step 2f: Handle termination
        if done:
            print 'Reward: ', reward
            reward = 0.0
            env.reset()

# Preprocesses the given image:
# (1) remove the scoreboard
# (2) make it monochromatic
# (3) make the background black
#
# obs: Image
# return: Image
# Image = np.array([n_rows, n_cols])
def preprocess(obs):
    obs = obs[34:194]
    obs = obs[::2,::2,0]
    obs[obs == 144] = 0
    return obs.astype(np.float)

# Assumes that the pixels of the given value in the given image
# exactly form a rectangle (or else there are no pixels of that color).
# Returns the rectangle if it exists, or else None.
#
# val: int
# obs: Image
# return: None | Rectangle
# Image = np.array([n_rows, n_cols])
def _get_rectangle(obs, val):
    min_val = np.argmax(obs.ravel() == val)
    max_val = len(obs.ravel()) - np.argmax(np.flip(obs.ravel(), 0) == val) - 1
    x_pos = min_val % obs.shape[1]
    y_pos = min_val / obs.shape[1]
    x_len = (max_val % obs.shape[1]) - x_pos + 1
    y_len = (max_val / obs.shape[1]) - y_pos + 1
    return None if x_pos == 0 and y_pos == 0 and x_len == obs.shape[1] and y_len == obs.shape[0] else np.array([x_pos + x_len/2, y_pos + y_len/2])

# Retrieves the rectangle representing our paddle.
def get_our_paddle(obs):
    return _get_rectangle(obs, 92)

# Retrieves the rectangle representing the system's paddle.
def get_sys_paddle(obs):
    return _get_rectangle(obs, 213)

# Retrieves the rectangle representing the ball.
def get_ball(obs):
    return _get_rectangle(obs, 236)

# Animate Pong for a few frames
def main():
    play_pong(True)

if __name__ == '__main__':
    main()

