from collections import namedtuple

TransitionTuple = namedtuple("TransitionTuple", ["state", "action", "next_state", "reward", "done"])

def average_weights(target_sd, moving_sd, tau):
    for key in target_sd:
        target_sd[key] = tau * target_sd[key] + (1-tau) * moving_sd[key]
    return target_sd