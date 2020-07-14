from collections import namedtuple

TransitionTuple = namedtuple("TransitionTuple", ["state", "action", "next_state", "reward", "done"])