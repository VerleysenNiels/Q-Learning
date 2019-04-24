from collections import namedtuple
import copy

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class Buffer(object):
    """ When the player dies in beamrider, the game waits 80 frames before subtracting a live
        This class saves these 80 transitions to make sure the agent does not train on them
    """
    def __init__(self):
        self.capacity = 83
        self.position = 0  # Index of the oldest transition in memory
        self.memory = []

    def push(self, *args):
        """Saves a new transition and returns the eldest transition in memory"""
        trans = None
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            trans = self.memory[self.position]
            self.memory[self.position] = Transition(*args)
            self.position += 1
            if self.position == self.capacity:
                self.position = 0
        return trans

    def deadTrans(self):
        """ Return transition in which the player actually dies, clear the buffer """
        trans = copy.deepcopy(self.memory[self.position])
        self.memory.clear()
        self.position = 0
        return trans

    def __len__(self):
        return len(self.memory)
