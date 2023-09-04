class tracker():

        def __init__(self) -> None:
                self.reward = 0
                self.iteration = 0
        
        def increment(self, reward):
                self.iteration += 1
                self.reward += reward

        def reset(self):
                self.reward = 0
                self.iteration = 0