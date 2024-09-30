import numpy as np
class ZExperienceCollector():
    def __init__(self):
        self.gamestates = []
        self.visit_counts = []
        self.rewards = []

        self.current_gamestates = []
        self.current_visit_counts = []

    
    def start_episode(self):
        self.current_gamestates = []
        self.current_visit_counts = []

    def record_data(self,gamestate,visit_count):
        (self.current_gamestates).append(gamestate)
        (self.current_visit_counts).append(visit_count)

    def end_episode(self,reward):
        episode_length = len(self.current_gamestates)
        self.gamestates += self.current_gamestates
        self.visit_counts += self.current_visit_counts
        self.rewards += [reward for _ in range(episode_length)]
        self.start_episode()

    def to_data(self):
        return ZExperienceData(self.gamestates,self.visit_counts,self.rewards)

class ZExperienceData():
    def __init__(self,gamestates=None,visit_counts=None,rewards=None):
        self.gamestates = gamestates
        self.visit_counts = visit_counts
        self.rewards = rewards
    
    def load(self,file):
        data = np.load(file)
        self.gamestates = data['arr_0']
        self.visit_counts = data['arr_1']
        self.rewards = data['arr_2']


    def write(self,file):
        np.savez(file,self.gamestates,self.visit_counts,self.rewards)


