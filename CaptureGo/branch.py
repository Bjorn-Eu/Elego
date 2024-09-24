
class Branch():
    def __init__(self,prior):
        self.prior = prior
        self.visits = 0
        self.total_value = 0.0