class State():
    # list of string
    def __init__(self, state):
        # input is list of string
        self.state = state
        self.id = ', '.join(self.state)
    def __eq__(self, other):
        return self.state == other.state
    def __len__(self):
        return len(self.state)
    def get_id(self):
        return self.id
    def get_state(self):
        return self.state