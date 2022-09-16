from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.state import ObservableState, FullState


class Human(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.id = None

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius, self.belief)

    def get_next_observable_state(self, action, boundary=None):
        assert False

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
