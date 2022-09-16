from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.state import ObservableState, FullState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.NAN = FullState(0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.full_states = []

    def act(self, obs):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        obs, graph = obs[0], obs[1]
        full_state = self.get_full_state()
        self.full_states.append(full_state)

        self.policy.obs_frames = obs_frames = len(obs)

        states = []
        for i in range(obs_frames):
            # observation 
            idx = len(self.full_states)-obs_frames+i
            if idx < 0:
                states.append(JointState(self.NAN, obs[i]))
            else:
                states.append(JointState(self.full_states[idx], obs[i]))

        action = self.policy.predict((states, graph))

        return action
