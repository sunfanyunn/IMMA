from gym.envs.registration import register

register(
    id='MFCrowdSim-v0',
    entry_point='crowd_sim.envs:MFCrowdSim',
)
register(
    id='POCrowdSim-v0',
    entry_point='crowd_sim.envs:POCrowdSim',
)
register(
    id='MYCrowdSim-v0',
    entry_point='crowd_sim.envs:MYCrowdSim',
)
register(
    id='MFMACrowdSim-v0',
    entry_point='crowd_sim.envs:MFMACrowdSim',
)
