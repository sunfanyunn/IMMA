from crowd_nav.configs.icra_benchmark.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        self.sim.human_num = 5
        self.sim.centralized_planning = True
        self.humans.policy = 'lforca'
        self.humans.v_pref = 1
        self.sim.circle_radius = 8


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'multistep_model_predictive_rl'
        self.action_space.speed_samples = 3
        self.action_space.rotation_samples = 4

        # gcn
        self.gcn.num_layer = 2
        self.gcn.X_dim = 32
        self.gcn.similarity_function = 'embedded_gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = True

        self.model_predictive_rl = Config()
        self.model_predictive_rl.linear_state_predictor = False
        self.model_predictive_rl.planning_depth = 1
        self.model_predictive_rl.planning_width = 1
        self.model_predictive_rl.do_action_clip = False
        self.model_predictive_rl.motion_predictor_dims = [64, 6]
        self.model_predictive_rl.value_network_dims = [32, 100, 100, 1]
        self.model_predictive_rl.share_graph_model = False


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)

        self.train.freeze_state_predictor = False
        self.train.detach_state_predictor = False
        self.train.reduce_sp_update_frequency = False
