import ray
from ray import tune
from ray.tune.registry import register_env
import sys
import os
from src.can_picking_env import CanPickingEnv

head_ip = None
if len(sys.argv) > 1:
    head_node = sys.argv[1]
    head_ip = os.popen("host " + head_node + " | awk '{print $4}'").read()

if not ray.is_initialized():
    if head_ip is not None:
        ray.init(redis_address=head_ip + ":6379")
    else:
        ray.init()

register_env("can_picking", lambda env_config: CanPickingEnv(env_config))

tune.run(
    "PPO",
    name="melanies_robbie",
    config={
        "env": "can_picking",
        "num_workers": 2,
        "sample_batch_size": 64,
        "train_batch_size": 4096,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 10,
        "observation_filter": "MeanStdFilter",
        "batch_mode": "complete_episodes",
        "env_config": {
            "num_robots": 1,
        },
        "model": {
            "use_lstm": True,
            "lstm_cell_size": 64,
            "fcnet_hiddens": [64, 64],
        },
    },
    stop={
        "time_total_s": 60 * 60 * 24
    },
    checkpoint_at_end=True,
    checkpoint_freq=10,
    queue_trials=True,
)
