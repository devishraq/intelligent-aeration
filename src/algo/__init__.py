from .sac import get_agent as get_sac_agent
from .td3 import get_agent as get_td3_agent
from .ddpg import get_agent as get_ddpg_agent
from .catd3 import get_agent as get_catd3_agent
from .catd3 import get_agent_cascade_only as get_catd3_cascade_agent
from .catd3 import get_agent_lagrangian_only as get_catd3_lagrangian_agent

ALGO_MAP = {
    "sac": get_sac_agent,
    "td3": get_td3_agent,
    "ddpg": get_ddpg_agent,
    "catd3": get_catd3_agent,
    "catd3_cascade": get_catd3_cascade_agent,
    "catd3_lagrangian": get_catd3_lagrangian_agent,
}


def build_sb3_agent(method, env, seed=42, verbose=0, tensorboard_log=None, device="cpu", **kwargs):
    method_key = method.lower().strip()

    return ALGO_MAP[method_key](env, seed=seed, tensorboard_log=tensorboard_log, verbose=verbose, device=device, **kwargs)
