import sys

sys.path.append("/kaggle_simulations/agent/")

import cicada.agent as agent  # noqa

agent.plans.models.load_lgb_models()
agent_obj = agent.Agent(noise_sd=0)


def action(obs):
    return agent_obj.action_wrapper(obs)
