import argparse;
import json
import random;


# N_ACTIONS = 20
# N_BATCHES = 10
# N_DECISIONS = 50
# N_ACTIONS_PER_DECISION = 3
# N_ADF = 20

default_config = {
    "actionCount": 1000, # number of distinct actions
    "decisionCount": 2000, #number of decisions in the log
    "actionsPerDecision": 15, #number of arms in each decision
    "featuresPerAction": 30, #how big is each action
    "sharedCount": 30, #how big is the shared context
}

def gen_action(config):
    res = dict()
    for i in range(0, config["featuresPerAction"]):
        res[f'f_{i}'] = random.random()
    return res

def gen_action_set(config):
    if config["actionCount"] == -1:
        return None
    all_actions = []
    for i in range(0, config["actionCount"]):
        all_actions.append(gen_action(config))
    return all_actions

def gen_decision(config, all_actions):
    decision = dict()
    c = dict()
    decision["Version"] = "1"
    decision["c"] = c

    shared_ctx = dict()
    for i in range(0, config["sharedCount"]):
        shared_ctx[f'c_{i}'] = random.random()
    c["TShared"] = shared_ctx
    
    actions = set()
    action_list = []
    while len(action_list) < config["actionsPerDecision"]:
        if all_actions == None:
            action_list.append(gen_action(config))
        else:
            idx = random.randint(0, len(all_actions) - 1)
            if not (idx in actions):
                actions.add(idx)
                action_list.append(all_actions[idx])

    c["_multi"] = action_list
    return json.dumps(decision)


def gen_log(name, config):
    all_actions = gen_action_set(config)
    with open(name, 'w+') as res:
        for j in range(0, config["decisionCount"]):
            res.write(gen_decision(config, all_actions))
            res.write( "\n")


parser = argparse.ArgumentParser(description="Log generation")
parser.add_argument('--name', help='Name to use for the logs')
parser.add_argument('--set1', help='Generate 100,200,600,1000 batch of actions', action='store_true')
parser.add_argument('--extreme', help='Generate batch with near to no action overlap', action='store_true')

args = parser.parse_args()


if args.set1:
    for actions in [100, 200, 600, 1000]:
        config = dict(default_config)
        config["actionCount"] = actions
        gen_log(f'{args.name}_{actions}.in', config)
if args.extreme:
        config = dict(default_config)
        config["actionCount"] = -1
        gen_log(f'{args.name}.in', config)
