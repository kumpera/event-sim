import json
import random;


# N_ACTIONS = 20
# N_BATCHES = 10
# N_DECISIONS = 50
# N_ACTIONS_PER_DECISION = 3
# N_ADF = 20

N_ACTIONS = 600
N_BATCHES = 1
N_DECISIONS = 2000
N_ACTIONS_PER_DECISION = 15
N_ADF = 30

def gen_action():
    res = dict()
    for i in range(0, N_ADF):
        res[f'f_{i}'] = random.random()
    return res


all_actions = []
for i in range(0, N_ACTIONS):
    action = gen_action()
    # hack to ease the compression prototype
    action["_idx"] = i
    all_actions.append(action)

def gen_decision():
    decision = dict()
    c = dict()
    decision["Version"] = "1"
    decision["c"] = c

    shared_ctx = dict()
    for i in range(0, 30):
        shared_ctx[f'c_{i}'] = random.random()
    c["TShared"] = shared_ctx
    # {
    #     "a": 10,
    #     "b=0": 1,
    #     "c": "d"
    # }
    
    actions = set()
    action_list = []
    while len(actions) < N_ACTIONS_PER_DECISION:
        idx = random.randint(0, N_ACTIONS - 1)
        if not (idx in actions):
            actions.add(idx)
            action_list.append(all_actions[idx])

    c["_multi"] = action_list
    return json.dumps(decision)

for i in range(0, N_BATCHES):
    with open(f'batch_{i}.in', 'w+') as res:
        for j in range(0, N_DECISIONS):
            res.write(gen_decision())
            res.write( "\n")
