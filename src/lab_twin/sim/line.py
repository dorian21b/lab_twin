# # sim/line.py
# from .nodes import ProcessNode

# def build_line(env, processes, log_event):
#     nodes = [ProcessNode(env, p, log_event) for p in processes if p.is_actionable]
#     for a, b in zip(nodes, nodes[1:]):
#         a.connect(b)
#     head = nodes[0] if nodes else None
#     tail = nodes[-1] if nodes else None
#     return head, tail, nodes

from lab_twin.sim.resources import make_resources
from lab_twin.sim.nodes import ProcessNode

def build_line(env, steps, log_event):
    res_map = make_resources(env)  # {name: simpy.Resource(...), ...}

    nodes = []
    actionable_idxs = [i for i, p in enumerate(steps) if p.is_actionable]
    head_i = actionable_idxs[0]
    tail_i = actionable_idxs[-1]

    for i, proc in enumerate(steps):
        rname = proc.resources[0] if proc.resources else (proc.actor_role or "")
        res = res_map.get(rname) if rname else None
        node = ProcessNode(
            env=env,
            proc=proc,
            log_event=log_event,
            resource=res,
            is_head=(i == head_i),
            is_tail=(i == tail_i),
        )
        nodes.append(node)

    for a, b in zip(nodes, nodes[1:]):
        a.connect(b)
    return nodes[0], nodes[-1], res_map
