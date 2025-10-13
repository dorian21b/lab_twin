# sim/arrivals.py
def launch_batches(env, head_node, batches, arrival_times):
    assert len(batches) == len(arrival_times)
    for batch, at in zip(batches, arrival_times):
        def _arrive(b=batch, t=at):
            yield env.timeout(t)
            if b.arrival_sim_min is None:
                b.mark_arrival(env.now)
            yield head_node.enqueue(b)
        env.process(_arrive())
