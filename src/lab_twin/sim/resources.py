import simpy

def make_resources(env: simpy.Environment):
    return {
        "Band 7": simpy.Resource(env, capacity=1),
        "Band 5": simpy.Resource(env, capacity=1),
        "LIMS:GW": simpy.Resource(env, capacity=1),
        "Hamilton Star": simpy.Resource(env, capacity=1),
        "Plate reader": simpy.Resource(env, capacity=1),
        "LVL Capper/Decapper": simpy.Resource(env, capacity=1),
        "Scanner": simpy.Resource(env, capacity=1),
        "Fridge": simpy.Resource(env, capacity=1),
        "Freezer": simpy.Resource(env, capacity=1),
    }
