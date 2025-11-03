# sim/nodes.py
import random, simpy
from lab_twin.sim.duration_scale import scale_service_time   # <-- ADD

class ProcessNode:
    def __init__(
        self,
        env: simpy.Environment,
        proc,
        log_event,
        *,
        resource: simpy.Resource | None = None,  # optional server
        is_head: bool = False,                   # first actionable step?
        is_tail: bool = False,                   # last step?
    ):
        self.env = env
        self.proc = proc
        self.in_store = simpy.Store(env)         # external/buffer queue for this step
        self.next: "ProcessNode|None" = None
        self.log = log_event
        self.resource = resource
        self.is_head = is_head
        self.is_tail = is_tail
        env.process(self._run())

    def connect(self, next_node: "ProcessNode|None"):
        self.next = next_node
        return next_node

    def _service_time(self) -> float:
        base = random.uniform(self.proc.duration_min_min, self.proc.duration_max_min)
        return scale_service_time(self.proc.process_id, base)  # <-- SCALE HERE

    def enqueue(self, batch):
        # arrival is when we enqueue at the head
        if self.is_head and getattr(batch, "arrival_sim_min", None) is None:
            batch.mark_arrival(self.env.now)

        # capture queue length *on arrival* (before the put)
        q_len_on_arrival = len(self.in_store.items)
        token = {
            "batch": batch,
            "enqueue_min": self.env.now,
            "q_len_on_arrival": q_len_on_arrival,
        }
        return self.in_store.put(token)

    def _service_time(self) -> float:
        return random.uniform(self.proc.duration_min_min, self.proc.duration_max_min)

    def _run(self):
        while True:
            token = yield self.in_store.get()
            batch = token["batch"]
            enqueue_min = token["enqueue_min"]
            q_len_on_arrival = token["q_len_on_arrival"]

            start = self.env.now
            # wait = time from enqueue to the moment service truly begins
            # (if a resource is used, this includes waiting in that resource’s queue)
            wait_min = self.env.now - enqueue_min

            try:
                # Acquire server if provided (single or multi-capacity)
                if self.resource is not None:
                    with self.resource.request() as req:
                        # If the server is busy, this yields until it becomes available
                        yield req
                        # now service starts
                        start = self.env.now
                        wait_min = start - enqueue_min
                        service = self._service_time()
                        yield self.env.timeout(service)
                else:
                    # No explicit server → single-server behavior via timeouts
                    service = self._service_time()
                    yield self.env.timeout(service)

                # Run side-effect for this process (e.g., assign barcodes)
                if self.proc.run:
                    self.proc.run(batch)

                status = "COMPLETED"

            except Exception as e:
                end = self.env.now
                self.log(
                    process_id=self.proc.process_id,
                    batch_id=batch.batch_id,
                    sim_start_min=start,
                    sim_end_min=end,
                    service_min=end - start,
                    wait_min=wait_min,
                    queue_len_on_arrival=q_len_on_arrival,
                    status="FAILED",
                    resource_name=(self.proc.resources[0] if self.proc.resources else self.proc.actor_role or ""),
                    note=str(e),
                )
                raise

            end = self.env.now

            # tail marks completion (so per-sample TAT uses true batch makespan)
            if self.is_tail:
                batch.mark_complete(end)

            # log
            self.log(
                process_id=self.proc.process_id,
                batch_id=batch.batch_id,
                sim_start_min=start,
                sim_end_min=end,
                service_min=end - start,
                wait_min=wait_min,
                queue_len_on_arrival=q_len_on_arrival,
                status=status,
                resource_name=(self.proc.resources[0] if self.proc.resources else self.proc.actor_role or ""),
                note="",
            )

            # pass downstream
            if self.next:
                yield self.next.enqueue(batch)


class DecisionNode:
    def __init__(self, env: simpy.Environment, name: str, condition, log_event):
        self.env = env
        self.name = name
        self.condition = condition
        self.log = log_event
        self.in_store = simpy.Store(env)
        self.branches = {}
        env.process(self._run())

    def connect_yes(self, node): self.branches["yes"] = node; return node
    def connect_no(self, node):  self.branches["no"]  = node; return node
    def connect_branch(self, name: str, node): self.branches[name] = node; return node

    def enqueue(self, batch):
        return self.in_store.put({"batch": batch, "enqueue_min": self.env.now})

    def _run(self):
        while True:
            token = yield self.in_store.get()
            batch = token["batch"]
            q_enter = token["enqueue_min"]
            q_len_on_arrival = len(self.in_store.items)
            wait = self.env.now - q_enter
            start = self.env.now

            outcome = self.condition(batch)
            branch = "yes" if isinstance(outcome, bool) and outcome else ("no" if isinstance(outcome, bool) else str(outcome))
            end = self.env.now

            self.log(
                process_id=f"decision:{self.name}",
                batch_id=batch.batch_id,
                sim_start_min=start,
                sim_end_min=end,
                service_min=0.0,
                wait_min=wait,
                queue_len_on_arrival=q_len_on_arrival,
                status=f"ROUTED:{branch}",
                resource_name="",
                note="",
            )

            nxt = self.branches.get(branch)
            if nxt is None:
                batch.mark_complete(end)
            else:
                yield nxt.enqueue(batch)
