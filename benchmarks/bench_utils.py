import statistics
from collections import defaultdict
from triton import runtime
from triton.testing import _summarize_statistics


def do_bench_count(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # interpret ``warmup`` and ``rep`` as iteration counts directly
    n_warmup = warmup
    n_repeat = rep
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_interleaved(
    fn_dict: dict,
    passes: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    # each function's median time in ms, over passes that
    # vary the order, as clocks follow the previous load
    names = list(fn_dict)
    # passes cycle through every rotation of the names,
    # forward then reversed (ABC, CBA, BCA, ACB, ...)
    orders = []
    for shift in range(len(names)):
        rotated = names[shift:] + names[:shift]
        orders.append(rotated)
        orders.append(rotated[::-1])

    times = defaultdict(list)
    for pass_index in range(passes):
        order = orders[pass_index % len(orders)]
        for name in order:
            median_ms = do_bench_count(
                fn=fn_dict[name],
                warmup=warmup,
                rep=repeats,
                return_mode="median",
            )
            times[name].append(median_ms)

    medians = {}
    for name, values in times.items():
        medians[name] = statistics.median(values)
    return medians
