"""Observation-only hooks for the hot-cache study.

Nothing here changes routing, cache contents or scheduling. Patches are applied right after the target modules
are imported, so they are in place before CUDA-graph capture.

QWEN38_RESEARCH_OUT        output directory inside the container (default /research/out)
QWEN38_RESEARCH_LRU=1      record every LRU call: per layer, the flattened topk_ids and the runtime miss count,
                           in a device ring buffer written inside the decode CUDA graphs
QWEN38_RESEARCH_LRU_RING   ring length in calls per layer (default 4096)
QWEN38_RESEARCH_STEPMEM=1  per step with more than 16 tokens: allocator peak, reserved, driver-free bytes
QWEN38_RESEARCH_MEMSNAP=N  record torch.cuda.memory history for the N-th step with at least
                           QWEN38_RESEARCH_MEMSNAP_MIN tokens (default 4096) and dump a snapshot after it

Dumps happen at the start of the next execute_model after `<out>/trigger` changes (its content is used as the
label), so a client writes the trigger after a request and sends a tiny request to flush.
"""
import importlib.abc
import importlib.util
import json
import os
import sys
import time

OUT = os.environ.get('QWEN38_RESEARCH_OUT', '/research/out')
LRU = os.environ.get('QWEN38_RESEARCH_LRU') == '1'
RING = int(os.environ.get('QWEN38_RESEARCH_LRU_RING', '4096'))
WIDTH = 40  # ids per call recorded; 4 tokens x top-10. Wider calls are flagged by their recorded count.
STEPMEM = os.environ.get('QWEN38_RESEARCH_STEPMEM') == '1'
MEMSNAP = int(os.environ.get('QWEN38_RESEARCH_MEMSNAP', '0'))
MEMSNAP_MIN = int(os.environ.get('QWEN38_RESEARCH_MEMSNAP_MIN', '4096'))

WNA16 = ('vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.'
         'compressed_tensors_moe_wna16')
RUNNER = 'vllm.v1.worker.gpu.model_runner'

_recorders = {}  # device index -> _Recorder


def _log(msg):
    print(f'[qwen38-research pid={os.getpid()}] {msg}', file=sys.stderr, flush=True)


class _Recorder:
    def __init__(self, torch, device):
        self.device = device
        self.ids = torch.full((48, RING, WIDTH), -1, dtype=torch.int16, device=device)
        self.meta = torch.zeros((48, RING, 2), dtype=torch.int32, device=device)  # (num_ids, misses)
        self.count = torch.zeros(48, dtype=torch.int64, device=device)
        self.scratch = torch.empty(2, dtype=torch.int32, device=device)
        self.last_count = None
        _log(f'LRU recorder on {device}: ring={RING} width={WIDTH} '
             f'bytes={self.ids.nbytes + self.meta.nbytes}')


def _layer_index(name):
    import re
    m = re.search(r'language_model\.model\.layers\.(\d+)\.', name or '')
    return int(m.group(1)) if m else None


def _patch_wna16(module):
    import torch
    cls = module.CompressedTensorsWNA16MoEMethod
    original = cls.apply

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        out = original(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
        cache = getattr(self, '_static_hot_cache', None)
        if cache is None or not cache.dynamic_lru:
            return out
        index = _layer_index(getattr(self, 'layer_name', ''))
        if index is None:
            return out
        device = x.device
        rec = _recorders.get(device.index)
        if rec is None:
            rec = _recorders[device.index] = _Recorder(torch, device)
        # Same conditions as the dynamic-LRU branch of CompressedTensorsWNA16MoEMethod.apply.
        num_ids = topk_ids.numel()
        if (x.shape[0] > self._static_hot_cache_max_tokens
                or getattr(self, '_cpu_experts_layer', None) is not None
                or num_ids > cache.miss_local_ids.numel() or num_ids > len(cache.global_ids)):
            return out
        pos = rec.count[index:index + 1] % RING
        row = torch.full((1, WIDTH), -1, dtype=torch.int16, device=device)
        k = min(num_ids, WIDTH)
        row[0, :k].copy_(topk_ids.reshape(-1)[:k])
        rec.ids[index].index_copy_(0, pos, row)
        rec.scratch[0].fill_(num_ids)
        rec.scratch[1].copy_((cache.miss_local_ids[:num_ids] >= 0).sum())
        rec.meta[index].index_copy_(0, pos, rec.scratch.view(1, 2))
        rec.count[index].add_(1)
        return out

    cls.apply = apply
    _log('patched CompressedTensorsWNA16MoEMethod.apply (LRU recorder)')


_step_events = []  # (cuda event recorded at the start of each LRU-sized step, scheduled tokens)


def _dump_lru(label):
    import numpy as np
    import torch
    torch.cuda.synchronize()
    events = list(_step_events)
    _step_events.clear()
    step_ms = np.array([a.elapsed_time(b) for (a, _), (b, _) in zip(events, events[1:])], dtype=np.float64)
    step_tokens = np.array([t for _, t in events], dtype=np.int32)
    if not _recorders:
        # No LRU in this runtime (e.g. hot-only): keep the step timing alone.
        path = os.path.join(OUT, f'steps-dev{torch.cuda.current_device()}-{label}.npz')
        np.savez_compressed(path, step_ms=step_ms, step_tokens=step_tokens)
        return
    for dev, rec in _recorders.items():
        count = rec.count.cpu().numpy()
        ids = rec.ids.cpu().numpy()
        meta = rec.meta.cpu().numpy()
        path = os.path.join(OUT, f'lru-dev{dev}-{label}.npz')
        # step_ms[i] is the GPU time from small step i to small step i+1 of this dump window (the last step of the
        # window has no successor). The window's small steps are the last len(step_tokens) LRU calls of layer 0.
        np.savez_compressed(path, count=count, ids=ids, meta=meta, ring=RING, width=WIDTH,
                            step_ms=step_ms, step_tokens=step_tokens,
                            last_count=rec.last_count if rec.last_count is not None else np.zeros_like(count))
        rec.last_count = count
        _log(f'dumped {path} calls={int(count.max())}')


def _patch_runner(module):
    import torch
    cls = module.GPUModelRunner
    original = cls.execute_model
    trigger = os.path.join(OUT, 'trigger')
    state = {'mtime': None, 'large_steps': 0, 'snap_armed': False}
    try:
        state['mtime'] = os.stat(trigger).st_mtime_ns
    except FileNotFoundError:
        pass
    stepmem = open(os.path.join(OUT, f'stepmem-{os.getpid()}.log'), 'a', buffering=1) if STEPMEM else None

    def execute_model(self, scheduler_output, *args, **kwargs):
        if kwargs.get('dummy_run'):
            return original(self, scheduler_output, *args, **kwargs)
        try:
            mtime = os.stat(trigger).st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if mtime != state['mtime']:
            state['mtime'] = mtime
            label = 'x'
            if mtime is not None:
                with open(trigger) as f:
                    label = f.read().strip() or 'x'
            if LRU:
                _dump_lru(label)
        tokens = getattr(scheduler_output, 'total_num_scheduled_tokens', 0) or 0
        large = tokens > 16
        snap = False
        if large:
            if tokens >= MEMSNAP_MIN:
                state['large_steps'] += 1
                if MEMSNAP and state['large_steps'] == MEMSNAP:
                    torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks='python')
                    snap = True
            if stepmem is not None:
                torch.cuda.reset_peak_memory_stats()
        if LRU and 0 < tokens <= 16:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            _step_events.append((ev, tokens))
        t0 = time.perf_counter()
        result = original(self, scheduler_output, *args, **kwargs)
        if large and (stepmem is not None or snap):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            if stepmem is not None:
                stepmem.write(json.dumps({
                    't': t0, 'host_s': time.perf_counter() - t0, 'tokens': tokens,
                    'device': torch.cuda.current_device(),
                    'peak_allocated': torch.cuda.max_memory_allocated(),
                    'peak_reserved': torch.cuda.max_memory_reserved(),
                    'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                    'driver_free': free, 'driver_total': total}) + '\n')
            if snap:
                path = os.path.join(OUT, f'memsnap-dev{torch.cuda.current_device()}-step{MEMSNAP}.pickle')
                torch.cuda.memory._dump_snapshot(path)
                torch.cuda.memory._record_memory_history(enabled=None)
                _log(f'memory snapshot {path} (tokens={tokens}, driver_free={free})')
        return result

    cls.execute_model = execute_model
    _log('patched GPUModelRunner.execute_model')


_PATCHES = {}
if LRU:
    _PATCHES[WNA16] = _patch_wna16
if LRU or STEPMEM or MEMSNAP:
    _PATCHES[RUNNER] = _patch_runner


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def patched_exec(module):
            exec_module(module)
            _PATCHES[name](module)

        loader.exec_module = patched_exec
        return spec


def install():
    if not _PATCHES:
        return
    os.makedirs(OUT, exist_ok=True)
    sys.meta_path.insert(0, _Finder())
