"""REVISION_REPLICATION P0: synthetic CPU timing only; no scientific inputs."""
import copy
import io
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpmr_net.features import encode_b0_features
from bpmr_net.models import B0Network
from bpmr_net.fusion import template_average, orientation_average
from paper_revision.models_revision import build_revision_bpmr


def main():
    torch.manual_seed(8341)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    rows = []
    source = B0Network()
    models = {}
    for variant in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']:
        model = copy.deepcopy(source) if variant in ['A0', 'A1'] else build_revision_bpmr(copy.deepcopy(source), variant)
        models[variant] = model
        for batch in [256, 4096]:
            x = torch.rand(batch, 400)
            target = torch.randint(0, 4, (batch,))
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            samples = []
            for iteration in range(5):
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                result = model(x)
                probability = result / result.sum(1, keepdim=True) if variant in ['A0', 'A1'] else result[1]
                loss = F.mse_loss(probability, F.one_hot(target, 4).float()) if variant in ['A0', 'A2'] else F.nll_loss(probability.clamp_min(1e-12).log(), target)
                loss.backward()
                optimizer.step()
                elapsed = time.perf_counter() - start
                assert torch.isfinite(loss)
                peak = max(peak, process.memory_info().rss)
                if iteration >= 2:
                    samples.append(elapsed)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            load_samples = []
            for _ in range(5):
                buffer.seek(0)
                start = time.perf_counter()
                state = torch.load(buffer, map_location='cpu', weights_only=True)
                model.load_state_dict(state, strict=True)
                load_samples.append(time.perf_counter() - start)
            rows.append(dict(variant=variant, batch=batch, step_seconds=samples,
                             median_step_seconds=statistics.median(samples),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             checkpoint_bytes=buffer.getbuffer().nbytes,
                             median_warm_load_seconds=statistics.median(load_samples)))
    # Serial voxel proxy: encode 3 planes, run 4 phase models per plane, fuse and sample.
    # Reuses one synthetic model per variant: excludes dispatch, ROI I/O and real Gibbs scheduling.
    labels = np.random.default_rng(99).integers(0, 4, (9, 9), dtype=np.int16)
    support = np.ones((9, 9), bool)
    proxy = {}
    with torch.inference_mode():
        for variant in ['A0', 'A3']:
            samples = []
            for iteration in range(40):
                start = time.perf_counter()
                orientations = []
                for axis in range(3):
                    x = torch.from_numpy(encode_b0_features(labels, support, [4], [4]))
                    probabilities = []
                    for phase in range(4):
                        result = models[variant](x)
                        p = result / result.sum(1, keepdim=True) if variant == 'A0' else result[1]
                        probabilities.append(p.numpy()[0])
                    orientations.append(template_average(probabilities))
                p = orientation_average(orientations)
                np.random.default_rng(iteration).choice(4, p=p)
                if iteration >= 5:
                    samples.append(time.perf_counter() - start)
            proxy[variant] = dict(median_seconds_per_voxel=statistics.median(samples), samples_seconds=samples)
    output = dict(identity='REVISION_REPLICATION', scientific_evidence=False,
                  python=sys.version, torch=torch.__version__, numpy=np.__version__,
                  platform=platform.platform(), processor=platform.processor(), threads=4,
                  logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                  total_ram_bytes=psutil.virtual_memory().total,
                  rss_baseline_bytes=baseline, sampled_peak_rss_bytes=peak,
                  memory_limitations='Sampled process RSS, not OS peak or full source staging; allocator retention included.',
                  imports_passed=['numpy', 'torch', 'bpmr_net', 'paper_revision.models_revision'],
                  optimizer_steps=rows, voxel_proxy=proxy,
                  limitations=['Synthetic inputs only; not accuracy evidence.', 'Warm in-memory checkpoint loading excludes cold disk latency.',
                               'Voxel proxy excludes actual scheduler, feature-plane extraction from 3D, RNG persistence and metric evaluation.',
                               'Historical reconstruction script imports Unix resource and external modules; Windows runner must be implemented and verified before scientific use.'])
    destination = Path('outputs/revision_mtcomm_2026/R1_RECONSTITUTION_P0_CPU_BENCHMARK.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'voxel_proxy': {k:v['median_seconds_per_voxel'] for k,v in proxy.items()}, 'rss':peak}))


if __name__ == '__main__':
    main()
