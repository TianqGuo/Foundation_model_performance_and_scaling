"""Validate matched Stage 3 runs and write a descriptive summary, no winner claim.

Usage: python -m cs336_alignment.section7_grpo.compare_runs RESULTS_FOLDER
No model imports; checkpoints themselves are not loaded.
"""
import argparse
import json
import math
from pathlib import Path
from cs336_alignment.section7_grpo.run_utils import validate_frozen_policy_agreement


def summarize(base, allow_clipping_difference=False):
    summary={}; configs={}; subsets={}
    for method in ('grpo','gspo'):
        runs=list(base.glob(f'stage3_{method}_*'))
        if len(runs)!=1: raise ValueError(f'Expected exactly one {method} run')
        run=runs[0]
        config=json.loads((run/'run_config.json').read_text()); configs[method]=config
        subsets[method]=json.loads((run/'evaluation_subset.json').read_text())
        def rows(name):return [json.loads(s) for s in (run/name).read_text().splitlines() if s.strip()]
        metrics=rows('system_metrics.jsonl')
        if len(metrics)!=config['args']['n_grpo_steps']: raise ValueError('Incomplete training run')
        if [m['grpo_step'] for m in metrics]!=list(range(1,len(metrics)+1)): raise ValueError('Training step order mismatch')
        if config['estimator']!={'baseline':'mean','advantage_normalizer':'std','importance_reweighting':method,'loss_normalization':'sequence'}:
            raise ValueError('Unexpected comparison estimator')
        if not json.loads((run/'server_shutdown.json').read_text())['stopped']: raise ValueError('Server not stopped')
        if not all(math.isfinite(v) for m in metrics for v in m['grad_norms']): raise ValueError('Nonfinite gradients')
        final=json.loads((run/'final_eval.json').read_text())
        evaluation=rows(f"eval_metrics_{config['args']['run_name']}.jsonl")
        expected=[s for s in range(1,len(metrics)+1) if s%config['args']['eval_interval']==0]
        if [m['grpo_step'] for m in evaluation]!=expected: raise ValueError('Evaluation schedule mismatch')
        checkpoint=Path(config['checkpoint'])
        # Validate saved files on cloud; never load the model.
        if not (checkpoint/'config.json').exists(): raise ValueError('Missing checkpoint configuration')
        if not list(checkpoint.glob('*.safetensors')): raise ValueError('Missing checkpoint weights')
        agreement = [d for m in metrics for d in m.get('policy_agreement_before_first_update', [])]
        if config['args'].get('log_prob_precision') == 'fp32':
            if any(not m.get('policy_agreement_before_first_update') for m in metrics):
                raise ValueError('Missing frozen-policy agreement diagnostics')
            if any(d['current_dtype'] != 'torch.float32' or d['old_dtype'] != 'torch.float32' for d in agreement):
                raise ValueError('Expected FP32 old/current log probabilities')
            if any(not math.isfinite(d['max_abs_sequence_ratio_minus_one']) for d in agreement):
                raise ValueError('Nonfinite frozen-policy agreement')
        if config.get('resolved', {}).get('microbatch_layout') == 'stable':
            validate_frozen_policy_agreement(agreement)
        summary[method]={'run':str(run), 'final_accuracy':final['accuracy'],
            'mean_sync_seconds':sum(m['sync_seconds'] for m in metrics)/len(metrics),
            'mean_rollout_seconds':sum(m['rollout_seconds'] for m in metrics)/len(metrics),
            'steps':metrics,'evaluations':evaluation,
            'max_frozen_sequence_ratio_deviation':max((d['max_abs_sequence_ratio_minus_one'] for d in agreement), default=None),
            'cliprange_low':config['args'].get('cliprange_low') if config['args'].get('cliprange_low') is not None else config['args'].get('cliprange'),
            'cliprange_high':config['args'].get('cliprange_high') if config['args'].get('cliprange_high') is not None else config['args'].get('cliprange')}
    a,b=configs['grpo'],configs['gspo']
    excluded={'run_name','importance_reweighting'}
    if allow_clipping_difference:
        excluded.update({'cliprange_low','cliprange_high'})
    if {k:v for k,v in a['args'].items() if k not in excluded}!={k:v for k,v in b['args'].items() if k not in excluded}:
        raise ValueError('Comparison settings differ beyond importance weighting/run name')
    for key in ('train_data','prompt','environment','resolved','sources','model','gpus','cuda_runtime'):
        if a[key]!=b[key]:raise ValueError(f'Unmatched {key}')
    if subsets['grpo']!=subsets['gspo']: raise ValueError('Evaluation subsets differ')
    (base/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (base/'comparison_summary.md').write_text(
        '# Stage 3 matched comparison\n\nSingle-seed smoke/pilot; no superiority claim. ' +
        ('Different clipping bounds explicitly allowed; exploratory comparison. ' if allow_clipping_difference else
         'Matched clipping bounds. ') +
        'Sequence normalization and eager inference; '
        'this is not a tuned GSPO replication.\n\n'
        '| Method | Clip low/high | Final accuracy | Mean sync (s) | Mean rollout (s) |\n'
        '|---|---|---:|---:|---:|\n'+''.join(
            f"| {k} | {v['cliprange_low']}/{v['cliprange_high']} | {v['final_accuracy']:.4f} | {v['mean_sync_seconds']:.3f} | {v['mean_rollout_seconds']:.3f} |\n"
            for k,v in summary.items()))
    print(f'PASS: matched GRPO/GSPO artifacts; summary saved to {base}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument("--allow-clipping-difference", action="store_true")
    args = parser.parse_args()
    summarize(args.results, args.allow_clipping_difference)

if __name__=='__main__':main()
