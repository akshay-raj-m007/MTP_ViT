"""Read saved epoch JSON files and generate a human-readable report; no training."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(f.read_text()) for f in sorted(a.run.glob('metrics_*.json'))]
    if not rows: raise SystemExit('No completed epoch metric files found.')
    best=max(rows,key=lambda r:r['map']);last=rows[-1]
    selection=json.loads((a.run/'best/selection.json').read_text())
    correction_file=a.run/'metric_correction.json'
    correction=json.loads(correction_file.read_text()) if correction_file.exists() else None
    consistent=selection['stage']==best['stage'] and selection['epoch']==best['epoch'] and abs(selection['map']-best['map'])<1e-10
    # Earliest max is the checkpoint retained by strict-improvement selection.
    def pct(v):return f'{100*v:.2f}%'
    lines=['# Training results', '', f'Run: `{a.run.name}`', '',
        f"Completed epochs: {len(rows)} ({sum(r['stage']=='A' for r in rows)} Stage A, {sum(r['stage']=='B' for r in rows)} Stage B).",
        f"Best checkpoint: Stage {best['stage']}, epoch {best['epoch']}; selection metadata matches epoch records: {consistent}.",
        '', '| Measure | Best checkpoint | Latest epoch |','|---|---:|---:|',
        f"| Validation mAP@0.50:0.95 | {pct(best['map'])} | {pct(last['map'])} |",
        f"| Validation AP@0.50 | {pct(best['map50'])} | {pct(last['map50'])} |",
        f"| Training loss | {best['train_loss']:.4f} | {last['train_loss']:.4f} |",'',
        'AP values are precision-recall summary metrics, not the percentage of images classified correctly.',
        '', '## Per-class results at the best checkpoint','',
        '| Class | AP@0.50:0.95 | AP@0.50 | Validation GT boxes |','|---|---:|---:|---:|']
    if correction:
        lines[4:4]=[
            '> **Historical AP warning:** Annotation ID 0 affected the epoch validation AP used for this run. These curves and epoch AP values have not been corrected.',
            '',
            f"The historically selected checkpoint's reevaluated validation mAP is {pct(correction['corrected_selected_best_validation_map'])}. Its historical selection may differ under corrected AP.",
            '',
            'See [the correction report](../../../../docs/COCO_ANNOTATION_ID_0_FIX.md) before comparing checkpoints.',
            '',
        ]
    for name,m in best['per_class_ap'].items():
        lines.append(f"| {name} | {pct(m['ap']) if m['ap'] is not None else 'N/A'} | {pct(m['ap50']) if m['ap50'] is not None else 'N/A'} | {m['support']} |")
    lines+=['','Support counts boxes, not images. The small validation set limits precision of these estimates.',
        '', '## Reading the curves','',
        '- Training loss: lower means a better fit to training targets; it does not guarantee better held-out detections.',
        '- Validation mAP: higher is better. This is the checkpoint-selection metric.',
        '- AP@0.50 uses a less strict box-overlap requirement than mAP averaged over 0.50 through 0.95.',
        '- Falling training loss alongside falling validation AP can indicate overfitting or unstable generalization; it does not by itself establish the cause.',
        '', '![Training curves](training_curves.png)', '', '## Every completed epoch','',
        '| Stage | Epoch | Training loss | Validation mAP | Validation AP50 |','|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['stage']} | {row['epoch']} | {row['train_loss']:.4f} | {pct(row['map'])} | {pct(row['map50'])} |")
    lines+=['','Threshold-dependent precision/recall/F1, image classification and clean-image false-alarm metrics require the separate validation evaluation command.',
            'No test data or inference was used to generate this report.']
    out=a.run/'readable_report';out.mkdir(exist_ok=True)
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=list(range(1,len(rows)+1));fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(x,[r['train_loss'] for r in rows],marker='.',color='steelblue');axes[0].set_ylabel('Training loss (lower is better)')
    axes[1].plot(x,[100*r['map'] for r in rows],label='mAP @ IoU 0.50:0.95',marker='.')
    axes[1].plot(x,[100*r['map50'] for r in rows],label='AP @ IoU 0.50',marker='.')
    axes[1].set_ylabel('Validation AP (%) ? higher is better');axes[1].legend()
    best_x=rows.index(best)+1;axes[1].axvline(best_x,color='green',linestyle=':',label='Best checkpoint')
    stage_a=sum(r['stage']=='A' for r in rows)
    for ax in axes:
        ax.set_xlabel('Overall epoch');ax.grid(alpha=.25)
        if stage_a and stage_a<len(rows):ax.axvline(stage_a+.5,color='gray',linestyle='--')
    fig.suptitle(f'{a.run.name}: dashed line = Stage A/B transition; green line = best mAP checkpoint')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    if correction:
        print(f"Historical selection: stage {best['stage']}, epoch {best['epoch']} | affected epoch mAP {pct(best['map'])}")
        print('Corrected selected-checkpoint validation mAP:',pct(correction['corrected_selected_best_validation_map']))
        print('Historical epoch AP curves and checkpoint selection may differ under corrected AP.')
    else:
        print(f"Best: stage {best['stage']}, epoch {best['epoch']} | mAP {pct(best['map'])} | AP50 {pct(best['map50'])}")
    print('Selection metadata matches:',consistent);print('Report:',out/'SUMMARY.md');print('Plot:',out/'training_curves.png')

if __name__=='__main__':main()
