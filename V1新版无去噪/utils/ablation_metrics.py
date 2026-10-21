"""Unambiguous paired deltas and scene-cluster bootstrap; no statistics dependency."""
from collections import defaultdict
import numpy as np
import math


def mining_summary(rows):
    """Exact epoch/pilot counts and pair-distance quantiles from detached rows."""
    if not rows:return {}
    valid=[r for r in rows if r['valid']]
    result=dict(triplet_anchor_count=len(rows),triplet_valid_count=len(valid),triplet_skipped=len(rows)-len(valid),
        semihard_selected=sum(r['selection_category']=='semihard' for r in valid),
        hard_fallback=sum(r['selection_category']=='hard_fallback' for r in valid),
        active_triplet_ratio=sum(r['active'] for r in valid)/len(valid) if valid else 'NA',
        valid_negative_count=float(np.mean([r['valid_negative_count'] for r in rows])))
    for name in ('d_ap','d_an'):
        if valid:
            values=[r[name] for r in valid]
            result[name]=float(np.mean(values))
            for suffix,q in (('p10',.1),('median',.5),('p90',.9)):result[name+'_'+suffix]=float(np.quantile(values,q))
    return result


def paired_summary(rows,seed=42,repeats=1000):
    """CI resamples complete content clusters, not correlated illumination images."""
    if not rows: return dict(delta_psnr_vs_input='NA',paired_reason='no images')
    values=np.asarray([float(r['delta_psnr_vs_input'] if 'delta_psnr_vs_input' in r else r['delta_PSNR_RGB']) for r in rows])
    if not np.isfinite(values).all():
        return dict(delta_psnr_vs_input='NA',delta_psnr_vs_input_median='NA',improvement_image_fraction='NA',
            cluster_bootstrap_ci95_low='NA',cluster_bootstrap_ci95_high='NA',
            paired_reason='Nonfinite PSNR differences (including perfect-match inf); no subset mean or fabricated CI.')
    groups=defaultdict(list)
    for r,v in zip(rows,values):
        key=(r.get('dataset') or 'mill',r.get('split') or 'validation',
             r.get('group_id') or r.get('content_id') or r.get('scene_id') or r['filename'])
        groups[key].append(v)
    clusters=list(groups.values());rng=np.random.default_rng(seed)
    samples=[]
    if len(clusters)>1:
        for _ in range(repeats):
            sampled=[clusters[i] for i in rng.integers(0,len(clusters),size=len(clusters))]
            samples.append(float(np.mean([v for group in sampled for v in group])))
    return dict(delta_psnr_vs_input=float(values.mean()),delta_psnr_vs_input_median=float(np.median(values)),
        improvement_image_fraction=float(np.mean(values>0)),decline_image_fraction=float(np.mean(values<0)),
        cluster_bootstrap_ci95_low=float(np.quantile(samples,.025)) if samples else 'NA',
        cluster_bootstrap_ci95_high=float(np.quantile(samples,.975)) if samples else 'NA',
        paired_reason='' if samples else 'Fewer than two independent content/group clusters; CI is NA.',
        bootstrap_unit='(dataset, split, verified group_id or full content_id)',bootstrap_clusters=len(groups),bootstrap_seed=seed)


def compare_control(candidate,control):
    """Matched full-image CSV comparison, never mix interventions or data splits.

    Matched input metrics are an extra check, not a substitute for confirming the
    same documented precision/metric/data protocol before scientific comparison.
    """
    def number(value):
        try:return float(value)
        except (ValueError,TypeError):return math.nan
    def metric_value(row,metric):
        if metric=='illumination_MAE':return number(row.get(metric,row.get('illumination_abs_error','NA')))
        return number(row.get('output_'+metric,row.get(metric,'NA')))
    def index(rows):
        mapping={}
        for row in rows:
            if 'mode' in row:raise ValueError('Condition interventions are not full-image control evaluation CSVs.')
            key=(row.get('dataset') or 'mill',row.get('split') or 'validation',row['filename'])
            if key in mapping:raise ValueError('Duplicate evaluation image key.')
            mapping[key]=row
        return mapping
    a,b=index(candidate),index(control)
    if not a or a.keys()!=b.keys():raise ValueError('Candidate/control image sets or splits differ; no subset averaging.')
    output=[]
    for key,row in a.items():
        reference=b[key]
        for metric in ('input_PSNR_RGB','input_SSIM_RGB'):
            if metric not in row or metric not in reference or not math.isclose(float(row[metric]),float(reference[metric]),rel_tol=1e-7,abs_tol=1e-5):
                raise ValueError('Input baseline metrics differ; check GT/data/metric protocol.')
        source='output_PSNR_RGB' if 'output_PSNR_RGB' in row else 'PSNR_RGB'
        ref_source='output_PSNR_RGB' if 'output_PSNR_RGB' in reference else 'PSNR_RGB'
        result=dict(row,dataset=key[0],split=key[1])
        result.update(control_PSNR_RGB=float(reference[ref_source]),
            delta_psnr_vs_control=float(row[source])-float(reference[ref_source]))
        if not math.isfinite(result['delta_psnr_vs_control']):
            result.update(delta_psnr_vs_control='NA',delta_psnr_vs_control_reason='Nonfinite/perfect-match PSNR comparison; see raw PSNR fields.')
        for metric in ('SSIM_RGB','illumination_MAE'):
            source=('output_'+metric) if 'output_'+metric in row else metric
            ref_source=('output_'+metric) if 'output_'+metric in reference else metric
            value,reference_value=metric_value(row,metric),metric_value(reference,metric)
            result[metric]=value if math.isfinite(value) else 'NA'
            result['control_'+metric]=reference_value if math.isfinite(reference_value) else 'NA'
            result['delta_'+metric+'_vs_control']=(value-reference_value if math.isfinite(value) and math.isfinite(reference_value) else 'NA')
        output.append(result)
    def summarize(rows,**identity):
        result=(paired_summary([dict(r,delta_psnr_vs_input=r['delta_psnr_vs_control']) for r in rows])
                if rows and all(isinstance(r['delta_psnr_vs_control'],(int,float)) for r in rows) else
                dict(delta_psnr_vs_input='NA',paired_reason='No images or nonfinite paired differences; no subset summary.',
                     cluster_bootstrap_ci95_low='NA',cluster_bootstrap_ci95_high='NA'))
        result={key.replace('delta_psnr_vs_input','delta_psnr_vs_control'):value for key,value in result.items()}
        contents={r.get('content_id') or r.get('scene_id') or r['filename'] for r in rows}
        result.update(identity,count=len(rows),scene_count=len(contents))
        for field in ('PSNR_RGB','SSIM_RGB','input_PSNR_RGB','input_SSIM_RGB','illumination_MAE',
                      'control_PSNR_RGB','control_SSIM_RGB','control_illumination_MAE',
                      'delta_SSIM_RGB_vs_control','delta_illumination_MAE_vs_control'):
            values=[number(r.get('output_'+field,r.get(field,'nan'))) for r in rows]
            result[field]=float(np.mean(values)) if values and np.isfinite(values).all() else 'NA'
        return result
    grouped=defaultdict(list)
    for row in output:grouped[(row['dataset'],row['split'])].append(row)
    by_dataset_split=[];by_level=[];by_content=[]
    for (dataset,split),rows in sorted(grouped.items()):
        identity=dict(dataset=dataset,split=split)
        by_dataset_split.append(summarize(rows,**identity))
        if dataset.casefold()=='mill':
            for level in range(1,11):
                subset=[r for r in rows if float(r.get('level') or r.get('illum_level') or -1)==level]
                by_level.append(summarize(subset,**identity,level=level))
            for content in sorted({r.get('content_id') or r.get('scene_id') or r['filename'] for r in rows}):
                subset=[r for r in rows if (r.get('content_id') or r.get('scene_id') or r['filename'])==content]
                by_content.append(summarize(subset,**identity,content_id=content))
    summary=dict(by_dataset_split=by_dataset_split,mill_per_level=by_level,mill_per_content=by_content,
                 comparison='candidate minus control; separate dataset/split summaries; no cross-dataset aggregate',
                 missing_dataset_policy='dataset=mill; missing split=validation (never inferred as test)')
    # Compatible scalar fields only when they unambiguously represent ONE split.
    if len(by_dataset_split)==1:summary.update(by_dataset_split[0])
    return output,summary
