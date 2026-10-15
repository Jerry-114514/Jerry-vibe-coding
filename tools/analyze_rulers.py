"""C1--C5/D5: frozen-ledger multiverse, paired stratified bootstrap and gate."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from experiment_common import AXES, LADDERS, ROOT, read_json, read_rows, sha, units, write_json, write_rows


def grid(protocol, frozen=None):
    names = ['drop', 'floor', 'headroom', 'rho', 'parse_fail']
    choices = [[frozen[n]] if frozen and n in frozen else protocol[n] for n in names]
    return [dict(zip(names, values)) for values in itertools.product(*choices)]


def canonical(setting, edge_count):
    # Preserve different admissible domains even when numerical formulas cancel.
    corrected = setting['floor'] or setting['headroom'] == 'corrected_family'
    if setting['headroom'] != 'none':
        formula = 'headroom'
    elif setting['drop'] == 'relative':
        formula = 'relative_corrected' if setting['floor'] else 'relative'
    else:
        formula = 'absolute_corrected' if setting['floor'] else 'absolute'
    return (formula, bool(corrected), setting['rho'] if edge_count > 1 else .5, setting['parse_fail'])


def ruler_vector(scores, floors, setting, eps=1e-12, u=None):
    small, large = scores[:-1], scores[1:]
    delta = large - small
    reasons = [[] for _ in AXES]
    corrected = setting['floor'] or setting['headroom'] == 'corrected_family'
    if corrected:
        for a in range(len(AXES)):
            if np.any(scores[:, a] <= floors[a]) or floors[a] >= 1 - eps:
                reasons[a].append('sub_floor')
    if setting['headroom'] != 'none':
        denominator = (scores[-1] if u is None else u) - small
        for a in range(len(AXES)):
            if np.any(denominator[:, a] < -eps):
                reasons[a].append('headroom_sign')
    elif setting['drop'] == 'relative':
        denominator = small - floors if corrected else small
    else:
        denominator = np.broadcast_to(1 - floors if corrected else np.ones(len(AXES)), delta.shape)
    for a in range(len(AXES)):
        if not np.isfinite(scores[:, a]).all() or not np.isfinite(floors[a]):
            reasons[a].append('no_valid_units')
        if np.any(np.abs(denominator[:, a]) <= eps):
            reasons[a].append('zero_denominator')
    with np.errstate(divide='ignore', invalid='ignore'):
        gaps = delta / denominator
    sorted_gaps = np.sort(gaps, axis=0)[::-1]
    weights = np.array([1.0]) if len(small) == 1 else np.array([setting['rho'], 1 - setting['rho']])
    result = np.sum(sorted_gaps * weights[:, None], axis=0)
    for a, reason in enumerate(reasons):
        if reason:
            result[a] = np.nan
    return result, reasons


def winner_credit(vector, tie_epsilon):
    valid = np.isfinite(vector)
    if not valid.any():
        return np.zeros(len(AXES)), []
    winners = np.flatnonzero(valid & (np.nanmax(vector) - vector <= tie_epsilon)).tolist()
    credit = np.zeros(len(AXES))
    credit[winners] = 1 / len(winners)
    return credit, winners


def scan(scores_by_mode, floors_by_mode, protocol, frozen=None):
    settings = grid(protocol, frozen)
    classes, mapping, structural = {}, [], {}
    for index, setting in enumerate(settings):
        key = canonical(setting, scores_by_mode[setting['parse_fail']].shape[0] - 1)
        if key not in structural:
            vector, reasons = ruler_vector(scores_by_mode[setting['parse_fail']], floors_by_mode[setting['parse_fail']], setting,
                                          protocol['denominator_epsilon'])
            credit, winners = winner_credit(vector, protocol['tie_epsilon'])
            structural[key] = {'id': len(structural), 'setting': setting, 'vector': vector, 'guards': reasons,
                            'credit': credit, 'winners': winners}
        value=structural[key]
        observed=tuple(None if not np.isfinite(v) else round(float(v),12) for v in value['vector'])
        if observed not in classes:
            classes[observed]={**value,'id':len(classes),'structural_members':[value['id']]}
        elif value['id'] not in classes[observed]['structural_members']:
            classes[observed]['structural_members'].append(value['id'])
        mapping.append({'nominal_id': index, **setting, 'structural_class_id':value['id'], 'class_id': classes[observed]['id']})
    usable = [value for value in classes.values() if value['winners']]
    pi = np.mean([value['credit'] for value in usable], axis=0) if usable else np.full(len(AXES), np.nan)
    stable = set(range(len(AXES)))
    for value in usable:
        stable &= set(value['winners'])
    all_tied = bool(usable) and all(len(value['winners']) == len(AXES) for value in usable)
    return {'classes': list(classes.values()), 'structural_class_count':len(structural), 'mapping': mapping, 'pi': pi,
            'F': float(1 - np.max(pi)) if usable and not all_tied else None,
            'stable': sorted(stable) if usable and not all_tied else [], 'all_tied': all_tied}


def bootstrap_scores(manifest, ledgers, family, n_boot, seed, homogeneous=False, cluster=False):
    keys = LADDERS[family]
    baseline = units(manifest, read_rows(ROOT / 'data/baselines/per_item.jsonl'))
    records = [baseline] + [units(manifest, ledgers[key]) for key in keys]
    identities = [(x['axis'], x['stratum'], x['unit']) for x in baseline]
    if any([(x['axis'], x['stratum'], x['unit']) for x in rows] != identities for rows in records):
        raise ValueError('Unpaired scoring units')
    bits = np.array([[x['correct_bit'] for x in rows] for rows in records], dtype=float)
    failed = np.array([[x['parse_fail_flag'] for x in rows] for rows in records], dtype=bool)
    scores = {mode: np.full((n_boot+1, len(keys), len(AXES)), np.nan) for mode in ['wrong', 'exclude']}
    floors = {mode: np.full((n_boot+1, len(AXES)), np.nan) for mode in ['wrong', 'exclude']}
    rng = np.random.default_rng(seed)
    for a, axis in enumerate(AXES):
        strata = sorted({x['stratum'] for x in baseline if x['axis'] == axis})
        contributions = {mode: [] for mode in scores}
        for stratum in strata:
            positions = np.array([i for i, x in enumerate(baseline) if x['axis'] == axis and x['stratum'] == stratum])
            if cluster:
                groups = {}
                for i in positions:
                    groups.setdefault(baseline[i]['image_group'], []).append(i)
                clusters = list(groups.values())
                draws = [positions] + [np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]) for _ in range(n_boot)]
            else:
                draws = [positions] + [positions[rng.integers(len(positions), size=len(positions))] for _ in range(n_boot)]
            for mode in scores:
                values = []
                for draw in draws:
                    if mode == 'exclude':
                        draw = draw[~failed[1:, draw].any(axis=0)]
                    values.append(bits[:, draw].mean(axis=1) if len(draw) else np.full(len(records), np.nan))
                contributions[mode].append(np.array(values))
        for mode in scores:
            means = np.mean(contributions[mode], axis=0)
            floors[mode][:, a] = means[:, 0]
            scores[mode][:, :, a] = means[:, 1:]
    if homogeneous:
        for mode in floors:
            floors[mode][:] = floors[mode].mean(axis=1)[:, None]
    return scores, floors


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def ci(values):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    return np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]


def nested_noise_ci(manifest,ledgers,family,n_outer,n_inner,seed,tie_epsilon,cluster=False):
    """Nested paired item/group bootstrap; endpoint gaps are sufficient for mean OWA.

    Compress groups by (endpoint correctness difference, group size). Multinomial
    group draws are exactly equivalent to resampling the original units, including
    the paired endpoint covariance. No Gaussian/Binomial independence approximation.
    """
    keys=LADDERS[family]
    small=units(manifest,ledgers[keys[0]])
    large=units(manifest,ledgers[keys[-1]])
    edge_count=len(keys)-1
    rng=np.random.default_rng(seed)
    vectors=np.zeros((n_outer,n_inner,len(AXES)))
    for a,axis in enumerate(AXES):
        strata=sorted({x['stratum'] for x in small if x['axis']==axis})
        for stratum in strata:
            groups={}
            for i,row in enumerate(small):
                if row['axis']==axis and row['stratum']==stratum:
                    identity=row['image_group'] if cluster else row['unit']
                    groups.setdefault(identity,[]).append(i)
            features=[(sum(large[i]['correct_bit']-small[i]['correct_bit'] for i in group),len(group)) for group in groups.values()]
            values,counts=np.unique(np.array(features),axis=0,return_counts=True)
            n_groups=len(features)
            outer=rng.multinomial(n_groups,counts/n_groups,size=n_outer)
            for b in range(n_outer):
                inner=rng.multinomial(n_groups,outer[b]/n_groups,size=n_inner)
                numerators=inner@values[:,0]
                denominators=inner@values[:,1]
                vectors[b,:,a]+=numerators/denominators/(len(strata)*edge_count)
    wins=(vectors.max(axis=2,keepdims=True)-vectors)<=tie_epsilon
    credits=wins/wins.sum(axis=2,keepdims=True)
    rates=1-credits.mean(axis=1).max(axis=1)
    return ci(rates),rates


def planted():
    scores = np.array([[.60, .60, .60, .60], [.90, .75, .70, .68]])
    floors = np.full(4, .20)
    protocol = read_json(ROOT / 'configs/experiment.json')
    scanned = scan({'wrong': scores, 'exclude': scores}, {'wrong': floors, 'exclude': floors}, protocol)
    if np.argmax(scanned['pi']) != 0:
        raise AssertionError('Homogeneous planted axis recovery failed')
    cf = np.array([.50, .02, .02, .02])
    heter = np.array([[.60, .60, .60, .60], [.74, .80, .60, .60]])
    setting = {'drop': 'absolute', 'floor': False, 'headroom': 'none', 'rho': .5, 'parse_fail': 'wrong'}
    raw, _ = ruler_vector(heter, cf, setting)
    corrected, _ = ruler_vector(heter, cf, {**setting, 'floor': True})
    if np.argmax(raw) != 1 or np.argmax(corrected) != 0:
        raise AssertionError('Heterogeneous planted flip failed')
    return {'homogeneous_recovered': True, 'planted_axis': AXES[0], 'pi': scanned['pi'].tolist(),
            'heterogeneous_flip_recovered': True, 'raw': raw.tolist(), 'corrected': corrected.tolist(),
            'flip_ratio_boundary': [(1-.5)/(1-.02), 1.0]}


def analyze(run_path, n_boot=None, cluster=False, synthetic_test=False):
    run_path = Path(run_path)
    manifest = read_rows(ROOT / 'data/manifest.jsonl')
    protocol = read_json(ROOT / 'configs/experiment.json')
    n_boot = protocol['n_boot'] if n_boot is None else n_boot
    ledgers = {}
    for key in {key for ladder in LADDERS.values() for key in ladder}:
        paths = sorted((run_path / 'ledger').glob(f'{key}-[0-9][0-9][0-9].jsonl'))
        if not paths:
            raise ValueError('Missing complete model ledger: ' + key)
        for path in paths:
            meta = read_json(path.with_suffix('.meta.json'))
            if meta['smoke'] or (meta.get('synthetic',False) and not synthetic_test) or meta['manifest_sha256'] != sha(ROOT / 'data/manifest.jsonl'):
                raise ValueError('Smoke/foreign ledger cannot enter formal analysis')
        ledgers[key] = [row for path in paths for row in read_rows(path)]
        units(manifest, ledgers[key])
    output = run_path / ('analysis-cluster' if cluster else 'analysis')
    output.mkdir(parents=True, exist_ok=True)
    family_results = {}
    summaries = []
    for family in LADDERS:
        samples, floors = bootstrap_scores(manifest, ledgers, family, n_boot, protocol['seed'], cluster=cluster)
        point = scan({mode: samples[mode][0] for mode in samples}, {mode: floors[mode][0] for mode in floors}, protocol)
        boot_pi, boot_F, noise_credits = [], [], []
        fixed = {'drop': 'absolute', 'floor': False, 'headroom': 'none', 'rho': .5, 'parse_fail': 'wrong'}
        for b in range(1, n_boot+1):
            scanned = scan({mode: samples[mode][b] for mode in samples}, {mode: floors[mode][b] for mode in floors}, protocol)
            boot_pi.append(scanned['pi'])
            boot_F.append(scanned['F'])
            vector, _ = ruler_vector(samples['wrong'][b], floors['wrong'][b], fixed)
            noise_credits.append(winner_credit(vector, protocol['tie_epsilon'])[0])
        noise_credits = np.array(noise_credits)
        noise = float(1-np.max(noise_credits.mean(axis=0)))
        rng = np.random.default_rng(protocol['seed']+1)
        # Monte-Carlo CI for the bootstrap noise rate, explicitly distinct from F_scale's sampling CI.
        noise_replicates = [1-np.max(noise_credits[rng.integers(n_boot, size=n_boot)].mean(axis=0)) for _ in range(n_boot)]
        noise_sampling_ci,noise_outer=nested_noise_ci(manifest,ledgers,family,n_boot,n_boot,protocol['seed']+2,
                                                      protocol['tie_epsilon'],cluster=cluster)
        mapping = point['mapping']
        write_rows(output / f'{family}-folding.jsonl', mapping)
        write_json(output / f'{family}-classes.json', finite_json(point['classes']))
        attribution = []
        for coordinate, value in fixed.items():
            restricted = scan({mode: samples[mode][0] for mode in samples}, {mode: floors[mode][0] for mode in floors}, protocol,
                              frozen={coordinate: value})
            attribution.append({'coordinate': coordinate, 'fixed_value': value, 'F_frozen': restricted['F'],
                                'F_reduction': point['F']-restricted['F'] if point['F'] is not None and restricted['F'] is not None else None,
                                'explained_flip_fraction':(point['F']-restricted['F'])/point['F'] if point['F'] not in (None,0) and restricted['F'] is not None else None})
        # D5 changes only c_a -> c_bar; reuse exactly the same paired draws.
        hom_floors={mode:np.repeat(floors[mode].mean(axis=1)[:,None],len(AXES),axis=1) for mode in floors}
        homogeneous = scan({m: samples[m][0] for m in samples}, {m: hom_floors[m][0] for m in hom_floors}, protocol)
        hom_pi,hom_F=[],[]
        for b in range(1,n_boot+1):
            value=scan({m:samples[m][b] for m in samples},{m:hom_floors[m][b] for m in hom_floors},protocol)
            hom_pi.append(value['pi'])
            hom_F.append(value['F'])
        hom_attribution=[]
        for coordinate,value in fixed.items():
            restricted=scan({m:samples[m][0] for m in samples},{m:hom_floors[m][0] for m in hom_floors},protocol,frozen={coordinate:value})
            hom_attribution.append({'coordinate':coordinate,'F_frozen':restricted['F'],
                                    'F_reduction':homogeneous['F']-restricted['F'] if homogeneous['F'] is not None and restricted['F'] is not None else None})
        write_rows(output/f'{family}-homogeneous-folding.jsonl',homogeneous['mapping'])
        write_json(output/f'{family}-homogeneous-classes.json',finite_json(homogeneous['classes']))
        result = {'family': family, 'models': LADDERS[family], 'nominal_settings': len(mapping),
                  'quotient_classes': len(point['classes']), 'pi_point': dict(zip(AXES, point['pi'].tolist())),
                  'structural_class_count':point['structural_class_count'],
                  'pi_boot_mean': dict(zip(AXES, np.nanmean(boot_pi, axis=0).tolist())),
                  'pi_CI': {a: ci(np.array(boot_pi)[:, i]) for i, a in enumerate(AXES)},
                  'F_scale': point['F'], 'F_scale_CI': ci(boot_F), 'F_noise': noise,
                  'F_noise_CI':noise_sampling_ci,'F_noise_CI_method':'nested paired bootstrap',
                  'F_noise_MC_CI': ci(noise_replicates),
                  'stable_axis_set': [AXES[i] for i in point['stable']], 'all_axes_tied': point['all_tied'],
                  'undefined_bootstraps': sum(x is None for x in boot_F), 'attribution': attribution,
                  'scores': {k: dict(zip(AXES, samples['wrong'][0, i].tolist())) for i, k in enumerate(LADDERS[family])},
                  'u_a': dict(zip(AXES, samples['wrong'][0, -1].tolist())),
                  'c_a': dict(zip(AXES, floors['wrong'][0].tolist())),
                  'homogeneous_floor_control': {'pi': homogeneous['pi'].tolist(),'pi_boot_mean':np.nanmean(hom_pi,axis=0).tolist(),
                                                'F_scale': homogeneous['F'],'F_scale_CI':ci(hom_F),
                                                'stable_axis_set':[AXES[i] for i in homogeneous['stable']],
                                                'attribution':hom_attribution,'n_boot':n_boot}}
        family_results[family] = finite_json(result)
        summaries.append(finite_json(result))
        np.savez_compressed(output / f'{family}-bootstrap.npz', scores_wrong=samples['wrong'], scores_exclude=samples['exclude'],
                            floors_wrong=floors['wrong'], floors_exclude=floors['exclude'], pi=np.array(boot_pi),
                            F=np.array([np.nan if x is None else x for x in boot_F]), noise_credits=noise_credits,
                            noise_outer_rates=noise_outer,homogeneous_pi=np.array(hom_pi),
                            homogeneous_F=np.array([np.nan if x is None else x for x in hom_F]))
    defined = all(x['F_scale'] is not None for x in summaries)
    falsified = defined and all(x['stable_axis_set'] and x['F_scale'] <= x['F_noise_CI'][1] for x in summaries)
    separated = defined and any(x['F_scale_CI'][0] is not None and x['F_scale_CI'][0] > x['F_noise_CI'][1] for x in summaries)
    decision = 'no-go' if falsified else ('go' if separated else 'indeterminate')
    gate = {'decision': decision, 'n_boot': n_boot, 'formal': n_boot == protocol['n_boot'] and not cluster and not synthetic_test,
            'synthetic_test': synthetic_test,
            'reason': 'All ladders stable within noise' if falsified else ('Scale exceeds noise CI' if separated else 'Insufficient separation or undefined argmax'),
            'manifest_sha256': sha(ROOT / 'data/manifest.jsonl'), 'protocol_sha256': sha(ROOT / 'configs/experiment.json'),
            'ledger_sha256': {str(p.relative_to(run_path)): sha(p) for p in sorted((run_path / 'ledger').glob('*.jsonl'))}}
    write_json(output / 'summary.json', {'protocol': protocol, 'families': family_results, 'gate': gate})
    write_json(output / 'gate.json', gate)
    write_json(output / 'planted.json', planted())
    print(gate, flush=True)
    return gate


def verify_go(run_path):
    run_path=Path(run_path)
    gate=read_json(run_path/'analysis/gate.json')
    if gate['decision']!='go' or not gate['formal'] or gate.get('synthetic_test'):
        raise ValueError('A formal C3 go decision is required')
    if gate['protocol_sha256']!=sha(ROOT/'configs/experiment.json') or gate['manifest_sha256']!=sha(ROOT/'data/manifest.jsonl'):
        raise ValueError('Protocol or manifest changed after C3')
    for path,digest in gate['ledger_sha256'].items():
        if sha(run_path/path)!=digest:
            raise ValueError('Inference ledger changed after C3')
    return gate


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', default=str(ROOT / 'runs/formal'))
    p.add_argument('--n-boot', type=int)
    p.add_argument('--cluster', action='store_true')
    p.add_argument('--planted-only', action='store_true')
    a = p.parse_args()
    if a.planted_only:
        write_json(ROOT / 'runs/preflight/planted.json', planted())
    else:
        analyze(a.run, a.n_boot, a.cluster)
