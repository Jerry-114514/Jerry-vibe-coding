"""Protocol-v2 formulas and estimands, kept separate from historical v1."""
import numpy as np
from analyze_rulers import canonical, grid, winner_credit


def ruler_vector(scores, floors, setting, relative_base='large', u=None, eps=1e-12):
    scores, floors = np.asarray(scores, dtype=float), np.asarray(floors, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(floors) or len(scores) not in (2, 3):
        raise ValueError('The registered primary OWA supports exactly two or three nodes')
    if relative_base not in ('small', 'large'):
        raise ValueError('Relative denominator must be explicit')
    small, large = scores[:-1], scores[1:]
    corrected = setting['floor'] or setting['headroom'] == 'corrected_family'
    guards = [[] for _ in floors]
    for a in range(len(floors)):
        if not np.isfinite(scores[:, a]).all() or not np.isfinite(floors[a]):
            guards[a].append('no_valid_units')
        if corrected and (floors[a] >= 1-eps or np.any(scores[:, a] <= floors[a])):
            guards[a].append('sub_floor')
    if setting['headroom'] != 'none':
        denominator = np.asarray(scores[-1] if u is None else u) - small
        for a in range(len(floors)):
            if np.any(denominator[:, a] < -eps):
                guards[a].append('headroom_sign')
    elif setting['drop'] == 'relative':
        denominator = (large if relative_base == 'large' else small) - (floors if corrected else 0)
    else:
        denominator = np.broadcast_to(1-floors if corrected else np.ones_like(floors), small.shape)
    for a in range(len(floors)):
        if not np.isfinite(denominator[:, a]).all():
            guards[a].append('nonfinite_denominator')
        if np.any(np.abs(denominator[:, a]) <= eps):
            guards[a].append('zero_denominator')
    with np.errstate(divide='ignore', invalid='ignore'):
        gaps = (large-small)/denominator
    weights = np.array([1.0]) if len(small)==1 else np.array([setting['rho'], 1-setting['rho']])
    with np.errstate(invalid='ignore'):
        values = (np.sort(gaps, axis=0)[::-1]*weights[:, None]).sum(axis=0)
    for a, reason in enumerate(guards):
        if reason:
            values[a] = np.nan
    return values, guards, denominator


def metrics(classes, axis_indices, tie_epsilon=.01, require_full_axes=True):
    usable, full_ties, partial_ties, invalid, winners = [], 0, 0, 0, []
    for row in classes:
        vector = np.array(row['vector'], dtype=float)[axis_indices]
        if not np.isfinite(vector).any() or (require_full_axes and not np.isfinite(vector).all()):
            invalid += 1
            continue
        finite = np.isfinite(vector)
        win = np.flatnonzero(finite & (np.nanmax(vector)-vector <= tie_epsilon)).tolist()
        credit = np.zeros(len(axis_indices))
        credit[win] = 1/len(win)
        full_ties += int(len(win)==len(axis_indices))
        partial_ties += int(1<len(win)<len(axis_indices))
        usable.append(credit)
        winners.append(win)
    if not usable:
        return {'F': None, 'pi': [None]*len(axis_indices), 'strict_pair_flip_rate': None,
                'usable_classes': 0, 'invalid_classes': invalid, 'full_ties': 0, 'partial_ties': 0,
                'F_without_full_ties': None, 'stable_axes': []}
    credits = np.array(usable)
    pi = credits.mean(axis=0)
    unique = [w[0] for w in winners if len(w)==1]
    counts = np.bincount(unique, minlength=len(axis_indices))
    conflicting_pairs = int((len(unique)**2-int(np.square(counts).sum()))//2)
    total_pairs = len(usable)*(len(usable)-1)//2
    informative = [c for c,w in zip(usable, winners) if len(w)<len(axis_indices)]
    stable = set(range(len(axis_indices)))
    for win in winners:
        stable &= set(win)
    return {'F': float(1-pi.max()), 'pi': pi.tolist(), 'strict_pair_flip_rate': conflicting_pairs/total_pairs if total_pairs else 0.,
            'strict_conflicting_pairs': conflicting_pairs, 'all_usable_pairs': total_pairs,
            'usable_classes': len(usable), 'invalid_classes': invalid, 'full_ties': full_ties, 'partial_ties': partial_ties,
            'F_without_full_ties': float(1-np.mean(informative, axis=0).max()) if informative else None,
            'stable_axes': [axis_indices[i] for i in sorted(stable)]}


def scan(scores, floors, protocol, relative_base='large', quotient='structural', axis_indices=None, require_full_axes=True):
    if quotient not in ('structural', 'legacy_empirical'):
        raise ValueError('Unknown quotient')
    axis_indices = list(range(4)) if axis_indices is None else list(axis_indices)
    structural, observed, mapping = {}, {}, []
    for index, setting in enumerate(grid(protocol)):
        mode = setting['parse_fail']
        key = canonical(setting, len(scores[mode])-1)
        if key not in structural:
            values, guards, denominators = ruler_vector(scores[mode], floors[mode], setting, relative_base,
                                                        eps=protocol['denominator_epsilon'])
            structural[key] = {'id': len(structural), 'setting': setting, 'vector': values,
                               'guards': guards, 'denominators': denominators.tolist()}
        row = structural[key]
        observed_key = tuple(None if not np.isfinite(x) else round(float(x), 12) for x in row['vector'])
        if observed_key not in observed:
            observed[observed_key] = {**row, 'id': len(observed)}
        mapping.append({'nominal_id': index, **setting, 'structural_id': row['id'],
                        'empirical_id': observed[observed_key]['id']})
    selected = list(structural.values()) if quotient=='structural' else list(observed.values())
    return {'classes': selected, 'mapping': mapping, 'structural_classes': len(structural),
            'empirical_classes': len(observed),
            **metrics(selected, axis_indices, protocol['tie_epsilon'], require_full_axes)}


def signed_gain(pre, post, floor, eps=1e-12):
    pre, post, floor = [np.asarray(x, dtype=float) for x in (pre, post, floor)]
    delta = post-pre
    valid_raw = np.isfinite(delta).all()
    valid_floor = valid_raw and np.isfinite(floor).all() and np.all(1-floor > eps)
    components = delta/(1-floor) if valid_floor else np.full_like(delta, np.nan)
    return {'G_raw': float(delta.sum()) if valid_raw else None,
            'G_floor': float(components.sum()) if valid_floor else None,
            'signed_raw': delta.tolist(), 'signed_floor': components.tolist(),
            'worst_axis_raw_change': float(delta.min()) if valid_raw else None,
            'defined': bool(valid_floor)}


def joint_cluster_features(records):
    """Sufficient statistics preserve the same image weight across every axis."""
    identities = [(r['axis'], r['stratum'], r['unit']) for r in records[0]]
    if any([(r['axis'], r['stratum'], r['unit']) for r in rows] != identities for rows in records):
        raise ValueError('Unpaired records')
    strata = sorted({(r['axis'], r['stratum']) for r in records[0]})
    groups = sorted({r['image_group'] for r in records[0]})
    group_index = {g: i for i,g in enumerate(groups)}
    stratum_index = {s: i for i,s in enumerate(strata)}
    width = len(records)+1
    features = np.zeros((len(groups), len(strata)*width))
    for i, row in enumerate(records[0]):
        group = group_index[row['image_group']]
        start = stratum_index[(row['axis'], row['stratum'])]*width
        features[group, start] += 1
        features[group, start+1:start+width] += [rows[i]['correct_bit'] for rows in records]
    return features, strata


def scores_from_cluster_totals(totals, strata, record_count, axes):
    totals = np.atleast_2d(totals).reshape(-1, len(strata), record_count+1)
    with np.errstate(divide='ignore', invalid='ignore'):
        stratum_means = totals[:, :, 1:]/totals[:, :, :1]
    scores = np.empty((len(totals), record_count, len(axes)))
    for a, axis in enumerate(axes):
        indices = [i for i,s in enumerate(strata) if s[0]==axis]
        scores[:, :, a] = stratum_means[:, indices].mean(axis=1)
    return scores


def paired_cluster_scores(records, axes, n_boot, seed):
    features, strata = joint_cluster_features(records)
    features, cluster_counts = np.unique(features, axis=0, return_counts=True)
    n_clusters = int(cluster_counts.sum())
    rng = np.random.default_rng(seed)
    totals = [cluster_counts[None, :]@features]
    for offset in range(0, n_boot, 128):
        weights = rng.multinomial(n_clusters, cluster_counts/n_clusters, size=min(128, n_boot-offset))
        totals.append(weights@features)
    return scores_from_cluster_totals(np.concatenate(totals), strata, len(records), axes)
