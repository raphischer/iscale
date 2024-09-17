import pandas as pd
import numpy as np

from util import load_meta, load_database, identify_property_meta


def _value_to_index(value, ref, higher_better=True):
    res = value / ref if higher_better else ref / value
    res[np.isinf(value)] = 1 if higher_better else 0
    res[np.isnan(value)] = 0
    return res

def _index_to_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i
    return 4 # worst rating if index does not fall in boundaries

def _index_to_value(index, ref, higher_better=True):
    raise NotImplementedError
    assert ref > 0, f'Invalid reference value {ref} (must be larger than zero)'
    if isinstance(index, float) and index == 0:
        index = 10e-4
    if isinstance(index, pd.Series) and index[index==0].size > 0:
        index = index.copy()
        index[index == 0] = 10e-4
    #      v = i * r                            OR         v = r / i
    return index * ref  if higher_better else ref / index

def index_scale_reference(input, properties_meta, reference):
    raise NotImplementedError

def index_scale_best(input, properties_meta, ___unused=None):
    results = {}
    for prop, meta in properties_meta.items():
        assert not np.any(input[prop] < 0), f"Found negative values in {prop}, please scale to only positive values!"
        if 'maximize' in meta and meta['maximize']:
            results[prop] = _value_to_index(input[prop], input[prop].max())
        else:
            results[prop] = _value_to_index(input[prop], input[prop].min(), higher_better=False)
    return pd.DataFrame(results)

def rate(input, ___unused=None, boundaries=None):
    assert(isinstance(boundaries, dict))
    results = {prop: np.digitize(input[prop], bins) for prop, bins in boundaries.items()}
    return pd.DataFrame(results, index=input.index)

def reverse_index_scale(input):
    raise NotImplementedError

def check_for_splitting(input):
    return [field for field in ['environment', 'task', 'dataset'] if field in input.columns]

def prepare_transformation(input, meta=None):
    if isinstance(input, pd.DataFrame):
        pass
    else:
        raise NotImplementedError('Please pass a pandas dataframe as input!')
    properties_meta = identify_property_meta(input, meta)
    split_by = check_for_splitting(input)
    return input, properties_meta, split_by

def prepare_boundaries(input, boundaries=None):
    assert (input.shape[0] > 0) and (input.shape[1] > 0)
    assert not np.any(np.logical_or(input>1, input<=0)), 'Found values outside of the interval (0, 1] - please index-scale your results first and remove all unimportant columns!'
    if isinstance(boundaries, dict):
        assert sorted(list(boundaries.keys())) == sorted(input.columns)
        return boundaries
    elif isinstance(boundaries, np.ndarray):
        quantiles = boundaries
    elif boundaries is None:
        quantiles = [0.8, 0.6, 0.4, 0.2]
    else:
        raise NotImplementedError('Please pass boundaries / reference as a dict (list of boundaries for each property), np.ndarray (quantiles) or None (for standard [0.8, 0.6, 0.4, 0.2] quantiles).')
    boundaries = {prop: np.quantile(input[prop], quantiles) for prop in input.columns}
    return boundaries

def scale(input, meta=None, reference=None, verbose=True, mode='index'):
    input, properties_meta, split_by = prepare_transformation(input, meta)
    assert mode in ['rating', 'index'], 'Only "index" and "rating" modes are supported!'
    mode_str = 'relative index scaling'
    scale_m = index_scale_best if reference is None else index_scale_reference
    if mode == 'rating':
        assert not np.any(np.logical_or(input[properties_meta.keys()]>1, input[properties_meta.keys()]<=00)), 'Found values outside of the interval (0, 1] - please index-scale your results first!'
        reference = prepare_boundaries(input[properties_meta.keys()], reference) # reference now controls the rating boundaries per property
        scale_m = rate
        mode_str = 'discrete rating'

    if len(split_by) > 0:
        if verbose:
            print(f'Performing {mode_str} for every separate combination of {str(split_by)}, with the following properties:\n    {", ".join(properties_meta.keys())}')
        sub_results = []
        for sub_config, sub_input in input.groupby(split_by):
            if 'task' in split_by:
                sub_meta = {prop: meta for prop, meta in properties_meta.items() if 'task' not in meta or meta['task'] == sub_input['task'].iloc[0]}
            else:
                sub_meta = properties_meta
            sub_results.append(scale_m(sub_input, sub_meta, reference))
            if verbose:
                print('   - scaled results for', sub_config)
        results = pd.concat(sub_results).sort_index()
    else:
        if verbose:
            print(f'Performing {mode_str} for the complete data frame, without any splitting. If you want internal splitting, please provide information on respective "environment", "task" or "dataset". Scaling the following properties:\n    {properties_str}')
        results = scale_m(input, properties_meta, reference)
    final = pd.concat([input[[col for col in input.columns if col not in properties_meta]], results], axis=1)
    return final

if __name__ == '__main__':
    
    # load database
    database = load_database('metaqure/database.pkl')
    # database[database['dataset'].isin(pd.unique(database['dataset'])[:5])].reset_index(drop=True).to_pickle('test_db/database.pkl')
    # load meta infotmation
    meta = load_meta('metaqure')

    scaled = scale(database, meta)
    rated = scale(scaled, meta, mode='rating')

    print(1)