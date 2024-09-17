import os
import json
import re
import pandas as pd

def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)
    
def load_meta(directory=None):
    if directory is None:
        directory = os.getcwd()
    meta = {}
    for fname in os.listdir(directory):
        re_match = re.match('meta_(.*).json', fname)
        if re_match:
            meta[re_match.group(1)] = read_json(os.path.join(directory, fname))
    return meta

def lookup_meta(meta, element_name, key='name', subdict=None):
    if key == 'name' and '_index' in element_name:
        return f'{element_name.replace("_index", "").capitalize()} Index'
    try:
        if subdict is not None and subdict in meta:
            found = meta[subdict][element_name]
        else:
            found = meta[element_name]
        if len(key) > 0:
            return found[key]
        return found
    except (KeyError, TypeError):
        return element_name
    
def load_database(fname):
    database = pd.read_pickle(fname)
    if hasattr(database, 'sparse'): # convert sparse databases to regular ones
        old_shape = database.shape
        database = database.sparse.to_dense()
        assert old_shape == database.shape
        for col in database.columns:
            try:
                fl = database[col].astype(float)
                database[col] = fl
            except Exception as e:
                pass
        database['environment'] = 'unknown'
    return database
    
def identify_property_meta(database, given_meta=None):
    properties_meta = {}
    if given_meta is not None and isinstance(given_meta, dict) and 'properties' in given_meta: # assess columns defined in meta
        cols_to_rate = [ key for key in given_meta['properties'] if key in database.select_dtypes('number').columns ]
    else: # assess all numeric columns
        cols_to_rate = database.select_dtypes('number').columns
    if len(cols_to_rate) < 1:
        raise RuntimeError('No rateable properties found!')
    for col in cols_to_rate:
        meta = lookup_meta(given_meta, col, '', 'properties')
        if not isinstance(meta, dict):
            # TODO improve by looking up group from properly characterized popular metrics in PWC and OpenML
            meta = { "name": col, "shortname": col[:4], "unit": "number", "group": "Quality", "weight": 1.0 / len(cols_to_rate) }
        properties_meta[col] = meta
    return properties_meta

def prop_dict_to_val(df, key='value'):
    if hasattr(df, "map"): # for newer pandas versions
        return df.map(lambda val: val[key] if isinstance(val, dict) and key in val else val)
    return df.applymap(lambda val: val[key] if isinstance(val, dict) and key in val else val)

def drop_na_properties(df):
    valid_cols = prop_dict_to_val(df).dropna(how='all', axis=1).columns
    return df[valid_cols]

def find_sub_db(database, dataset=None, task=None, environment=None):
    if dataset is not None:
        database = database[database['dataset'] == dataset]
    if task is not None:
        database = database[database['task'] == task]
    if environment is not None:
        database = database[database['environment'] == environment]
    return drop_na_properties(database) # drop columns with full NA

def find_relevant_metrics(database, meta):
    print('    search relevant metrics')
    all_metrics = {}
    x_default, y_default = {}, {}
    to_delete = []
    properties_meta = identify_property_meta(meta, database)
    for ds in pd.unique(database['dataset']):
        subds = find_sub_db(database, ds)
        for task in pd.unique(database[database['dataset'] == ds]['task']):
            lookup = (ds, task)
            subd = find_sub_db(subds, ds, task)
            metrics = {}
            for col, meta in properties_meta.items():
                if col in subd.columns or ('independent_of_task' in meta and meta['independent_of_task'] and col in subds.columns):
                    val = properties_meta[col]
                    metrics[col] = (val['weight'], val['group']) # weight is used for identifying the axis defaults
            if len(metrics) < 2:
                to_delete.append(lookup)
            else:
                # TODO later add this, such that it can be visualized
                # metrics['resource_index'] = {sum([weight for (weight, group) in metrics.values() if group != 'Performance']), 'Resource'}
                # metrics['quality_index'] = {sum([weight for (weight, group) in metrics.values() if group == 'Performance']), 'Performance'}
                # metrics['compound_index'] = {1.0, 'n.a.'}
                weights, groups = zip(*list(metrics.values()))

                argsort = np.argsort(weights)
                groups = np.array(groups)[argsort]
                metrics = np.array(list(metrics.keys()))[argsort]
                # use most influential Performance property on y-axis
                if 'Performance' not in groups:
                    raise RuntimeError(f'Could not find quality property for {lookup}!')
                y_default[lookup] = metrics[groups == 'Performance'][-1]
                if 'Resources' in groups: # use the most influential resource property on x-axis
                    x_default[lookup] = metrics[groups == 'Resources'][-1]
                elif 'Complexity' in groups: # use most influential complexity
                    x_default[lookup] = metrics[groups == 'Complexity'][-1]
                else:
                    try:
                        x_default[lookup] = metrics[groups == 'Performance'][-2]
                    except IndexError:
                        print(f'No second Performance property and no Resources or Complexity properties were found for {lookup}!')
                        to_delete.append(lookup)
                all_metrics[lookup] = metrics
    drop_rows = []
    for (ds, task) in to_delete:
        print(f'Not enough numerical properties found for {task} on {ds}!')
        try:
            del(all_metrics[(ds, task)])
            del(x_default[(ds, task)])
            del(y_default[(ds, task)])
        except KeyError:
            pass
        drop_rows.extend( find_sub_db(database, ds, task).index.to_list() )
    database = database.drop(drop_rows)
    database = database.reset_index(drop=True)
    return database, all_metrics, x_default, y_default