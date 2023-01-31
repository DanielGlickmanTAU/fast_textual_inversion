from dataclasses import dataclass
from typing import Dict, List, Union
from itertools import product
import random


@dataclass
class ListArgument:
    # will paste as is, without parathansis
    argument: list


def gridsearch(default_params: Dict, params_to_grid_search: Union[Dict[object, List], List[Dict[object, List]]],
               shuffle=True) -> \
        List[Dict]:
    if isinstance(params_to_grid_search, List) and len(params_to_grid_search) > 0:
        return sum([gridsearch(default_params, d) for d in params_to_grid_search], [])

    def prod(params_to_grid_search):
        values = []
        for v in params_to_grid_search.values():
            if isinstance(v, list):
                if v and isinstance(v[0], ListArgument):
                    v = [str(x.argument)[1:-1].replace(',', ' ').replace('  ', ' ') for x in v]
                values.append(v)
            elif isinstance(v, str):
                values.append([f'"{v}"'])
            elif isinstance(v, ListArgument):
                list_as_str_in_arg_parse_format = str(v.argument)[1:-1].replace(',', ' ').replace('  ', ' ')
                values.append([list_as_str_in_arg_parse_format])
            else:
                values.append([v])
        return product(*values)

    def flatten_tuples(d):
        ret = {}
        for k, v in d.items():
            if isinstance(k, tuple):
                k_1, k_2 = k
                v_1, v_2 = v
                ret[k_1] = v_1
                ret[k_2] = v_2
            else:
                ret[k] = v
        return ret

    params_as_dicts = [dict(zip(params_to_grid_search.keys(), v)) for v in prod(params_to_grid_search)]
    params_as_dicts = [flatten_tuples(d) for d in params_as_dicts]

    rets = []
    for d in params_as_dicts:
        ret = default_params.copy()
        ret.update(d)
        rets.append(ret)
    if shuffle:
        random.shuffle(rets)
    return rets
