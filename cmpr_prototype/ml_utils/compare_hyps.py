"""
Argue two JSON files. This script will print out the values of all keys that
are different between the two json files.

    $ python3 compare_hyps.py file0.json file1.json

Prints:
    my_key:  value0 -- value1
    my_key2: value0 -- value1
"""

import json
import sys

if __name__=="__main__":
    json_data = []
    for arg in sys.argv[1:]:
        if ".json" in arg:
            with open(arg, "r") as f:
                data = json.load(f)
                json_data.append(data)

    keys = set()
    for d in json_data:
        keys = keys.union(set(d.keys()))
    keys = sorted(list(keys))

    for k in keys:
        all_equal = True
        for i,data in enumerate(json_data):
            if k not in data: data[k] = "UNDEFINED"
            if i == 0: val = data[k]
            elif val != data[k]: all_equal = False
        if not all_equal:
            s = str(k) + ": "
            for data in json_data:
                s += str(data[k]) + " --- "
            s = s[:-4]
            print(s)

