# -*- coding: utf-8 -*-
import json
import numpy as np
d = json.load(open('results_exp1_2.json', encoding='utf-8'))
for scene in d:
    for row in d[scene]:
        per_seed = row['dE_per_seed']['ours']
        det = sum(1 for e in row.get('est_info', []) if e.get('detected'))
        dEs = [round(x, 2) for x in per_seed]
        print(f"{scene} s={row['strength']}: det={det}/{len(per_seed)}, dE={dEs}, s_err={row.get('strength_est_err')}")
