#!/usr/bin/env python
import subprocess
import os
import argparse
from pathlib import Path
import json
import csv


allexamp = [
    'axispermute-dense-10',
    'axispermute-sparse-10',
    'mul-dense-dense-fwd',
    'mul-dense-dense-rev',
    'mul-dense-sparse-fwd',
    'mul-dense-sparse-rev',
    'mul-diag-dense-dense-fwd',
    'mul-diag-dense-dense-rev',
    'mul-diag-dense-sparse-fwd',
    'mul-diag-dense-sparse-rev',
    'mul-diag-sparse-dense-fwd',
    'mul-diag-sparse-dense-rev',
    'mul-diag-sparse-sparse-fwd',
    'mul-diag-sparse-sparse-rev',
    'mul-par-dense-dense-fwd',
    'mul-par-dense-dense-rev',
    'mul-par-dense-sparse-fwd',
    'mul-par-dense-sparse-rev',
    'mul-par-sparse-dense-fwd',
    'mul-par-sparse-dense-rev',
    'mul-par-sparse-sparse-fwd',
    'mul-par-sparse-sparse-rev',
    'mul-sparse-dense-fwd',
    'mul-sparse-dense-rev',
    'mul-sparse-sparse-fwd',
    'mul-sparse-sparse-rev',
    'repeat-dense-0-256-3',
    'repeat-dense-1-256-3',
    'repeat-dense-2-256-3',
    'repeat-sparse-0-374-3',
    'repeat-sparse-1-374-3',
    'repeat-sparse-2-374-3',
    'stack-dense-0-256',
    'stack-dense-1-256',
    'stack-dense-2-256',
    'stack-sparse-0-374',
    'stack-sparse-1-374',
    'stack-sparse-2-374',
    'sumon-dense-10-012',
    'sumon-dense-10-135',
    'sumon-dense-10-345',
    'sumon-sparse-10-012',
    'sumon-sparse-10-135',
    'sumon-sparse-10-345',
]


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--rerun', default=False, action="store_true")
    P.add_argument('--no-rebuild', default=False, action="store_true")
    P.add_argument('--output','-o',default=Path(__file__).parent.joinpath('target','benchmark-data.json'))
    P.add_argument('--summary','-s',default=Path(__file__).parent.joinpath('target','benchmark-summary.csv'))

    a = P.parse_args()

    if a.rerun:
        if a.no_rebuild:
            blddir = Path(__file__).parent.joinpath('target','release','deps')
            candidates = [ (blddir.joinpath(f),os.stat(blddir.joinpath(f)).st_ctime) for f in os.listdir(blddir) if f.startswith('exprs-') and not f.endswith('.d') ]
            candidates.sort(key=lambda v: v[1])
            executable = candidates[-1][0]
            subprocess.check_call([executable,'--bench'],cwd=str(Path(__file__).parent))
        else:
            subprocess.check_call(['cargo','bench'],cwd=str(Path(__file__).parent))

    estimates = {}
    
    for base,dirs,files in os.walk(Path(__file__).parent.joinpath('target','criterion')):
        try: 
            with open(Path(base).joinpath('sample.json'),'rb') as f:
                samp = json.load(f)
  
        except OSError:
            pass
        else:
            with open(Path(base).joinpath('estimates.json'),'rb') as f:
                est = json.load(f)
            with open(Path(base).joinpath('benchmark.json'),'rb') as f:
                bm = json.load(f)
            estimates[bm['title']] = [est,samp]

    with open(a.output,'wt') as f:
        json.dump(estimates,f)

    summaryfile = Path(a.summary)
    with open(summaryfile,"w",newline='') as f:
        if summaryfile.suffix == '.tab':
            w = csv.writer(f,delimiter='\t')
        else:
            w = csv.writer(f)
        w.writerow(['Name','Mean','Std dev.'])
        for k in allexamp:
            if k in estimates:
                (est,samp) = estimates[k]
                w.writerow([k.lower(),est['mean']['point_estimate']*1e-9, est['std_dev']['point_estimate']*1e-9])
            else:
                w.writerow([k.lower(),'',''])

        
    

