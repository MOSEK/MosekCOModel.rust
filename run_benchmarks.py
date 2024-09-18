#!/usr/bin/env python
import subprocess
import os
import argparse
from pathlib import Path
import json
import csv



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
        for k in sorted(estimates):
            (est,samp) = estimates[k]

            w.writerow([k,est['mean']['point_estimate']*1e-9, est['std_dev']['point_estimate']*1e-9])
        
    

