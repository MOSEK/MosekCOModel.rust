#!/bin/bash

BD=$(cd $(dirname $0); pwd)
export LD_LIBRARY_PATH=$BD/minidist/bin
export PATH=/remote/public/linux/64-x86/rust/current/bin:$PATH

( cd $BD/..; python3 run_benchmarks.py --rerun -s condor/result.tab )
