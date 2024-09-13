#!/bin/bash

cd $(dirname $0)
export LD_LIBRARY_PATH=$(pwd)/minidist/bin
export PATH=/remote/public/linux/64-x86/rust/current/bin:$PATH

( cd ..; python3 run_benchmarks.py --rerun -s condor/result.tab )
