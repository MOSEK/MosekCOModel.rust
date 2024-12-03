#!/bin/bash


BD=$(cd $(dirname $0); pwd)
RES=${1:-$BD/result.tab}
export LD_LIBRARY_PATH=$BD/minidist/bin
export PATH=/remote/public/linux/64-x86/rust/current/bin:$PATH

RES=$(cd $(dirname $RES);pwd)/$(basename $RES)
( cd $BD/..; python3 run_benchmarks.py --rerun --no-rebuild -s $RES )
