This is an OptServer backend for [mosekcomodel](https://crates.io/crates/mosekcomodel).

# Backend features:
- Linear variables and constraints,
- Integer variables,
- Solver status callbacks and integer solution callbacks
- Offloading optimization to a MOSEK OptServer instance, e.g. [solve.mosek.com](http://solve.mosek.com:30080) or a local OptServerLight server.
- HTTP or HTTPS
