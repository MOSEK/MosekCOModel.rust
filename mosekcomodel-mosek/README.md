MOSEK solver backend for [mosekcomodel](https://crates.io/crates/mosekcomodel). This supports all features and all cones supported by `mosekcomodel`.

# Dependencies  

The crate directly depends on:
- [mosek](https://crates.io/crates/mosek),
- [mosekcomodel](https://crates.io/crates/mosekcomodel)
- [itertools]((https://crates.io/crates/itertools)

The total set of direct and indirect dependencies is quite minial.

Tests and examples depends furthermore on `criterion`, `rand` and `rand_distr`.

Graphical demo examples located in `examples/demos/...` depend on a million libraries.
