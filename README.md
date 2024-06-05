The `MosekModel` crate is a modeling package for building optimization models
with `Mosek.rs`. The [Mosek.rs](https://crates.io/crates/mosek) package is a
relatively thin interface in top of low-level [MOSEK](https://mosek.com)
optimizer C API, where `MosekModel` is an attempt to create an interface that
is more like the MOSEK Fusion modelling interface.

# Design principle
`MosekModel` allows building a model of the form
```
min/max   c^t x + c_fix
such that Ax_b ∊ K_c
          x ∊ K_x  
```
That is affine expressions and conic domains of constraints and variables.

The `MosekModel` package provides functionality to build the linear expressions.

# Simple conic example
Implementing the models
```
minimize y1 + y2 + y3
such that
         x1 + x2 + 2.0 x3 = 1.0
                 x1,x2,x3 ≥ 0.0
and
         (y1,x1,x2) in C₃,
         (y2,y3,x3) in K₃
```

where `C₃` and `K₃` are respectively the quadratic and
rotated quadratic cone of size 3 defined as
```
    C₃ = { z1,z2,z3 :      z1 ≥ √(z2² + z3²) }
    K₃ = { z1,z2,z3 : 2 z1 z2 ≥ z3²          }
```

This is the included model `cqo1.rs`:

```rust
extern crate mosekmodel;
use mosekmodel::*;
use mosekmodel::expr::*;

fn main() {
    let mut m = Model::new(Some("cqo1"));
    let x = m.variable(Some("x"), greater_than(vec![0.0;3]));
    let y = m.variable(Some("y"), 3);

    // Create the aliases
    //      z1 = [ y[0],x[0],x[1] ]
    //  and z2 = [ y[1],y[2],x[2] ]

    let z1 = Variable::vstack(&[&y.index(0..1), &x.index(0..2)]);
    let z2 = Variable::vstack(&[&y.index(1..3), &x.index(2..3)]);

    // Create the constraint
    //      x[0] + x[1] + 2.0 x[2] = 1.0
    let aval = &[1.0, 1.0, 2.0];
    let _ = m.constraint(Some("lc"), &aval.dot(x.clone()), equal_to(1.0));

    // Create the constraints
    //      z1 belongs to C_3
    //      z2 belongs to K_3
    // where C_3 and K_3 are respectively the quadratic and
    // rotated quadratic cone of size 3, i.e.
    //                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
    //  and  2.0 z2[0] z2[1] >= z2[2]^2
    let qc1 = m.constraint(Some("qc1"), &z1, in_quadratic_cone(3));
    let _qc2 = m.constraint(Some("qc2"), &z2, in_rotated_quadratic_cone(3));

    // Set the objective function to (y[0] + y[1] + y[2])
    m.objective(Some("obj"), Sense::Minimize, &y.clone().sum());

    // Solve the problem
    m.solve();

    // Get the linear solution values
    let solx = m.primal_solution(SolutionType::Default,&x);
    let soly = m.primal_solution(SolutionType::Default,&y);
    println!("x = {:?}", solx);
    println!("y = {:?}", soly);

    // Get primal and dual solution of qc1
    let qc1lvl = m.primal_solution(SolutionType::Default,&qc1);
    let qc1sn  = m.dual_solution(SolutionType::Default,&qc1);

    println!("qc1 levels = {:?}", qc1lvl);
    println!("qc1 dual conic var levels = {:?}", qc1sn);
}
```


