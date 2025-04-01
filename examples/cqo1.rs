//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File: cqo1.rs
//!
//! Purpose: Demonstrates how to solve the problem
//! ```math
//! minimize y1 + y2 + y3
//! such that
//!          x1 + x2 + 2.0 x3  = 1.0
//!                  x1,x2,x3 >= 0.0
//! and
//!          (y1,x1,x2) in C_3,
//!          (y2,y3,x3) in K_3
//! ```
//! where `C_3` and `K_3` are respectively the quadratic and
//! rotated quadratic cone of size 3 defined as
//! ```math
//!     C_3 = { z1,z2,z3 :      z1 >= sqrt(z2^2 + z3^2) }
//!     K_3 = { z1,z2,z3 : 2 z1 z2 >= z3^2              }
//! ```

extern crate mosekcomodel;
use mosekcomodel::*;

fn main() {
    let mut m = Model::new(Some("cqo1"));
    let x = m.variable(Some("x"), greater_than(vec![0.0;3]));
    let y = m.variable(Some("y"), 3);

    // Create the aliases
    //      z1 = [ y[0],x[0],x[1] ]
    //  and z2 = [ y[1],y[2],x[2] ]

    // TODO: Variable.vstack(y[0], x[0..2]);
    let z1 = Variable::vstack(&[&y.index(0..1), &x.index(0..2)]);
    let z2 = Variable::vstack(&[&y.index(1..3), &x.index(2..3)]);

    // Create the constraint
    //      x[0] + x[1] + 2.0 x[2] = 1.0
    let aval = &[1.0, 1.0, 2.0];
    let _ = m.constraint(Some("lc"), aval.dot(&x), equal_to(1.0));

    // Create the constraints
    //      z1 belongs to C_3
    //      z2 belongs to K_3
    // where C_3 and K_3 are respectively the quadratic and
    // rotated quadratic cone of size 3, i.e.
    //                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
    //  and  2.0 z2[0] z2[1] >= z2[2]^2
    let qc1 = m.constraint(Some("qc1"), &z1, in_quadratic_cone());
    let _qc2 = m.constraint(Some("qc2"), &z2, in_rotated_quadratic_cone());

    // Set the objective function to (y[0] + y[1] + y[2])
    m.objective(Some("obj"), Sense::Minimize, y.sum());

    // Solve the problem
    m.write_problem("cqo1.task");
    m.solve();

    // Get the linear solution values

    let solx = m.primal_solution(SolutionType::Default,&x);
    let soly = m.primal_solution(SolutionType::Default,&y);
    println!("x = {:?}", solx);
    println!("y = {:?}", soly);

    // Get conic solution of qc1
    let qc1lvl = m.primal_solution(SolutionType::Default,&qc1);
    let qc1sn  = m.dual_solution(SolutionType::Default,&qc1);

    println!("qc1 levels = {:?}", qc1lvl);
    println!("qc1 dual conic var levels = {:?}", qc1sn);
}

#[test]
fn test() { main() }
