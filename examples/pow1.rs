//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File:      pow1.rs
//!
//! Purpose: Demonstrates how to solve the problem
//! ```
//! maximize x^0.2*y^0.8 + z^0.4 - x
//!       st x + y + 0.5z = 2
//!          x,y,z >= 0
//! ```

extern crate mosekcomodel;
use expr::const_expr;
use mosekcomodel::*;

fn main() {
    let mut model = Model::new(Some("pow1"));
    let x  = model.variable(Some("x"), 3);
    let x3 = model.variable(None,unbounded());
    let x4 = model.variable(None,unbounded());

    // Create the linear constraint
    let aval = &[1.0, 1.0, 0.5];
    model.constraint(None, &x.clone().dot(aval.to_vec()), equal_to(2.0));

    // Create the conic constraints
    model.constraint(None,&vstack![x.clone().index([0..2]), x3.clone().with_shape(&[1])], in_power_cone(3,&[0.2,0.8]));
    model.constraint(None,&vstack![x.clone().index([2..3]),const_expr(&[1], 1.0),x4.clone().with_shape(&[1])],in_power_cone(3,&[0.4,0.6]));

    // Set the objective function
    let cval = &[1.0, 1.0, -1.0];
    model.objective(None,Sense::Maximize, & cval.to_vec().dot(vstack![x3.clone().with_shape(&[1]), x4.clone().with_shape(&[1]), x.clone().index([0..1])]));

    // Solve the problem
    model.solve();

    // Get the linear solution values
    let solx = model.primal_solution(SolutionType::Default,&x).unwrap();
    println!("x, y, z = {}, {}, {}", solx[0], solx[1], solx[2]);
}

#[test]
fn test() {
    main();
}
