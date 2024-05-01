//
//  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
//  File:      lo1.rs
//
//  Purpose: Demonstrates how to solve the problem
//
//  maximize 3*x0 + 1*x1 + 5*x2 + x3
//  such that
//           3*x0 + 1*x1 + 2*x2        = 30,
//           2*x0 + 1*x1 + 3*x2 + 1*x3 > 15,
//                  2*x1 +      + 3*x3 < 25
//  and
//           x0,x1,x2,x3 > 0,
//           0 < x1 < 10
//
extern crate mosekmodel;

use mosekmodel::{SolutionType,Model,Sense,greater_than,less_than,equal_to,ModelItemIndex};
use mosekmodel::expr::*;

fn main() {
    let a0 = vec![ 3.0, 1.0, 2.0, 0.0 ];
    let a1 = vec![ 2.0, 1.0, 3.0, 1.0 ];
    let a2 = vec![ 0.0, 2.0, 0.0, 3.0 ];
    let c  = vec![ 3.0, 1.0, 5.0, 1.0 ];

    // Create a model with the name 'lo1'
    let mut m = Model::new(Some("lo1"));
    // Create variable 'x' of length 4
    let x = m.variable(Some("x"), greater_than(vec![0.0,0.0,0.0,0.0]));

    // Create constraints
    let _ = m.constraint(None, &x.index(1), less_than(10.0));
    let _ = m.constraint(Some("c1"), &x.clone().dot(a0.as_slice()), equal_to(30.0));
    let _ = m.constraint(Some("c2"), &x.clone().dot(a1.as_slice()), greater_than(15.0));
    let _ = m.constraint(Some("c3"), &x.clone().dot(a2.as_slice()), less_than(25.0));

    // Set the objective function to (c^t * x)
    m.objective(Some("obj"), Sense::Maximize, &x.clone().dot(c.as_slice()));

    // Solve the problem
    m.write_problem("lo1-nosol.ptf");
    m.solve();

    m.write_problem("lo1.ptf");

    // Get the solution values
    let (psta,dsta) = m.solution_status(SolutionType::Default);
    println!("Status = {:?}/{:?}",psta,dsta);
    let xx = m.primal_solution(SolutionType::Default,&x);
    println!("x = {:?}", xx);
}

