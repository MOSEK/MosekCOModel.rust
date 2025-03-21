//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File:      lo1.rs
//!
//! Purpose: Demonstrates how to solve the problem
//! ```
//! maximize 3*x0 + 1*x1 + 5*x2 + x3
//! such that
//!          3*x0 + 1*x1 + 2*x2        = 30,
//!          2*x0 + 1*x1 + 3*x2 + 1*x3 > 15,
//!                 2*x1 +      + 3*x3 < 25
//! and
//!          x0,x1,x2,x3 > 0,
//!          0 < x1 < 10
//! ```
extern crate mosekcomodel;

use mosekcomodel::*;

fn lo1() -> (SolutionStatus,SolutionStatus,Result<Vec<f64>,String>) {
    let a0 = &[ 3.0, 1.0, 2.0, 0.0 ];
    let a1 = &[ 2.0, 1.0, 3.0, 1.0 ];
    let a2 = &[ 0.0, 2.0, 0.0, 3.0 ];
    let c  = &[ 3.0, 1.0, 5.0, 1.0 ];

    // Create a model with the name 'lo1'
    let mut m = Model::new(Some("lo1"));
    // Create variable 'x' of length 4
    let x = m.variable(Some("x"), greater_than(vec![0.0,0.0,0.0,0.0]));

    // Create constraints
    let _ = m.constraint(None, &x.index(1), less_than(10.0));
    let _ = m.constraint(Some("c1"), &x.dot(a0), equal_to(30.0));
    let _ = m.constraint(Some("c2"), &x.dot(a1), greater_than(15.0));
    let _ = m.constraint(Some("c3"), &x.dot(a2), less_than(25.0));

    // Set the objective function to (c^t * x)
    m.objective(Some("obj"), Sense::Maximize, &x.dot(c));

    // Solve the problem
    m.write_problem("lo1-nosol.ptf");
    m.solve();

    // Get the solution values
    let (psta,dsta) = m.solution_status(SolutionType::Default);
    println!("Status = {:?}/{:?}",psta,dsta);
    let xx = m.primal_solution(SolutionType::Default,&x);
    println!("x = {:?}", xx);

    (psta,dsta,m.primal_solution(SolutionType::Default,&x))
}

fn main() {
    let (psta,dsta,xx) = lo1();
    println!("Status = {:?}/{:?}",psta,dsta);
    println!("x = {:?}", xx);
}



#[cfg(test)]
#[test]
fn test() {
    let a0 = vec![ 3.0, 1.0, 2.0, 0.0 ];
    let a1 = vec![ 2.0, 1.0, 3.0, 1.0 ];
    let a2 = vec![ 0.0, 2.0, 0.0, 3.0 ];

    let (_psta,_dsta,xx) = lo1();
    let xx = xx.unwrap();
    assert!((a0.iter().zip(xx.iter()).map(|(&a,&b)| a*b).sum::<f64>()-30.0).abs() < 1e-7);
    assert!(a1.iter().zip(xx.iter()).map(|(&a,&b)| a*b).sum::<f64>() >= 15.0);
    assert!(a2.iter().zip(xx.iter()).map(|(&a,&b)| a*b).sum::<f64>() <= 25.0);
}
