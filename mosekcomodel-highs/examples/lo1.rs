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
//!          0 < x1 < 10
//!          x0,x2,x3 > 0,
//! ```
extern crate mosekcomodel;

use mosekcomodel::*;
use mosekcomodel_highs::Model;

fn lo1() -> (SolutionStatus,SolutionStatus,Result<Vec<f64>,String>) {
    let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
    let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
    let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
    let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];

    // Create a model with the name 'lo1'
    let mut m = Model::new(Some("lo1"));
    // Create variable 'x' of length 4
    let x = m.variable(Some("x0"), nonnegative().with_shape(&[4]));

    // Create constraints
    let _ = m.constraint(None, x.index(1), less_than(10.0));
    let _ = m.constraint(Some("c1"), x.dot(a0), equal_to(30.0));
    let _ = m.constraint(Some("c2"), x.dot(a1), greater_than(15.0));
    let _ = m.constraint(Some("c3"), x.dot(a2), less_than(25.0));

    // Set the objective function to (c^t * x)
    m.objective(Some("obj"), Sense::Maximize, x.dot(c));

    // Solve the problem
    //m.write_problem("lo1-nosol.ptf");
    m.solve();

    // Get the solution values
    let (psta,dsta) = m.solution_status(SolutionType::Default);
    let xx = m.primal_solution(SolutionType::Default,&x);

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

    println!("xx : {:?}", xx);
    println!("tol : {}", (a0.iter().zip(xx.iter()).map(|(&a,&b)| a*b).sum::<f64>()-30.0).abs());

    assert!((a0.iter().zip(xx.iter()).map(|(&a,&b)| a*b).sum::<f64>()-30.0).abs() < 1e-7);
    assert!(a1.iter().zip(xx.iter()).map(|(&a,&b)| a*b).sum::<f64>() >= 15.0);
    assert!(a2.iter().zip(xx.iter()).map(|(&a,&b)| a*b).sum::<f64>() <= 25.0);
}
