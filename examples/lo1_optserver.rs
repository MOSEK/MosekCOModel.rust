//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File:      lo1_optserver.rs
//!
//! Demonstrates how to solve a small problem using MOSEK Optserver. To test, start optserverlight
//! included with the MOSEK distro, e.g.
//! ```sh 
//! optserverlight -port 9999
//! ```
//! then run this example as 
//! ```sh 
//! lo1_optserver http://localhost:9999
//! ```
//!
extern crate mosekcomodel;

use mosekcomodel::*;

fn lo1(hostname : &str, accesstoken : Option<&str>) -> (SolutionStatus,SolutionStatus,Result<Vec<f64>,String>) {
    let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
    let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
    let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
    let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];

    // Create a model with the name 'lo1'
    let mut m = Model::new(Some("lo1"));
    m.set_log_handler(|msg| print!("{}",msg));
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
    m.put_optserver(hostname, accesstoken);
    println!("Solving on {}",hostname);
    m.solve();

    // Get the solution values
    let (psta,dsta) = m.solution_status(SolutionType::Default);
    let xx = m.primal_solution(SolutionType::Default,&x);

    (psta,dsta,m.primal_solution(SolutionType::Default,&x))
}

fn main() {
    let mut args = std::env::args();
    args.next();
    if let Some(hostname) = args.next() {
        let accesstoken = args.next();
        let (psta,dsta,xx) = lo1(hostname.as_str(),accesstoken.as_ref().map(|s| s.as_str()));
        println!("Status = {:?}/{:?}",psta,dsta);
        println!("x = {:?}", xx);
    }
    else {
        println!("Syntax: lo1_optserver HOSTNAME [ ACCESSTOKEN ]");
    }
}
