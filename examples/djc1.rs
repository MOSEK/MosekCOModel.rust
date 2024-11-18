////
//   Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
//   File:      djc1.rs
//
//   Purpose: Demonstrates how to solve the problem with two disjunctions:
//
//      minimize    2x0 + x1 + 3x2 + x3
//      subject to   x0 + x1 + x2 + x3 >= -10
//                  (x0-2x1<=-1 and x2=x3=0) or (x2-3x3<=-2 and x1=x2=0)
//                  x0=2.5 or x1=2.5 or x2=2.5 or x3=2.5
////


extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::disjunction::*;

fn djc1() -> (SolutionStatus,Result<Vec<f64>,String>) {
    let model = Model::new(Some("djc1"));

    // Create variable 'x' of length 4
    let x = model.variable(Some("x"),4);

    // First disjunctive constraint
    
    model.disjunction(
        Some("D1"),
        &(term(x.clone().index(0..2).dot(vec![1.0,-2.0]), less_than(-1.0))      // x0 - 2x1 <= -1  
            .and(x.clone().index(2..4),equal_to(0.0))                         // x2 = x3 = 0
            .or( term(x.clone().index(2..4).dot(vec![1.0,-3.0]), less_than(-2.0)) // x2 - 3x3 <= -2
                 .and(x.index(0..2), equal_to(0.0)))));             // x0 = x1 = 0

    // Second disjunctive constraint
    // Array of terms reading x_i = 2.5 for i = 0,1,2,3
    let terms : Vec<Box<dyn ClauseTrait>> = 
        (0..4).map(|i| Box::new((x.clone().index(i), equal_to(2.5)))).collect::<Vec<Box<dyn ClauseTrait>>>();
    // The disjunctive constraint from the array of terms
    model.disjunction(Some("VarTerms"), &terms);

    // The linear constraint
    model.constraint(None, &x.clone().sum(), greater_than(-10.0));

    // Objective
    model.objective(None, Sense::Minimize, &x.clone().dot(vec![2.0,1.0,3.0,1.0]));

    // Useful for debugging
    model.write_problem("djc1.ptf");                                 // Save to a readable file
    //model.setLogHandler(new java.io.PrintWriter(System.out));   // Enable log output

    // Solve the problem
    model.solve();

    let (psta,dsta) = model.solution_status(SolutionType::Default);
    (psta, model.primal_solution(SolutionType::Integer, &x))
}

fn main() {
    let (solsta,res) = djc1();
    let xx = res.unwrap();
    if solsta == SolutionStatus::Optimal {
      println!("[x0,x1,x2,x3] = [{0}, {1}, {2}, {3}]", xx[0], xx[1], xx[2], xx[3]);
    }
    else {
      println!("Anoter solution status");
    }
}
