//!
//!  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//! 
//!  File: lj-outer.rs
//!
//!  Computes the Löwner-John outer ellipsoid for a convex polygon. 
//!
extern crate mosekcomodel;

use mosekcomodel::*;
use mosekcomodel_mosek::Model;
use mosekcomodel::matrix::{speye,dense};

/// Purpose: Models the hypograph of the n-th power of the
/// determinant of a positive definite matrix. See [1,2] for more details.
///
///   The convex set (a hypograph)
///
///   C = { (X, t) ∈ S^n_+ x R |  t ≤ det(X)^{1/n} },
///
///   can be modeled as the intersection of a semidefinite cone
///
///   | X   Z       |
///   |             | ≽ 0
///   | Z^T Diag(Z) |  
///
///   and a geometric mean bound
///
///   t <= (Z11*Z22*...*Znn)^{1/n} 
#[allow(non_snake_case)]
fn det_rootn(name : Option<&str>, M : &mut Model, t : Variable<0>, n : usize) -> Variable<2> {
    // Setup variables
    let Y = M.variable(name, in_psd_cone().with_dim(2*n));

    // Setup Y = [X, Z; Z^T , diag(Z)]
    let X  = Y.index([0..n, 0..n]);
    let Z  = Y.index([0..n,   n..2*n]);
    let DZ = Y.index([n..2*n, n..2*n]);


    // Z is lower-triangular
    _ = M.constraint(Some("triu(Z)=0"), Z.triuvec(false), equal_to(vec![0.0; n*(n-1)/2].as_slice()));
    // DZ = Diag(Z)
    _ = M.constraint(Some("DZ=Diag(Z)"), DZ.sub(Z.mul_elem(speye(n))).reshape(&[n*n]), equal_to(vec![0.0; n*n]));
    // (Z11*Z22*...*Znn) >= t^n
    _ = M.constraint(name,vstack!(DZ.diag().to_expr(),t.reshape(&[1])), in_geometric_mean_cone());

    // Return an n x n PSD variable which satisfies t <= det(X)^(1/n)
    X
}

///  The outer ellipsoidal approximation to a polytope given
///  as the convex hull of a set of points
///
///    S = conv{ x1, x2, ... , xm }
///
///  minimizes the volume of the enclosing ellipsoid,
///
///    { x | || P*x-c ||_2 <= 1 }
///
///  The volume is proportional to det(P)^{-1/n}, so the problem can
///  be solved as
///
///    maximize         t
///    subject to       t       <= det(P)^(1/n)
///                || P*xi - c ||_2 <= 1,  i=1,...,m
///                P is PSD.
#[allow(non_snake_case)]
fn lowner_john_outer<const N : usize>(x : &[[f64;N]]) -> Option<(SolutionStatus,SolutionStatus,Vec<f64>,Vec<f64>)> {
    let mut M = Model::new(Some("lownerjohn_outer"));
    M.set_log_handler(|msg| print!("{}",msg)); 

    let m = x.len();
    let n = N;

    // Setup variables
    let t = M.variable(Some("t"), nonnegative());
    let P = det_rootn(Some("det_rootn"),&mut M, t.clone(), n);
    let c = M.variable(Some("c"), unbounded().with_shape(&[1,n]));
    let x = dense([m, n], x.iter().flat_map(|row| row.iter()).cloned().collect::<Vec<f64>>());

    // (1, Px-c) in cone
    _ = M.constraint(Some("qc"),
                     hstack![ Expr::from(vec![1.0; m]).reshape(&[m,1]), 
                               x.mul(&P).sub(c.repeat(0,m))],
                     in_quadratic_cones(&[m,n+1],1));

    // Objective: Maximize t
    M.objective(None,Sense::Maximize, &t);
    M.solve(); 

    let Psol = M.primal_solution(SolutionType::Default, &P);
    let csol = M.primal_solution(SolutionType::Default, &c);
    //P, c = P.level(), c.level()
    let (psta,dsta) = M.solution_status(SolutionType::Default);
    if let (Ok(P),Ok(c)) = (Psol,csol) {
        Some((psta,dsta,P,c))
    }
    else {
        None 
    }
}

#[allow(non_snake_case)]
fn main() {
    let points = &[[0., 0.], [1., 3.], [5.5, 4.5], [7., 4.], [7., 1.], [3., -2.]];

    if let Some((psta,dsta,P,c)) = lowner_john_outer(points) {
        println!(" Outer ellipsoid Status = {:?}/{:?}",psta,dsta);
        println!(" P = | {:4.2} {:4.2} |, c = | {:4.2} |",P[0],P[1],c[0]);
        println!("     | {:4.2} {:4.2} |      | {:4.2} |",P[2],P[3],c[1]);

        println!("P should transform all points into the unit-ball:");
        for p in points {
            let (tx,ty) = (P[0]*p[0]+P[1]*p[1] - c[0],
                           P[2]*p[0]+P[3]*p[1] - c[1]);
            println!("\tP*{:?}-c -> ({:.4},{:.4}), norm = {:.4}",p,tx,ty,(tx*tx+ty*ty).sqrt());
        }
    }
    else {
        println!("Failed to solve outer ellipsoid problem");
    }
    
}

#[test]
fn test() { main() }
