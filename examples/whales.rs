//
//  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
// 
//  File: whales.rs
//
//  Computes the minimal ellipsoid containing a set of ellipsoids

#[allow(mixed_script_confusables)]

use mosekmodel::matrix::{dense,speye};
use mosekmodel::*;

#[allow(non_snake_case)]
struct Ellipsoid<const N : usize> {
    A : [ [ f64; N ] ; N ],
    b : [ f64; N ],
    c : f64
}

#[allow(non_snake_case)]
impl<const N : usize> Ellipsoid<N> {
    fn get_A(&self) -> &[ [ f64; N ]; N ] { &self.A }
    fn get_b(&self) -> &[f64; N] { &self.b }
    fn get_c(&self) -> f64    { self.c }
}

/// # Arguments
/// - `es` List of ellipsoids
///
/// # Returns
/// The minimal bounding ellipsoid parameterized as `‖Px+q‖ ≤ 1`
/// - `P` an `n x n` martrix in row major format,
/// - `q` an `n` vector
#[allow(non_snake_case)]
fn outer_ellipsoid<const N : usize>(es : &[Ellipsoid<N>]) -> ([[f64;N];N], [f64;N]) {
    if es.len() < 2 { panic!("At least two ellipsoids required"); }
    let n = N; 
    let datasize = n*(n+1)+1;
    let mut M = Model::new(Some("lownerjohn_outer"));
    M.set_log_handler(|msg| print!("{}",msg)); 

    let m = es.len();

    let t = M.variable(Some("t"), unbounded());
    let τ = M.variable(Some("τ"), nonnegative().with_shape(&[m]));
    let P_q = M.variable(Some("P_q"), unbounded().with_shape(&[n]));
    let P_sq = det_rootn(Some("log(det(P²))"), & mut M, t.clone(), n);

    // LogDetConeSquare = { (t,u,X) ∊ R^(2+d²) | t ≤ u log(det(X/u)), X symmetric, u > 0 }
    // x > y exp(z/y), x,y > 0
    // y log(x/y) > z

    for (i,e) in es.iter().enumerate() {
        let Adata : Vec<f64> = e.get_A().iter().map(|v| v.iter()).flatten().cloned().collect();
        let A = dense(n,n,Adata);
        let b = e.get_b();
        let c = e.get_c();

        // Implement the constraint 
        // | P²-A*τ[i]     P_q-τ[i]*b  0    |
        // | (P_q-τ[i]*b)' (-1-τ[i]*c) P_q' | ∊ S^n_+
        // | 0             P_q         -P²  |
        let name = format!("EllipsoidBound[{}]",i+1);
        let Xi = M.variable(Some(name.as_str()), in_psd_cone(n*2+1));
        // P²-A*τ[i] = Xi[0..n,0..n]
        _ = M.constraint(Some(format!("EllipsBound[{}][1,1]",i+1).as_str()), 
                         &Xi.clone().slice(&[0..n,0..n])
                            .sub(P_sq.clone().sub(τ.clone().index(i).mul(&A))),
                         zeros(&[n,n]));
        // P_q-τ[i]*b = Xi[n..n+1,0..n]]

        _ = M.constraint(Some(format!("EllipsBound[{}][2,1]",i+1).as_str()), 
                         &Xi.clone().slice(&[n..n+1,0..n]).reshape(&[n])
                            .sub(P_q.clone().sub(b.mul_right(τ.index(i)))),
                         zeros(&[n]));
        // -(1+τ[i]*c) = Xi[n,n]
        _ = M.constraint(Some(format!("EllipsBound[{}][2,2]",i+1).as_str()), 
                         &Xi.clone().index(&[n,n]).add(τ.index(i).mul(c).add(1.0)),
                         zero());
        // 0 = Xi[n+1..2n+1,0..n]
        _ = M.constraint(Some(format!("EllipsBound[{}][3,1]",i+1).as_str()),
                         &Xi.clone().slice(&[n+1..2*n+1,0..n]),
                         zero().with_shape(&[n,n]));
        // P_q = Xi[n+1..2n+1,n..n+1]
        _ = M.constraint(Some(format!("EllipsBound[{}][3,2]",i+1).as_str()),
                         &Xi.clone().slice(&[n+1..2*n+1,n..n+1]).reshape(&[n]).sub(P_q.clone()),
                         zeros(&[n]));
        // P`= Xi[n+1..2n+1,n+1..2n+1]
        _ = M.constraint(Some(format!("EllipsBound[{}][3,3]",i+1).as_str()),
                         &Xi.clone().slice(&[n+1..2*n+1, n+1..2*n+1]).sub(P_sq.clone()),
                         zeros(&[n,n]));
    }

    let Psol  = M.primal_solution(SolutionType::Default,&P_sq).unwrap();
    let Pqsol = M.primal_solution(SolutionType::Default,&P_q).unwrap();

    let mut Psq_res = [[0.0;N];N];
    let mut Pq_res  = [0.0;N];

    Psol.iter().zip(Psq_res.iter_mut().map(|item| item.iter_mut()).flatten()).for_each(|(&s,t)| *t = s);
    Pqsol.iter().zip(Pq_res.iter_mut()).for_each(|(&s,t)| *t = s);

    (Psq_res,Pq_res)
}


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
    let Y = M.variable(name, in_psd_cone(2*n));

    // Setup Y = [X, Z; Z^T , diag(Z)]
    let X  = (&Y).slice(&[0..n, 0..n]);
    let Z  = (&Y).slice(&[0..n,   n..2*n]);
    let DZ = (&Y).slice(&[n..2*n, n..2*n]);

    // Z is lower-triangular
    _ = M.constraint(Some("triu(Z)=0"), &Z.clone().triuvec(false), equal_to(vec![0.0; n*(n-1)/2].as_slice()));
    // DZ = Diag(Z)
    _ = M.constraint(Some("DZ=Diag(Z)"), &DZ.clone().sub(Z.mul_elem(speye(n))).reshape(&[n*n]), equal_to(vec![0.0; n*n]));
    // (Z11*Z22*...*Znn) >= t^n
    _ = M.constraint(name,&vstack![DZ.clone().diag(),t.reshape(&[1])], in_geometric_mean_cone(n+1));

    // Return an n x n PSD variable which satisfies t <= det(X)^(1/n)
    X
}

fn main() {
    let ellipses = [
        Ellipsoid{A : [[1.2576, -0.3873], [-0.3873,0.3467]], b : [ 0.2722,  0.1969], c : 0.1831},
        Ellipsoid{A : [[1.4125, -2.1777], [-2.1777,6.7775]], b : [-1.228,  -0.0521], c : 0.3295},
        Ellipsoid{A : [[1.7018,  0.8141], [ 0.8141,1.7538]], b : [-0.4049,  1.5713], c : 0.2077},
        Ellipsoid{A : [[0.9742, -0.7202], [-0.7202,1.5444]], b : [ 0.0265,  0.5623], c : 0.2362},
        Ellipsoid{A : [[0.6798, -0.1424], [-0.1424,0.6871]], b : [-0.4301, -1.0157], c : 0.3284},
        Ellipsoid{A : [[0.1796, -0.1423], [-0.1423,2.6181]], b : [-0.3286,  0.557 ], c : 0.4931} ];

    let (Psq_sol,Pq_sol) = outer_ellipsoid(&ellipses);

    println!("P² = {:?}",Psq_sol);
    println!("Pq = {:?}",Pq_sol);
}
