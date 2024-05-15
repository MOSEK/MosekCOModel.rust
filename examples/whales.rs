//
//  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
// 
//  File: whales.rs
//
//  Computes the minimal ellipsoid containing a set of ellipsoids

use mosekmodel::{in_exponential_cone, in_geometric_mean_cone, in_psd_cone, matrix};

struct Ellipsoid<const N : usize> {
    data : Vec<f64> // length must be `dim*(dim+1)+1`
}

#[allow(non_snake_case)]
impl<const N : usize> Ellipsoid<N> {
    fn get_A(&self) -> &[f64] { &self.data[0..N*N] }
    fn get_b(&self) -> &[f64] { &self.data[N*N..N*(N+1)] }
    fn get_c(&self) -> f64    { self.data[N*(N+1)] }
}

/// # Arguments
/// - `es` List of ellipsoids
///
/// # Returns
/// The minimal bounding ellipsoid parameterized as `‖Px+q‖ ≤ 1`
/// - `P` an `n x n` martrix in row major format,
/// - `q` an `n` vector
#[allow(non_snake_case)]
fn outer_ellipsoid<const N : usize>(es : &[Ellipsoid<N>]) -> (Vec<f64>,Vec<f64>) {
    if es.len() < 2 { panic!("At least two ellipsoids required"); }
    let n = N; 
    let datasize = n*(n+1)+1;
    if es.iter().any(|e| e.data.len() != datasize) { panic!("Invalid ellipsoid data"); }
    let mut M = Model::new(Some("lownerjohn_outer"));
    M.set_log_handler(|msg| print!("{}",msg)); 

    let m = es.len();

    let t = M.variable(Some("t"), unbounded());
    let τ = M.variable(Some("τ"), nonnegative().with_shape(&[m]));
    let P_sq = M.variable(Some("P²"), in_psd_cone(n));
    let P_q = M.variable(Some("P_q"), unbounded().with_shape(&[n]));

    // LogDetConeSquare = { (t,u,X) ∊ R^(2+d²) | t ≤ u log(det(X/u)), X symmetric, u > 0 }
    // x > y exp(z/y), x,y > 0
    // y log(x/y) > z

    for (i,e) in es.iter().enumerate() {
        let A = dense(n,n,e.get_A());
        let b = e.get_b();
        let c = e.get_c();

        // Implement the constraint 
        // | P²-A*τ[i]     P_q-τ[i]    0    |
        // | (P_q-τ[i]*b)' (-1-τ[i]*c) P_q' | ∊ S^n_+
        // | 0             P_q         -P²  |
        let name = format!("EllipsoidBound[{}]",i+1);
        let Xi = M.variable(Some(name.as_str()), in_psd_cone(n*2+1));
        _ = M.constraint(Some(format!("EllipsBound[{}][1,1]",i+1).as_str()), 
                         &Xi.clone().slice(&[0..n,0..n]).sub(P_sq.clone().sub(τ.clone().index(i))),
                         zeros(&[n,n]));
        _ = M.constraint(Some(format!("EllipsBound[{}][2,1]",i+1).as_str()), 
                         &Xi.clone().slice(&[n,n+1,0..n]).reshape(&[n]).sub( P_q.clone().sub(τ.index(i).mul(b)) ),
                         zeros(&[n]));
        _ = M.constraint(Some(format!("EllipsBound[{}][2,2]",i+1).as_str()), 
                         &Xi.clone().index(&[n,n]).add(τ.index(i).mul(c).add(1.0)),
                         zero());
        _ = M.constraint(Some(format!("EllipsBound[{}][3,1]",i+1).as_str()),
                         &Xi.clone().slice(&[n+1..2*n+1,0..n]),
                         zero().with_shape(&[n,n]));
        _ = M.constraint(Some(format!("EllipsBound[{}][3,2]",i+1).as_str()),
                         &Xi.clone().slice(n+1..2*n+1,n..n+1).reshape(&[n]).sub(P_q.clone()),
                         zeros(&[n]));
        _ = M.constraint(Some(format!("EllipsBound[{}][3,3]",i+1).as_str()),
                         &Xi.clone().slice(n+1..2*n+1, n+1..2*n+1).sub(P_sq),
                         zeros(&[n,n]));
    }


}


/// Implement convex set 
/// ```math
/// { (t,u,X) ∊ R^(2+d²) | t ≤ u log(det(X/u)), X symmetric, u > 0 }
/// ``` log(det(X/u)) = log((1/u)^n det(X)) = log((1/u)^n) + log(det(X)) = - n log(u det(X))
/// The set `{ t ≤ u log(v/u), v,u > 0 }` is equivalent to `{ v ≥ u exp(t/u), v,u > 0 }`.
/// 
/// The set 
/// ```math
/// { (t,X) ∊ R × S^n_+ | t ≤ det(X)^(1/n) }
/// ```
/// can be modeled as
/// ```math
/// | X    Z       |
/// |              | ≽ 0
/// | Z^T  Diag(Z) |  
/// t ≤ (Z11*Z22*...*Znn)^{1/n} 
/// ```
/// 
/// So the complete set can be modeled as:
/// ```math
/// | X   Z       |
/// |             | ≽ 0
/// | Z^T Diag(Z) |
/// w ≤ n (Z11*Z22*...*Znn)^{1/n}
/// t ≤ u log(w/u)
/// ```
///
/// # Argument
/// - `t` a free scalar variable
/// - `X` a symmetric variable
#[allow(non_snake_case)]
fn log_det_cone_square(M : & mut Model, t : Variable<0>, X : &Variable<2>) {
    if X.shape[0] != X.shape[1] { panic!("Invalid X argument shape"); }
    let n = X.shape[0];
    let S = M.variable(None, in_psd_cone(2*n));
    let w = M.variable(None, unbounded());
    _ = M.constraint(None, 
                     S.clone().slice(&[0..n,0..n])
                        .sub(X.clone()), 
                     zeros(&[n,n]));
    _ = M.constraint(None,
                     S.clone().slice(&[n..2*n,0..n])
                        .mul_elem(matrix::speye(n))
                        .sub(S.clone().slice(&[n..2*n,n..2*n])), 
                     zeros(&[n,n]));
    _ = M.constraint(None,
                     S.clone().slice(&[n..2*n,0..n])
                        .mul_elem(matrix::speye(n))
                        .sum_on(&[0])
                        .vstack(w),
                     in_geometric_mean_cone(n+1));
    _ = M.constraint(None,
                     vstack![ w.with_shape(&[1]), 
                              (1.0).into_expr().with_shape(&[1]),
                              t.with_shape(&[1])],
                     in_exponential_cone());
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
    _ = M.constraint(name,&vstack!(DZ.clone().diag(),t.reshape(&[1])), in_geometric_mean_cone(n+1));

    // Return an n x n PSD variable which satisfies t <= det(X)^(1/n)
    X
}

