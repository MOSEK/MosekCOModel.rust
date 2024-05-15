//
//  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
// 
//  File: whales.rs
//
//  Computes the minimal ellipsoid containing a set of ellipsoids

struct Ellipsoid<const N : usize> {
    data : Vec<f64> // length must be `dim*(dim+1)+1`
}

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

    let τ = M.variable(Some("τ"), nonnegative().with_shape(&[m]));
    let P_sq = M.variable(Some("P²"), in_psd_cone(n));
    let P_q = M.variable(Some("P_q"), unbounded().with_shape(&[n]));

    for (i,e) in es.iter().enumerate() {
        let Adata = dense(n,n,e.get_A());
        let bdata = e.get_b();
        let c     = e.get_c();

//        X = [
//        #! format: off
//        (P² - τ[i] * A)   (P_q - τ[i] * b) zeros(n, n)
//        (P_q - τ[i] * b)' (-1 - τ[i] * c)  P_q'
//        zeros(n, n)       P_q              -P²
//        #! format: on
//    ]
//
        let name = format!("EllipsoidBound[{}]",i);
        let Xi = M.variable(Some(name.as_str()), in_psd_cone(n*2+1));
        _ = M.constraint(None, 
                         Xi.slice(&[0..n,0..n])
                         .sub( P_sq.clone().sub( τ.clone().index(i)) ),
                         equal_to(0.0).with_shape(&[n,n]));

             
        );
    }
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

