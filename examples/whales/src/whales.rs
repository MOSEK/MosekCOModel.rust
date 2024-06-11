//!
//!  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//! 
//!  File: whales.rs
//!
//!  # Purpose
//!  Computes the minimal ellipsoid containing a set of ellipsoids
//!  References:
//!    [1] "Lectures on Modern Optimization", Ben-Tal and Nemirovski, 2000.    
//!    [2] "MOSEK modeling manual", 2013
//!    [3] "Convex Optimization" Boyd and Vandenberghe, 2004
//!
//!
//!  # The Model
//! 
//!  We wish to find the minimal (by volume) ellipsoid containing a set of other ellipsoids.
//!
//!  
//!
//!
//!  The containing ellipsoid is parameterized as 
//!  ```math 
//!  Eb = { x: ‖Ax+b‖₂ ≤ 1 }, A ∊ S^n_+
//!  ```
//!  and each of the `m` ellipsoids to be surrounded are parameteized as 
//!  ```math
//!  Ei = { x'A_i x + b_i'x + c_i ≤ 0 }, x ∊ R^n
//!  ```
//!  
//!  From [3] we can formulate the model as 
//!  ```math
//!  min log det A^{-1}
//!  such that 
//!     t_i ∊ n R^m_+
//!     A ∊ S^n_+
//!     b ∊ R^n
//!     | A²-t_i A_i     Ab-t_i b_i  0     |
//!     | (Ab-t_i b_i)'  -1-t_i c_i  (Ab)' | ∊ S^{2n+1}_-
//!     | 0              Ab          -A²   |
//!  ```
//!  Since `A²` is PSD, we can safely replace `A²` and `Ab` by a variables `P` and `q` to get
//!  ```math
//!  min log det A^{-1}
//!  such that 
//!     t_i ∊ R^m_+
//!     P ∊ S^n_+
//!     q ∊ R^n
//!     | P-t_i A_i      q-t_i b_i  0   |
//!     | (q-t_i b_i)'  -1-t_i c_i  q'  | ∊ S^{2n+1}_-
//!     | 0              q          -P  |
//!  ```


use mosekmodel::matrix::{dense,speye};
use mosekmodel::*;
use itertools::izip;

/// Structure defining an ellipsoid as
/// 1.
///     ```math 
///     { x | ‖ Px+q ‖₂ ≤ 1 }
///     ```
/// 2. It can be alternatively represented as 
///     ```math 
///     x'Ax + bx + c ≤ 0
///     ```
///     with
///     ```math 
///     A = P²
///     b = 2Pqx
///     c = q'q-1
///   ```
/// 3. or, as a third alternative as 
///     ```math
///     { Zu+w | ‖ u ‖₂ ≤ 1 }
///     ```
///     where 
///     ```math
///     Z = P^{-1}
///     w = -P^{-1}q
///     ```
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Ellipsoid<const N : usize> {
    P : [ [ f64; N ] ; N ],
    q : [ f64; N ]
}

#[allow(non_snake_case)]
impl<const N : usize> Ellipsoid<N> {
    pub fn new(P : &[[f64;N];N], q : &[f64;N]) -> Ellipsoid<N> { Ellipsoid{P : *P, q : *q } }
    pub fn get_P(&self) -> &[ [ f64; N ]; N ] { &self.P }
    pub fn get_q(&self) -> &[f64; N] { &self.q }

//    /// Get `Z`,`w`  representation of the ellipsis, where
//    /// ```math 
//    /// { Zx+w : ‖ x ‖₂ ≤ 1 }
//    /// ```
//    pub fn get_Zw(&self) -> ( [[f64;N];N],[f64;N] ) {
//        let Z = [[0.0; N];N];
//        let w = [0.0;N];
//
//        
//
//
//        (Z,w)
//    }

    /// For alternative parameterization
    /// ```math
    /// { x'Ax + b'x + c ≤ 0 }
    /// ```
    /// get the values of `A`, `b` and `c`, which will given by expanding 
    /// ```math 
    /// ‖ Px+q ‖₂ ≤ 1
    /// ```
    /// into 
    /// ```math 
    /// x'P²x + 2Pqx + q'q-1 ≤ 0
    /// ```
    /// Implying that
    /// ```math 
    /// A = P²
    /// b = 2Pq
    /// c = q'q-1
    /// ```
    pub fn get_Abc(&self) -> ([[f64;N];N],[f64;N],f64) {
        (self.get_A(),
         self.get_b(),
         self.get_c())
    }

    // A = P²
    pub fn get_A(&self) -> [ [ f64; N ]; N ] { 
        let mut Pt = [[0.0;N];N];
        for i in 0..N {
            for j in 0..N {
                Pt[i][j] = self.P[j][i];
            }
        }
        
        let mut res = [[0.0; N]; N];
       
        for (res_row,P_row) in izip!(res.iter_mut(),self.P.iter()) {
            for (r,P_col) in izip!(res_row.iter_mut(),Pt.iter()) {
                *r = izip!(P_row.iter(),P_col.iter()).map(|(&a,&b)| a*b).sum();
            }
        }

        res
    }

    // b = 2Pq
    pub fn get_b(&self) -> [f64; N] { 
        let mut res = [0.0;N];
        self.P.iter()
            .zip(std::iter::repeat(self.q))
            .zip(res.iter_mut())
            .for_each(|((Pi,q),r)| *r = Pi.iter().zip(q.iter()).map(|(&Pij,&qi)| 2.0*Pij*qi).sum());
        res
    }

    // c = q'q-1
    pub fn get_c(&self) -> f64    { self.q.iter().map(|v| v*v).sum::<f64>() - 1.0}
}

#[allow(non_snake_case)]
pub fn ellipsoid_contains<const N : usize>
(   M : & mut Model,
    P : &Variable<2>, 
    q : &Variable<1>, 
    e : &Ellipsoid<N>) {

    let (A,b,c) = e.get_Abc();

    let Pshp = P.shape();
    let qshp = q.shape();
    if Pshp[0] != Pshp[1] || qshp[0] != Pshp[0] {
        panic!("Invalid or mismatching P and/or q");
    }
    let n = qshp[0];
   
    let S = M.variable(Some("S"), in_psd_cone(2*n+1));
    let S11 = (&S).slice(&[0..n,0..n]);
    let S21 = (&S).slice(&[n..n+1,0..n]).reshape(&[n]);
    let S22 = (&S).slice(&[n..n+1,n..n+1]).reshape(&[]);
    let S31 = (&S).slice(&[n+1..2*n+1,0..n]);
    let S32 = (&S).slice(&[n+1..2*n+1,n..n+1]).reshape(&[n]);
    let S33 = (&S).slice(&[n+1..2*n+1,n+1..2*n+1]);
    let tau = M.variable(Some("tau"), nonnegative());

    let A = dense(N,N,A.iter().flat_map(|arow| arow.iter()).cloned().collect::<Vec<f64>>());

    _ = M.constraint(None, &P.clone().sub(tau.clone().mul(&A))           .add(S11), zero().with_shape(&[n,n]));
    _ = M.constraint(None, &q.clone().sub(tau.clone().mul(b.as_slice())) .add(S21), zero().with_shape(&[n]));
    _ = M.constraint(None, &tau.clone().mul(c).add(1.0).neg()            .add(S22), zero());
    _ = M.constraint(None, &q.clone()                                    .add(S32), zero().with_shape(&[n]));
    _ = M.constraint(None, &P.clone().neg()                              .add(S33), zero().with_shape(&[n,n]));
    _ = M.constraint(None,                                                   &S31,  zero().with_shape(&[N,N]));
}

#[allow(non_snake_case)]
pub fn minimal_bounding_ellipsoid<const N : usize>(data : &[Ellipsoid<N>]) -> Result<([[f64;N];N],[f64;N]),String> {
    let n = N;

    let mut M = Model::new(Some("lownerjohn-outer"));
    M.set_log_handler(|msg| print!("{}",msg));

    let m = data.len();

    // Maximize log(det(P))
    let t = M.variable(Some("t"), unbounded());
    
    //let P = M.variable(Some("P"),in_psd_cone(n));
    let P = det_rootn(Some("P"), & mut M, t.clone(), n);
    let q = M.variable(Some("q"), unbounded().with_shape(&[n]));
    M.objective(None, Sense::Maximize, &t);


    for e in data.iter() {
       ellipsoid_contains(&mut M,&P,&q,e);
    }

    M.solve();

    M.write_problem("whales.ptf");

    let Psol = M.primal_solution(SolutionType::Default, &P)?;
    let qsol = M.primal_solution(SolutionType::Default, &q)?;

    let mut rP = [[0.0;N];N]; rP.iter_mut().flat_map(|row| row.iter_mut()).zip(Psol.iter()).for_each(|(t,&s)| *t = s);
    let mut rq = [0.0;N]; rq.clone_from_slice(qsol.as_slice());

    Ok( (rP,rq) )
}


/// # Arguments
/// - `es` List of ellipsoids
///
/// # Returns
/// The minimal bounding ellipsoid is parameterized as `‖Px+q‖ ≤ 1`, and the returned values are
/// - `P_sq` an `n x n` martrix in row major format, which is the square of `P`.
/// - `Pq` an `n` vector, which is the product `P*q`
/// The actual `P` and `q` can be obtained from these values.
#[allow(non_snake_case)]
pub fn outer_ellipsoid<const N : usize>(es : &[Ellipsoid<N>]) -> ([[f64;N];N], [f64;N]) {
    //if es.len() < 1 { panic!("At least two ellipsoids required"); }
    let n = N; 
    let datasize = n*(n+1)+1;
    let mut M = Model::new(Some("lownerjohn_outer"));
    M.set_log_handler(|msg| print!("{}",msg)); 

    let m = es.len();

    let t = M.variable(Some("t"), unbounded());


    let τ = M.variable(Some("tau"), unbounded().with_shape(&[m]));
    let P_q = M.variable(Some("P_q"), unbounded().with_shape(&[n]));
    let P_sq = det_rootn(Some("X_Psq"), & mut M, t.clone(), n);
    //let P_sq = M.variable(Some("Psq"), in_psd_cone(n));

    let X = M.variable(Some("X"), in_psd_cones(&[m,2*n+1,2*n+1], 1,2));

    // 1/2 x0 ... xn > | x{n+1} |
    //M.constraint(Some("vdet"),
    //             vstack![
    //                P_sq.clone().index(&[0,0])
    //             ]
    //             Expr.vstack([ Expr.mul(0.5, vA2.index([0,0])), vA2.index([1,1]), vA2.index([0,1]), vdet ]), Domain.inRotatedQCone());

    M.objective(None, Sense::Maximize, &t);

    for (i,e) in es.iter().enumerate() {
        let (Adata,b,c) = e.get_Abc();

        //let Adata : Vec<f64> = e.get_A().iter().flat_map(|v| v.iter()).cloned().collect();
        let A = dense(n,n,Adata.iter().flat_map(|r| r.iter()).cloned().collect::<Vec<f64>>());
            
        // Implement the constraint 
        // | A*τ[i]-P²     τ[i]*b-P_q  0     |
        // | (τ[i]*b-P_q)' (1+τ[i]*c)  -P_q' | ∊ S^n_+
        // | 0             -P_q        P²    |
        //let name = format!("EllipsoidBound[{}]",i+1);

        let Xi : Variable<2> = (&X).slice(&[i..i+1,0..2*n+1,0..2*n+1]).reshape(&[2*n+1,2*n+1]);
        // P²-A*τ[i] = Xi[0..n,0..n]
        _ = M.constraint(Some(format!("EllipsBound[{}][1,1]",i+1).as_str()), 
                         &P_sq.clone().add(τ.clone().index(i).mul(&A))
                            .sub(Xi.clone().slice(&[0..n,0..n])),
                         zeros(&[n,n]));
        // P_q-τ[i]*b = Xi[n..n+1,0..n]]
        _ = M.constraint(Some(format!("EllipsBound[{}][2,1]",i+1).as_str()), 
                         & b.mul_right(τ.index(i)).sub(P_q.clone())
                            .sub(Xi.clone().slice(&[n..n+1,0..n]).reshape(&[n])),
                         zeros(&[n]));
        // -(1+τ[i]*c) = Xi[n,n]
        _ = M.constraint(Some(format!("EllipsBound[{}][2,2]",i+1).as_str()), 
                         &τ.index(i).mul(c).add(1.0)
                            .sub(Xi.clone().index(&[n,n])),
                         zero());
        // 0 = Xi[n+1..2n+1,0..n]
        _ = M.constraint(Some(format!("EllipsBound[{}][3,1]",i+1).as_str()),
                         &Xi.clone().slice(&[n+1..2*n+1,0..n]),
                         zero().with_shape(&[n,n]));
        // P_q = Xi[n+1..2n+1,n..n+1]
        _ = M.constraint(Some(format!("EllipsBound[{}][3,2]",i+1).as_str()),
                         &P_q.clone().neg()
                             .sub(Xi.clone().slice(&[n+1..2*n+1,n..n+1]).reshape(&[n])),
                         zeros(&[n]));
        // P²= Xi[n+1..2n+1,n+1..2n+1]
        _ = M.constraint(Some(format!("EllipsBound[{}][3,3]",i+1).as_str()),
                         &P_sq.clone()
                            .sub(Xi.clone().slice(&[n+1..2*n+1, n+1..2*n+1])),
                         zeros(&[n,n]));
    }

    M.solve();
    M.write_problem("whales.ptf");

    match M.solution_status(SolutionType::Default) {
        (SolutionStatus::Optimal,SolutionStatus::Optimal) => {},
        _ => panic!("Solution not optimal")
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
///   ```math
///   C = { (X, t) ∈ S^n_+ x R |  t ≤ det(X)^{1/n} },
///   ```
///   can be modeled as the intersection of a semidefinite cone
///
///   ```math
///   | X   Z       |
///   |             | ≽ 0
///   | Z^T Diag(Z) |  
///   ```
///
///   and a geometric mean bound
///
///   ```math
///   t <= (Z11*Z22*...*Znn)^{1/n} 
///   ```
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
    _ = M.constraint(Some("DZ=Diag(Z)"), &DZ.clone().sub(Z.mul_elem(speye(n))), equal_to(dense(n,n,vec![0.0; n*n])));
    // (Z11*Z22*...*Znn) >= t^n
    _ = M.constraint(name,&vstack![DZ.clone().diag(),t.reshape(&[1])], in_geometric_mean_cone(n+1));

    X
}

#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use super::*;
    use crate::utils2d::*;

    fn ellipse_from_param(dx : f64, dy : f64, sx : f64, sy : f64, theta : f64) -> Ellipsoid<2> {
        let A = [ [ sx*theta.cos(), sy*theta.sin()], [-sx*theta.sin(), sy*theta.cos() ] ];
        let b = [ dx, dy ];

        Ellipsoid::new(&A,&b)
    }

    //xA'Ax + 2Abx + b'b-1 = 0

    #[test]
    fn test() {
        let ellipses : &[Ellipsoid<2>] = &[
            Ellipsoid{P : [[1.09613, -0.236851], [-0.236851, 0.539075]], q : [ 0.596594, 1.23438] },
            Ellipsoid{P : [[1.01769, -0.613843], [-0.613843, 2.52996 ]], q : [-1.74633, -0.568805] },
            Ellipsoid{P : [[1.26487,  0.319239], [ 0.319239, 1.28526 ]], q : [-0.856775, 1.29365] },
            Ellipsoid{P : [[0.926849,-0.339339], [-0.339339, 1.19551 ]], q : [ 0.452287, 0.575005] },
            Ellipsoid{P : [[0.819939,-0.0866013],[-0.0866013,0.824379]], q : [-0.985105,-1.6824] },
            Ellipsoid{P : [[0.417981,-0.0699427],[-0.0699427,1.61654 ]], q : [-1.73581,  0.118404] },

            //Ellipsoid{A : [[1.2576, -0.3873], [-0.3873,0.3467]], b : [ 0.2722,  0.1969], c : 0.1831},
            //Ellipsoid{A : [[1.4125, -2.1777], [-2.1777,6.7775]], b : [-1.228,  -0.0521], c : 0.3295},
            //Ellipsoid{A : [[1.7018,  0.8141], [ 0.8141,1.7538]], b : [-0.4049,  1.5713], c : 0.2077},
            //Ellipsoid{A : [[0.9742, -0.7202], [-0.7202,1.5444]], b : [ 0.0265,  0.5623], c : 0.2362},
            //Ellipsoid{A : [[0.6798, -0.1424], [-0.1424,0.6871]], b : [-0.4301, -1.0157], c : 0.3284},
            //Ellipsoid{A : [[0.1796, -0.1423], [-0.1423,2.6181]], b : [-0.3286,  0.557 ], c : 0.4931} 
        ];

        let (P,q) = minimal_bounding_ellipsoid(ellipses);

        println!("P = {:?}",P);
        println!("q = {:?}",q);

        // Now, bonud ellipsoid is defined by A,b, with
        // A² = P => P = sqrt(A)
        // Ab = q => A\q
    
        
    }
}
