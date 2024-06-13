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
use ellipsoids::*;

//   /// Structure defining an ellipsoid as
//   /// 1.
//   ///     ```math 
//   ///     { x | ‖ Px+q ‖₂ ≤ 1 }
//   ///     ```
//   /// 2. It can be alternatively represented as 
//   ///     ```math 
//   ///     x'Ax + bx + c ≤ 0
//   ///     ```
//   ///     with
//   ///     ```math 
//   ///     A = P²
//   ///     b = 2Pqx
//   ///     c = q'q-1
//   ///   ```
//   /// 3. or, as a third alternative as 
//   ///     ```math
//   ///     { Zu+w | ‖ u ‖₂ ≤ 1 }
//   ///     ```
//   ///     where 
//   ///     ```math
//   ///     Z = P^{-1}
//   ///     w = -P^{-1}q
//   ///     ```
//   #[allow(non_snake_case)]
//   #[derive(Clone)]
//   pub struct Ellipsoid<const N : usize> {
//       P : [ [ f64; N ] ; N ],
//       q : [ f64; N ]
//   }
//   
//   #[allow(non_snake_case)]
//   impl<const N : usize> Ellipsoid<N> {
//       /// Specify ellipsoid by `P` and `q` as
//       /// ```math 
//       /// { x | ‖ Px+q ‖₂ ≤ 1 }
//       /// ```
//       pub fn new(P : &[[f64;N];N], q : &[f64;N]) -> Ellipsoid<N> { Ellipsoid{P : *P, q : *q } }
//       pub fn get_Pq(&self) -> ([ [ f64; N ]; N ],[f64;N]) { (self.P,self.q) }
//   
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
//   
//       /// For alternative parameterization
//       /// ```math
//       /// { x'Ax + b'x + c ≤ 0 }
//       /// ```
//       /// get the values of `A`, `b` and `c`, which will given by expanding 
//       /// ```math 
//       /// ‖ Px+q ‖₂ ≤ 1
//       /// ```
//       /// into 
//       /// ```math 
//       /// x'P²x + 2Pqx + q'q-1 ≤ 0
//       /// ```
//       /// Implying that
//       /// ```math 
//       /// A = P²
//       /// b = 2Pq
//       /// c = q'q-1
//       /// ```
//       pub fn get_Abc(&self) -> ([[f64;N];N],[f64;N],f64) {
//           (self.get_A(),
//            self.get_b(),
//            self.get_c())
//       }
//   
//       // A = P²
//       fn get_A(&self) -> [ [ f64; N ]; N ] { 
//           let mut Pt = [[0.0;N];N];
//           for i in 0..N {
//               for j in 0..N {
//                   Pt[i][j] = self.P[j][i];
//               }
//           }
//           
//           let mut res = [[0.0; N]; N];
//          
//           for (res_row,P_row) in izip!(res.iter_mut(),self.P.iter()) {
//               for (r,P_col) in izip!(res_row.iter_mut(),Pt.iter()) {
//                   *r = izip!(P_row.iter(),P_col.iter()).map(|(&a,&b)| a*b).sum();
//               }
//           }
//   
//           res
//       }
//   
//       // b = 2Pq
//       fn get_b(&self) -> [f64; N] { 
//           let mut res = [0.0;N];
//           self.P.iter()
//               .zip(std::iter::repeat(self.q))
//               .zip(res.iter_mut())
//               .for_each(|((Pi,q),r)| *r = Pi.iter().zip(q.iter()).map(|(&Pij,&qi)| Pij*qi).sum());
//           res
//       }
//   
//       // c = q'q-1
//       fn get_c(&self) -> f64 { self.q.iter().map(|v| v*v).sum::<f64>() - 1.0 }
//   }

//   #[allow(non_snake_case)]
//   pub fn ellipsoid_contains<const N : usize>
//   (   M : & mut Model,
//       P : &Variable<2>, 
//       q : &Variable<1>, 
//       e : &Ellipsoid<N>) {
//   
//       let (A,b,c) = e.get_Abc();
//   
//       let Pshp = P.shape();
//       let qshp = q.shape();
//       if Pshp[0] != Pshp[1] || qshp[0] != Pshp[0] {
//           panic!("Invalid or mismatching P and/or q");
//       }
//       let n = qshp[0];
//      
//       let S = M.variable(Some("S"), in_psd_cone(2*n+1));
//       let S11 = (&S).slice(&[0..n,0..n]);
//       let S21 = (&S).slice(&[n..n+1,0..n]).reshape(&[n]);
//       let S22 = (&S).slice(&[n..n+1,n..n+1]).reshape(&[]);
//       let S31 = (&S).slice(&[n+1..2*n+1,0..n]);
//       let S32 = (&S).slice(&[n+1..2*n+1,n..n+1]).reshape(&[n]);
//       let S33 = (&S).slice(&[n+1..2*n+1,n+1..2*n+1]);
//       let tau = M.variable(Some("tau"), nonnegative());
//   
//       let A = dense(N,N,A.iter().flat_map(|arow| arow.iter()).cloned().collect::<Vec<f64>>());
//   
//       _ = M.constraint(None, &P.clone().sub(tau.clone().mul(&A))           .add(S11), zero().with_shape(&[n,n]));
//       _ = M.constraint(None, &q.clone().sub(tau.clone().mul(b.as_slice())) .add(S21), zero().with_shape(&[n]));
//       _ = M.constraint(None, &tau.clone().mul(c).add(1.0).neg()            .add(S22), zero());
//       _ = M.constraint(None, &q.clone()                                    .add(S32), zero().with_shape(&[n]));
//       _ = M.constraint(None, &P.clone().neg()                              .add(S33), zero().with_shape(&[n,n]));
//       _ = M.constraint(None,                                                   &S31,  zero().with_shape(&[N,N]));
//   }

#[allow(non_snake_case)]
pub fn minimal_bounding_ellipsoid<const N : usize>(data : &[Ellipsoid<N>]) -> Result<([[f64;N];N],[f64;N]),String> {
    let n = N;

    let mut M = Model::new(Some("lowner-john-outer"));
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

    M.write_problem("lj-outer-ellipsoid.ptf");

    let Psol = M.primal_solution(SolutionType::Default, &P)?;
    let qsol = M.primal_solution(SolutionType::Default, &q)?;

    let mut rP = [[0.0;N];N]; rP.iter_mut().flat_map(|row| row.iter_mut()).zip(Psol.iter()).for_each(|(t,&s)| *t = s);
    let mut rq = [0.0;N]; rq.clone_from_slice(qsol.as_slice());

    Ok( (rP,rq) )
}

#[allow(non_snake_case)]
pub fn maximal_contained_ellipsoid<const N : usize>(data : &[Ellipsoid<N>]) -> Result<([[f64;N];N],[f64;N]),String> 
{
    let mut M = Model::new(Some("lowner-john-inner"));

    let t = M.variable(Some("t"), unbounded());
    
    //let P = M.variable(Some("P"),in_psd_cone(n));
    let P = det_rootn(Some("P"), & mut M, t.clone(), N);
    let q = M.variable(Some("q"), unbounded().with_shape(&[N]));
    M.objective(None, Sense::Maximize, &t);

    for e in data.iter() {
       ellipsoid_contained(&mut M,&P,&q,e);
    }

    M.solve();

    M.write_problem("lj-inner-ellipsoid.ptf");

    let Psol = M.primal_solution(SolutionType::Default, &P)?;
    let qsol = M.primal_solution(SolutionType::Default, &q)?;

    let mut rP = [[0.0;N];N]; rP.iter_mut().flat_map(|row| row.iter_mut()).zip(Psol.iter()).for_each(|(t,&s)| *t = s);
    let mut rq = [0.0;N]; rq.clone_from_slice(qsol.as_slice());

    Ok( (rP,rq) )
}



//   /// Purpose: Models the hypograph of the n-th power of the
//   /// determinant of a positive definite matrix. See [1,2] for more details.
//   ///
//   ///   The convex set (a hypograph)
//   ///   ```math
//   ///   C = { (X, t) ∊ S^n_+ x R |  t ≤ det(X)^{1/n} },
//   ///   ```
//   ///   can be modeled as the intersection of a semidefinite cone
//   ///
//   ///   ```math
//   ///   | X   Z       |
//   ///   |             | ≽ 0
//   ///   | Z^T Diag(Z) |  
//   ///   ```
//   ///   and `Z` lower triangular.
//   ///
//   ///   and a geometric mean bound
//   ///
//   ///   ```math
//   ///   t <= (Z11*Z22*...*Znn)^{1/n} 
//   ///   ```
//   #[allow(non_snake_case)]
//   fn det_rootn(name : Option<&str>, M : &mut Model, t : Variable<0>, n : usize) -> Variable<2> {
//       // Setup variables
//       let Y = M.variable(name, in_psd_cone(2*n));
//   
//       // Setup Y = [X, Z; Z^T , diag(Z)]
//       let X  = (&Y).slice(&[0..n, 0..n]);
//       let Z  = (&Y).slice(&[0..n,   n..2*n]);
//       let DZ = (&Y).slice(&[n..2*n, n..2*n]);
//   
//       // Z is lower-triangular
//       _ = M.constraint(Some("triu(Z)=0"), &Z.clone().triuvec(false), equal_to(vec![0.0; n*(n-1)/2].as_slice()));
//       // DZ = Diag(Z)
//       _ = M.constraint(Some("DZ=Diag(Z)"), &DZ.clone().sub(Z.mul_elem(speye(n))), equal_to(dense(n,n,vec![0.0; n*n])));
//       // (Z11*Z22*...*Znn) >= t^n
//       _ = M.constraint(name,&vstack![DZ.clone().diag(),t.reshape(&[1])], in_geometric_mean_cone(n+1));
//   
//       X
//   }
//   
#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use super::*;
    use crate::utils2d::*;
    use ellipsoids::*;

    fn ellipse_from_param(dx : f64, dy : f64, sx : f64, sy : f64, theta : f64) -> Ellipsoid<2> {
        let A = [ [ sx*theta.cos(), sy*theta.sin()], [-sx*theta.sin(), sy*theta.cos() ] ];
        let b = [ dx, dy ];

        Ellipsoid::new(&A,&b)
    }

    //xA'Ax + 2Abx + b'b-1 = 0

    #[test]
    fn test() {
        let ellipses : &[Ellipsoid<2>] = &[
            Ellipsoid::new(&[[2.3246108597249653, -0.5023002219069703],  [-0.5023002219069703, 1.143239126869145]], &[ 1.6262243355626635,   2.3572628105100137]),
            Ellipsoid::new(&[[0.7493116238095875, -0.4519632176898713],  [-0.4519632176898713, 1.8627720007697564]],&[-0.7741774083306816,  -0.19900209213742667]),
            Ellipsoid::new(&[[0.8582679328352112,  0.21661818880463501], [ 0.21661818880463501,0.8721042500171801]],&[-0.30881539733571506,  0.6395981801584628]),
            Ellipsoid::new(&[[2.9440718650596165, -1.0778871720849044],  [-1.0778871720849044, 3.797461570034363]], &[ 2.2609091903635528,   5.387366401851428]),
            Ellipsoid::new(&[[0.6104500401008942, -0.06447520306755025], [-0.06447520306755025,0.6137552998087111]],&[-0.36695528785675785, -0.7214779986292089]),
            Ellipsoid::new(&[[1.1044036516422946, -0.18480500741119338], [-0.18480500741119338,4.271283557279645]], &[-5.123066175367,       2.1838724317503617]),
       ];

        let (P,q) = minimal_bounding_ellipsoid(ellipses).uwrap();

        println!("P = {:?}",P);
        println!("q = {:?}",q);

        // Now, bonud ellipsoid is defined by A,b, with
        // A² = P => P = sqrt(A)
        // Ab = q => A\q
    
        
    }
}
