extern crate mosekmodel;
extern crate itertools;

use mosekmodel::*;
use mosekmodel::matrix;
use itertools::izip;

// Structure defining an ellipsoid as
// 1.
//     ```math 
//     { x | ‖ Px+q ‖₂ ≤ 1 }
//     ```
// 2. It can be alternatively represented as 
//     ```math 
//     x'Ax + 2b'x + c ≤ 0
//     ```
//     with
//     ```math 
//     A = P²
//     b = Pqx
//     c = q'q-1
//   ```
// 3. or, as a third alternative as 
//
//     ```math
//     { Zu+w | || u || ≤ 1 }
//     ```
//
//     where 
//
//     ```math
//     Z = P^{-1}
//     w = -P^{-1}q
//     ```
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Ellipsoid<const N : usize> {
    P : [ [ f64; N ] ; N ],
    q : [ f64; N ]
}
#[allow(non_snake_case)]
impl<const N : usize> Ellipsoid<N> {
    /// Specify ellipsoid by `P` and `q` as
    /// ```math 
    /// { x | ‖ Px+q ‖² ≤ 1 }
    /// ```
    pub fn new(P : &[[f64;N];N], q : &[f64;N]) -> Ellipsoid<N> { Ellipsoid{P : *P, q : *q } }
    pub fn from_arrays(P : &[f64], q : &[f64]) -> Ellipsoid<N> {
        let mut e = Ellipsoid{ P : [[0.0;N];N], q : [0.0;N] };
        e.q.copy_from_slice(q);
        e.P.iter_mut().flat_map(|row| row.iter_mut()).zip(P.iter()).for_each(|(t,&s)| *t = s);
        e
    }
    pub fn get_Pq(&self) -> ([ [ f64; N ]; N ],[f64;N]) { (self.P,self.q) }

    /// For alternative parameterization
    /// ```math
    /// { x'Ax + 2b'x + c ≤ 0 }
    /// ```
    /// get the values of `A`, `b` and `c`, which will given by expanding 
    /// ```math 
    /// ‖ Px+q ‖² ≤ 1
    /// ```
    /// into 
    /// ```math 
    /// x'P²x + 2Pqx + q'q-1 ≤ 0
    /// ```
    /// Implying that
    /// ```math 
    /// A = P²
    /// b = Pq
    /// c = q'q-1
    /// ```
    pub fn get_Abc(&self) -> ([[f64;N];N],[f64;N],f64) {
        (self.get_A(),
         self.get_b(),
         self.get_c())
    }

    // A = P²
    fn get_A(&self) -> [ [ f64; N ]; N ] { 
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

    // b = Pq
    fn get_b(&self) -> [f64; N] { 
        let mut res = [0.0;N];
        self.P.iter()
            .zip(std::iter::repeat(self.q))
            .zip(res.iter_mut())
            .for_each(|((Pi,q),r)| *r = Pi.iter().zip(q.iter()).map(|(&Pij,&qi)| Pij*qi).sum());
        res
    }

    // c = q'q-1
    fn get_c(&self) -> f64 { self.q.iter().map(|v| v*v).sum::<f64>() - 1.0}
}



/// For a fixed ellipsoid E add a constraint to the effect that 
/// ```math
/// E ⊂ { x: || Px+q || ≤ 1 }
/// ```
///
/// The two variables `P_squared` and `Pq` are the parameters of the computed enclosing ellipsoid.
/// At optimum, the values of the variables will be
/// ```text
/// P_squared = P²
/// Pq        = P * q
/// ```
///
/// # Arguments
/// - `M` Model
/// - `P_squared` must be a symmetric positive semidefinite `n x n` variable
/// - 'Pq' is a variable vector of length `n`.
/// - `E` The contained ellipsoid.
#[allow(non_snake_case)]
pub fn ellipsoid_contains<const N : usize>
(   M : & mut Model,
    P_squared : &Variable<2>, 
    Pq : &Variable<1>, 
    e : &Ellipsoid<N>) -> Variable<0> {

    let (A,b,c) = e.get_Abc();

    let Pshp = P_squared.shape();
    let qshp = Pq.shape();
    if Pshp[0] != Pshp[1] || qshp[0] != Pshp[0] {
        panic!("Invalid or mismatching P and/or q");
    }
    let n = qshp[0];
   
    let S = M.variable(Some("S"), in_psd_cone(2*n+1));
    let S11 = (&S).index([0..n,0..n]);
    let S21 = (&S).index([n..n+1,0..n]).reshape(&[n]);
    let S22 = (&S).index([n..n+1,n..n+1]).reshape(&[]);
    let S31 = (&S).index([n+1..2*n+1,0..n]);
    let S32 = (&S).index([n+1..2*n+1,n..n+1]).reshape(&[n]);
    let S33 = (&S).index([n+1..2*n+1,n+1..2*n+1]);
    let tau = M.variable(Some("tau"), nonnegative());

    let A = matrix::dense([N,N],A.iter().flat_map(|arow| arow.iter()).cloned().collect::<Vec<f64>>());

    _ = M.constraint(None, &P_squared.clone().sub(tau.clone().mul(&A))    .add(S11), zero().with_shape(&[n,n]));
    _ = M.constraint(None, &Pq.clone().sub(tau.clone().mul(b.as_slice())) .add(S21), zero().with_shape(&[n]));
    _ = M.constraint(None, &tau.clone().mul(c).add(1.0).neg()             .add(S22), zero());
    _ = M.constraint(None, &Pq.clone()                                    .add(S32), zero().with_shape(&[n]));
    _ = M.constraint(None, &P_squared.clone().neg()                       .add(S33), zero().with_shape(&[n,n]));
    _ = M.constraint(None,                                                    &S31,  zero().with_shape(&[N,N]));

    tau
}

/// Adds a constraint
/// ```math 
/// p_i ∊ { x: || Px+q || ≤ 1 }, i ∊ 1..m
/// ```
///
/// # Arguments
/// - `M` the Model
/// - `P` 
/// - `q`
/// - `points`
#[allow(non_snake_case)]
pub fn ellipsoid_contains_points<const N : usize>
(   M : & mut Model,
    P : &Variable<2>,
    q : &Variable<1>,
    points : &[ [f64;N] ]) {

    let mx = matrix::dense([points.len(), N], points.iter().flat_map(|p| p.iter()).cloned().collect::<Vec<f64>>());
    let m = points.len();
    // 1 >=||P p_i + q||^2
    _ = M.constraint(None, &hstack![ expr::ones(&[m,1]) , P.clone().rev_mul(mx).add( q.clone().reshape(&[1,N]).repeat(0, m))], in_quadratic_cones(&[m,N+1], 1));
}


/// For a fixed ellipsoid E add a constraint to the effect
/// { Zx+w : || x || ≤ 1 } ⊂ E
///
/// #Arguments
/// - `M` Model
/// - `Z`, `w` are the variable parameters of the contained ellipsoid
/// - `E` is the containing ellipsoid
#[allow(non_snake_case)]
pub fn ellipsoid_contained<const N : usize> 
(   M : &mut Model,
    Z : &Variable<2>,
    w : &Variable<1>,
    e : &Ellipsoid<N>) {
  
    let S = M.variable(None, in_psd_cone(2*N+1));
    let S11 = (&S).index([0..N,0..N]);
    let S21 = (&S).index([N..N+1,0..N]);
    let S22 = (&S).index([N..N+1,N..N+1]).reshape(&[]);
    let S31 = (&S).index([N+1..2*N+1,0..N]);
    let S32 = (&S).index([N+1..2*N+1,N..N+1]).reshape(&[N]);
    let S33 = (&S).index([N+1..2*N+1,N+1..2*N+1]);
    let lambda = M.variable(None, nonnegative());

    let (B,c) = e.get_Pq();
    let B = matrix::dense([N,N],B.iter().flat_map(|arow| arow.iter()).cloned().collect::<Vec<f64>>());
    let c = matrix::dense([1,N],&c[..]);
    
    _ = M.constraint(None, &expr::eye(N).sub(S11),zero().with_shape(&[N,N]));
    _ = M.constraint(None, &w.clone().reshape(&[1,N]).mul(B.clone()).add(c).sub(S21),zero().with_shape(&[1,N]));
    _ = M.constraint(None, &Z.clone().rev_mul(B).sub(S31), zero().with_shape(&[N,N]));
    _ = M.constraint(None, &lambda.clone().neg().add(1.0).sub(S22), zero());
    _ = M.constraint(None, &S32, zero().with_shape(&[N]));
    _ = M.constraint(None, &lambda.clone().mul(&matrix::speye(N)).sub(S33),zero().with_shape(&[N,N]));
}



#[allow(non_snake_case)]
pub fn ellipsoid_subject_to<const N : usize> 
(   M : &mut Model,
    Z : &Variable<2>,    
    w : &Variable<1>,
    A : &[[f64;N]],
    b : &[f64])
{
    let m = A.len();
    assert_eq!(b.len(),m);
    let A = matrix::dense([m,N],A.iter().flat_map(|a| a.iter()).cloned().collect::<Vec<f64>>());
    let b = matrix::dense([m, 1], b.to_vec());
    _ = M.constraint(Some("E_Axb"), 
                     &hstack![ w.clone().reshape(&[2,1]).rev_mul(A.clone()).sub(b).neg(), Z.clone().rev_mul(A) ], 
                     in_quadratic_cones(&[m,N+1],1));
}



/// Create a semidefinite variable `X` such that
/// ```math
/// t ≤ det(X)^{1/n}
/// ```
/// This is modeled as
/// ```math
/// | X   Z       |
/// |             | ≽ 0
/// | Z^T Diag(Z) |  
/// t <= (Z11*Z22*...*Znn)^{1/n} 
/// ```
/// and `Z` lower triangular.
/// # Arguments
/// - `name` Optional name to use to created model items.
/// - `M` Model.
/// - `t` Scalar variable.
/// - `n` Dimension of the returned semidefinite variable.
/// # Returns
/// A symmetric positive `X` semidefinite variable such that 
/// ```math
/// t ≤ det(X)^{1/n}
/// ```
#[allow(non_snake_case)]
pub fn det_rootn(name : Option<&str>, M : &mut Model, t : Variable<0>, n : usize) -> Variable<2> {
    // Setup variables
    let Y = M.variable(name, in_psd_cone(2*n));

    // Setup Y = [X, Z; Z^T , diag(Z)]
    let X  = (&Y).index([0..n, 0..n]);
    let Z  = (&Y).index([0..n,   n..2*n]);
    let DZ = (&Y).index([n..2*n, n..2*n]);

    // Z is lower-triangular
    _ = M.constraint(None, &Z.clone().triuvec(false), equal_to(vec![0.0; n*(n-1)/2].as_slice()));
    // DZ = Diag(Z)
    _ = M.constraint(None, &DZ.clone().sub(Z.mul_elem(matrix::speye(n))), equal_to(matrix::dense([n,n],vec![0.0; n*n])));
    // (Z11*Z22*...*Znn) >= t^n
    _ = M.constraint(name,&vstack![DZ.clone().diag(),t.reshape(&[1])], in_geometric_mean_cone(n+1));

    X
}


#[cfg(test)]
mod test {
    use mosekmodel::{unbounded, Model, SolutionType};
    use itertools::izip;

    #[allow(non_snake_case)]
    #[test]
    fn test_ellipsoid_subject_to_1() {
        let mut M = Model::new(None);
        let t = M.variable(None, unbounded());
        let P = super::det_rootn(None, & mut M, t.clone(), 2);
        let q = M.variable(None, unbounded().with_shape(&[2]));

        M.objective(None, mosekmodel::Sense::Maximize, &t);

        let A = [ [-1.0, -1.0], [1.0, 0.0], [-1.0,3.0] ];
        let b = [ -3.0, 6.0, -9.0 ];
            
        super::ellipsoid_subject_to(& mut M, &P, &q, A.as_slice(), b.as_slice());

        M.solve();

        let _Psol = M.primal_solution(SolutionType::Default, &P).unwrap();
        let _qsol = M.primal_solution(SolutionType::Default, &q).unwrap();
    }


    #[allow(non_snake_case)]
    #[test]
    fn test_ellipsoid_subject_to_2() {
        let points = [ [0.0,3.0],[6.0,5.0],[6.0,-3.0],
                       [3.0,3.0],[3.0,8.0],[8.0,8.0],[8.0,3.0]];
        let polygons = [0usize,3];


        let mut M = Model::new(None);
        let t = M.variable(None, unbounded());
        let P = super::det_rootn(None, & mut M, t.clone(), 2);
        let q = M.variable(None, unbounded().with_shape(&[2]));

        M.objective(None, mosekmodel::Sense::Maximize, &t);
          
        let mut A = vec![ [0.0;2]; points.len()];
        let mut b = vec![ 0.0; points.len() ];
            
        for ((p0,p1),a,b) in izip!(polygons.iter().zip(polygons[1..].iter())
                                   .flat_map(|(&pb,&pe)| points[pb..pe].iter().zip(points[1..].iter().chain(std::iter::once(&points[pb])))),
                                   A.iter_mut(),
                                   b.iter_mut()) {
            a[0] = p0[1]-p1[1];
            a[1] = p1[0]-p0[0];
            *b = a[0] * p0[0] + a[1] * p0[1]; 
        }

        super::ellipsoid_subject_to(& mut M, &P, &q, A.as_slice(), b.as_slice());

        M.solve();

        M.write_problem("lw-inner-2.ptf");

        let _Psol = M.primal_solution(SolutionType::Default, &P).unwrap();
        let _qsol = M.primal_solution(SolutionType::Default, &q).unwrap();
    }
}


