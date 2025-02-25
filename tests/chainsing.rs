extern crate mosekcomodel;
use mosekcomodel::*;
use mosekcomodel::expr::*;

// Different formulations of the chained singular function
// (CHAINSING) problem

// min.  sum_{i=0,2,...,n-4}  (x[i]+10x[i+1])^2  +  5(x[i+2]-x[i+3])^2 + 
//              (x[i+1]-2x[i+2])^4  +  10(x[i]-10x[i+3])^2

// s.t.  0.1 <= x[i] <= 1.1,  i=0, 2, ... , n-4

// where n is multiple of 4.

// The CHAINSING problem is described in [1] and the different formulations in [2].

// [1]   Testing a Class of Methods for Solving Minimization Problems with 
// Simple Bounds on the Variables. A. Conn, N. Gould, P. Toint, 
// Mathematics of Computation, Vol. 50, No. 182 (1988), pp. 399-430.

// [2]   Sparse second order cone programming formulations for convex
// optimization problems. K. Kobayashi, S.-Y. Kim, M. Kojima,
// Journal of the Operations Research Society of Japan, Vol. 51, No. 3 (2008),
// pp. 241-264.




// min.  sum_{j=0,...,(n-2)/2} s[j] + t[j] + p[j] + q[j]

// s.t.  (1/2, s[j], x[i]+10x[i+1]) \in Qr, 
//       (1/2, t[j], 5^{1/2}*(x[i+2]-x[i+3])) \in Qr
//       (1/2, r[j], (x[i+1]-2x[i+2])) \in Qr
//       (1/2, u[j], 10^{1/4}*(x[i]-10x[i+3])) \in Qr
//       (1/2, p[j], r[j]) \in Qr
//       (1/2, q[j], u[j]) \in Qr,                       j=0,...,(n-2)/2, i = 2j

//       0.1 <= x[i] <= 1.1,                             i=0,2,...,n-2
pub fn chainsing1(n : usize) -> Model {
    let mut model = Model::new(Some("chainsing-1"));
    let m = (n-1) >> 1;

    let x = model.variable(None,n);
    let p = model.variable(None,m);
    let q = model.variable(None,m);
    let r = model.variable(None,m);
    let s = model.variable(None,m);
    let t = model.variable(None,m);
    let u = model.variable(None,m);

    for j in 0..m {
        let i = j << 1;

        // s[j] >= (x[i] + 10*x[i+1])^2
        model.constraint(None,
                         &vstack![Expr::from(0.5).flatten(),
                                  (&s).index(j).flatten(),
                                  (&x).index(i).add((&x).index(i+1).mul(10.0)).flatten()],
		         in_rotated_quadratic_cone(3));

        // t[j] >= 5^0.5*(x[i+2] - x[i+3])^2
        model.constraint(None,
                         &vstack![Expr::from(0.5).flatten(),
                                  (&t).index(j).flatten(),
                                  (&x).index(i+1).sub((&x).index(i+3)).mul(5.0f64.sqrt()).flatten()],
		         in_rotated_quadratic_cone(3));

        // r[j] >= (x[i+1] - 2*x[i+2])^2
        model.constraint(None,
                         &vstack![Expr::from(0.5).flatten(),
                                  (&r).index(j).flatten(),
                                  (&x).index(i+1).sub((&x).index(i+2).mul(2.0)).flatten()],
		         in_rotated_quadratic_cone(3));

        // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
        model.constraint(None,
                         &vstack![Expr::from(0.5*10.0f64.powf(-0.25)).flatten(),
                                  (&u).index(j).flatten(),
                                  (&x).index(i).sub((&x).index(i+3).mul(10.0)).flatten()],
		         in_rotated_quadratic_cone(3));

        // p[j] >= r[j]^2
        model.constraint(None,
                         &vstack![Expr::from(0.5).flatten(),
                                  (&p).index(j).flatten(),
                                  (&r).index(j).flatten()],
		         in_rotated_quadratic_cone(3));

      // q[j] >= u[j]^2
        model.constraint(None,
                         &vstack![Expr::from(0.5).flatten(),
                                  (&q).index(j).flatten(),
                                  (&u).index(j).flatten()],
		         in_rotated_quadratic_cone(3));

    }

    // 0.1 <= x[i] <= 1.1, i=0,2,...,n-2
    for i in (0..n).step_by(2) {
	model.constraint(None, &((&x).index(i)), greater_than(0.1));
	model.constraint(None, &((&x).index(i)), less_than(1.1));
    }

    model.objective(None,
                    Sense::Minimize,
                    &(Variable::vstack(&[&s,&t,&p,&q]).sum()));

    model
}

  // public static void chainsing2
  //   (Model  M,
  //    int    n)
  // {
  //   /*
  //     min.  s + sum_j s[j] + t[j] + p[j] + q[j],  j = 0,...,(n-2)/2

  //     s.t.  (1/2, s, [x[i]+10x[i+1], 5^{1/2}*(x[i+2]-x[i+3])]_{i=0,2,...,n-4}  \in Qr, 

  //           (1/2, r[j], (x[i+1]-2x[i+2])) \in Qr
  //           (1/2, u[j], 10^{1/4}*(x[i]-10x[i+3])) \in Qr
  //           (1/2, p[j], r[j]) \in Qr
  //           (1/2, q[j], u[j]) \in Qr,                       i=0,2,...,n-4, j = i/2
        
  //           0.1 <= x[i] <= 1.1,                             i=0,2,...,n-1
  //   */
  //   double T1 = 0.001 * System.currentTimeMillis();
  
  //   int m = (n-2) >> 1;

  //   Variable x = model.variable("x", n, Domain.unbounded());
  //   Variable p = model.variable("p", m, Domain.unbounded());
  //   Variable q = model.variable("q", m, Domain.unbounded());
  //   Variable r = model.variable("r", m, Domain.unbounded());
  //   Variable s = model.variable("s", 1, Domain.unbounded());
  //   Variable u = model.variable("u", m, Domain.unbounded());

  //   Expression c = Expr.constTerm(1,0.5);

  //   Expression se[] = new Expression[2*m+2];
  //   se[0] = s.asExpr();
  //   se[1] = c;
    
  //   for (int j = 0; j < m; ++j) {

  //     int i = j << 1;

  //     // s >= sum_i (x[i] + 10*x[i+1])^2 + 5*(x[i+2]-x[i+3])^2
  //     se[2*j+2] = Expr.add((&x).index(i), Expr.mul(10.0, (&x).index(i+1)));
  //     se[2*j+3] = Expr.mul(Math.sqrt(5),Expr.sub((&x).index(i+2), (&x).index(i+3)));
    
  //     // r[j] >= (x[i+1] - 2*x[i+2])^2
  //     model.constraint(Expr.vstack(c,
  //       		       (&r).index(j).asExpr(), 
  //       		       Expr.sub((&x).index(i+1), Expr.mul(2.0,(&x).index(i+2)))),
  //       	   Domain.inRotatedQCone());

  //     // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
  //     model.constraint(Expr.vstack(Expr.constTerm(1,0.5*Math.pow(10,-0.25)),
  //       		       (&u).index(j).asExpr(), 
  //       		       Expr.sub((&x).index(i), Expr.mul(10.0,(&x).index(i+3)))),
  //       	   Domain.inRotatedQCone());
      
  //     // p[j] >= r[j]^2
  //     model.constraint(Expr.vstack(c,
  //       		       (&p).index(j).asExpr(),
  //       		       (&r).index(j).asExpr()), 
  //       	   Domain.inRotatedQCone());

  //     // q[j] >= u[j]^2
  //     model.constraint(Expr.vstack(c,
  //       		       (&q).index(j).asExpr(), 
  //       		       (&u).index(j).asExpr()),
  //       	   Domain.inRotatedQCone());

  //   }
    
  //   // 0.1 <= x[i] <= 1.1, i=0,2,...,n-1
  //   for (int i = 0; i < n; i+=2) {
  //       model.constraint((&x).index(i), Domain.inRange(0.1,1.1));
  //   }

  //   model.constraint(Expr.vstack(se), Domain.inRotatedQCone());

  //   model.objective(ObjectiveSense.Minimize, 
  //       	Expr.sum(Var.vstack(new Variable[] {s, p, q})));
  //   double T2 = 0.001 * System.currentTimeMillis();
  // }



/// min.  s
///
/// s.t.  (1/2, s, [x[i]+10x[i+1], 5^{1/2}*(x[i+2]-x[i+3]), r[i], u[i]]_{i=0,2,...,n-4}  \in Qr, 
///
///       (1/2, r[j], (x[i+1]-2x[i+2])) \in Qr
///       (1/2, u[j], 10^{1/4}*(x[i]-10x[i+3])) \in Qr,   i=0,2,...,n-4, j = i/2
///
///       0.1 <= x[i] <= 1.1,                             i=0,2,...,n-1
pub fn chainsing3(n : usize) -> Model {
    let mut model = Model::new(Some("chainsing-3"));
    let m = (n-2) >> 1;

    let x = model.variable(Some("x"), n);
    let r = model.variable(Some("r"), m);
    let s = model.variable(Some("s"), &[]);
    let u = model.variable(Some("u"), m);

    for j in 0..m {
      let i = j << 1;
      // r[j] >= (x[i+1] - 2*x[i+2])^2
        model.constraint(None,
                         &vstack![Expr::from(0.5).flatten(),
                                  (&r).index(j).flatten(),
                                  (&x).index(i+1).sub((&x).index(i+2).mul(2.0)).flatten()],
		   in_rotated_quadratic_cone(3));

      // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
        model.constraint(None,
                         &vstack![Expr::from(0.5f64.powf(-0.25)).flatten(),
                                  (&u).index(j).flatten(),
                                  (&x).index(i).sub((&x).index(i+3).mul(10.0)).flatten()],
		         in_rotated_quadratic_cone(3));
    }

    // 0.1 <= x[i] <= 1.1, i=0,2,...,n-1
    for i in (0..n).step_by(2) {
        model.constraint(None, &(&x).index(i), greater_than(0.1));
        model.constraint(None, &(&x).index(i), less_than(1.1));
    }

    model.constraint(None,
                     &vstack![s.clone().flatten(),
                              Expr::from(0.5).flatten(),
                              vstack((0..m).map(|j| {
                                  let i = j << 1;
                                  vstack![(&x).index(i).add((&x).index(i+1).mul(10.0f64)).flatten(),
                                          (&x).index(i+2).mul(0.5f64.sqrt()).sub((&x).index(i+3)).flatten(),
                                          (&r).index(j).flatten(),
                                          (&u).index(j).flatten()].dynamic() }).collect())],
                     in_rotated_quadratic_cone(2+m*4));

    model.objective(None,Sense::Minimize,&s);
    model
}

pub fn chainsing4(n : usize) -> Model {
    let mut model = Model::new(Some("chainsing-4"));
    let m = (n-2) / 2;
    let x = model.variable(None,&[n,1]);
    let p = model.variable(None,&[m,1]);
    let q = model.variable(None,&[m,1]);
    let r = model.variable(None,&[m,1]);
    let s = model.variable(None,&[m,1]);
    let t = model.variable(None,&[m,1]);
    let u = model.variable(None,&[m,1]);

    let xr       = x.clone().with_shape(&[n/2,2]);
    let x_i      = (&xr).index([0..n/2-1,0..1]);
    let x_iplus1 = (&xr).index([0..n/2-1,1..2]);
    let x_iplus2 = (&xr).index([1..n/2,  0..1]);
    let x_iplus3 = (&xr).index([1..n/2,  1..2]);

    // s[j] >= (x[i] + 10*x[i+1])^2    
    model.constraint(None,
                     &hstack![ Expr::from(vec![0.5; m]).into_column(), 
                               s.clone(), 
                               x_i.clone().add(x_iplus1.clone().mul(10.0))],
                     in_rotated_quadratic_cones(&[m,3],1));
    // t[j] >= 5*(x[i+2] - x[i+3])^2
    model.constraint(None,
                     &hstack![Expr::from(vec![0.5; m]).into_column(), t.clone(), x_iplus2.clone().sub(x_iplus3.clone()).mul(0.5f64.sqrt())],
                     in_rotated_quadratic_cones(&[m,3],1));
    // r[j] >= (x[i+1] - 2*x[i+2])^2
    model.constraint(None,
                     &hstack![Expr::from(vec![0.5; m]).into_column(), Expr::from(r.clone()), x_iplus1.clone().sub(x_iplus2.clone().mul(2.0))],
                     in_rotated_quadratic_cones(&[m,3],1));
    // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
    model.constraint(None,
                     &hstack![Expr::from(vec![0.5/10.0f64.sqrt(); m]).into_column(),u.clone(),x_i.clone().sub(x_iplus3.clone().mul(10.0))],
                     in_rotated_quadratic_cones(&[m,3],1));
    // p[j] >= r[j]^2
    model.constraint(None,
                     &hstack![Expr::from(vec![0.5; m]).into_column(),p.clone(),r],
                     in_rotated_quadratic_cones(&[m,3],1));
    // q[j] >= u[j]^2
    model.constraint(None,
                     &hstack![Expr::from(vec![0.5;m]).into_column(), q.clone(), u],
                     in_rotated_quadratic_cones(&[m,3],1));
    // 0.1 <= x[j] <= 1.1
    model.constraint(None,&x.clone().flatten(),greater_than(vec![0.1;n]));
    model.constraint(None,&x.clone().flatten(),less_than(vec![1.1;n]));

    model.objective(None, Sense::Minimize, &Variable::vstack(&[&s, &t, &p, &q]).sum());
    model
}

//const N : usize = 10000;
const NSMALL : usize = 100;

#[test]
fn test_chainsing1_large() {
    let _ = chainsing1(NSMALL);
}

#[test]
fn test_chainsing3_large() {
    let _ = chainsing3(NSMALL);
}
#[test]
fn test_chainsing4_large() {
    let _ = chainsing4(NSMALL);
}

