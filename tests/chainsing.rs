extern crate MosekModel;
extern crate test;
use MosekModel::*;
use MosekModel::expr::*;
use test::Bencher;

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
    let mut model = Model::new("chainsing-1");
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
                         &((0.5)
                           .vstack(s.index(j))
                           .vstack(x.index(i).add(x.index(i+1).mul(10.0)))),
		       in_rotated_quadratic_cone(3));

        // t[j] >= 5^0.5*(x[i+2] - x[i+3])^2
        model.constraint(None,
                         &((0.5)
                           .vstack(t.index(j))
                           .vstack((5.0).sqrt().mul(x.index(i+1).sub(x.index(i+3))))),
		       in_rotated_quadratic_cone(3));

        // r[j] >= (x[i+1] - 2*x[i+2])^2
        model.constraint(None,
                         &((0.5)
                           .vstack(r.index(j))
                           .vstack(x.index(i+1).sub((2.0).mul(x.index(i+2))))),
		         in_rotated_quadratic_cone(3));

        // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
        model.constraint(None,
                         &((0.5*10.0.powf(-0.25))
                           .vstack(u.index(j))
                           .vstack(x.index(i).sub((10.0).mul(x.index(i+3))))),
		         in_rotated_quadratic_cone(3));

        // p[j] >= r[j]^2
        model.constraint(None,
                         &((0.5)
                           .vstack(p.index(j))
                           .vstack(r.index(j))),
		         in_rotated_quadratic_cone(3));

      // q[j] >= u[j]^2
        model.constraint(None,
                         &((0.5)
                           .vstack(q.index(j))
                           .vstack(u.index(j)))
		         in_rotated_quadratic_cone(3));

    }

    // 0.1 <= x[i] <= 1.1, i=0,2,...,n-2
    for i in (0..n).step_by(2) {
	model.constraint(None, &(x.index(i)), greater_than(0.1));
	model.constraint(None, &(x.index(i)), less_than(1.1));
    }

    model.objective(Sense::Minimize,
                    &Variable::vstack(&[&s,&t,&p,&q]).sum());

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
  //     se[2*j+2] = Expr.add(x.index(i), Expr.mul(10.0, x.index(i+1)));
  //     se[2*j+3] = Expr.mul(Math.sqrt(5),Expr.sub(x.index(i+2), x.index(i+3)));
    
  //     // r[j] >= (x[i+1] - 2*x[i+2])^2
  //     model.constraint(Expr.vstack(c,
  //       		       r.index(j).asExpr(), 
  //       		       Expr.sub(x.index(i+1), Expr.mul(2.0,x.index(i+2)))),
  //       	   Domain.inRotatedQCone());

  //     // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
  //     model.constraint(Expr.vstack(Expr.constTerm(1,0.5*Math.pow(10,-0.25)),
  //       		       u.index(j).asExpr(), 
  //       		       Expr.sub(x.index(i), Expr.mul(10.0,x.index(i+3)))),
  //       	   Domain.inRotatedQCone());
      
  //     // p[j] >= r[j]^2
  //     model.constraint(Expr.vstack(c,
  //       		       p.index(j).asExpr(),
  //       		       r.index(j).asExpr()), 
  //       	   Domain.inRotatedQCone());

  //     // q[j] >= u[j]^2
  //     model.constraint(Expr.vstack(c,
  //       		       q.index(j).asExpr(), 
  //       		       u.index(j).asExpr()),
  //       	   Domain.inRotatedQCone());

  //   }
    
  //   // 0.1 <= x[i] <= 1.1, i=0,2,...,n-1
  //   for (int i = 0; i < n; i+=2) {
  //       model.constraint(x.index(i), Domain.inRange(0.1,1.1));
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
pub fn chainsing3(n : usize) {
    let mut model = Model::new(Some("chainsing-3"));
    let m = (n-2) >> 1;

    let x = model.variable(Some("x"), n);
    let r = model.variable(Some("r"), m);
    let s = model.variable(Some("s"), 1);
    let u = model.variable(Some("u"), m);

    for j in 0..m {
      let i = j << 1;
      // r[j] >= (x[i+1] - 2*x[i+2])^2
        model.constraint(None,
                         &((0.5)
                           .vstack(r.index(j))
                           .vstack(x.index(i+1).sub((2.0).mul(x.index(i+2))))),
		   in_rotated_quadratic_cone(3));

      // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
        model.constraint(None,
                         &((0.5).powf(-0.25)
                           .vstack(u.index(j))
                           .vstack(x.index(i).sub((10.0).mul(x.index(i+3))))),
		         in_rotated_quadratic_cone(3));
    }

    // 0.1 <= x[i] <= 1.1, i=0,2,...,n-1
    for i in (0..n).step_by(2) {
	model.constraint(None, &x.index(i), greater_than(0.1));
	model.constraint(None, &x.index(i), less_than(1.1));
    }

    model.constraint(None,
                     &((0..m).fold(s.vstack(0.5),|e,j| {
                         let i = j << 1;
                         e.vstack(x.index(i).add((10.0).mul(x.index(i+1))))
                             .vstack((0.5).sqrt().mul(x.index(i+2).sub(x.index(i+3))))
                             .vstack(r.index(j))
                             .vstack(u.index(j)) })),
                     in_rotated_quadratic_cone(2+m*4));

    model.objective(Sense::Minimize, &s);
    model
}


pub fn chainsing4(n : usize) {
    let mut model = Model::new("chainsing-4");
    let m = (n-2) / 2;
    let x = model.variable(None,n);
    let p = model.variable(None,m);
    let q = model.variable(None,m);
    let r = model.variable(None,m);
    let s = model.variable(None,m);
    let t = model.variable(None,m);
    let u = model.variable(None,m);

    let xr = x.with_shape(&[n/2,2]);
    let x_i      = xr.slice(&[0..n/2-1,0..1]);
    let x_iplus1 = xr.slice(&[0..n/2-1,1..2]);
    let x_iplus2 = xr.slice(&[1..n/2,0..1]);
    let x_iplus3 = xr.slice(&[1..n/2,1..2]);

    // s[j] >= (x[i] + 10*x[i+1])^2
    model.constraint(None,
                     &((0.5).hstack(s).hstack(x_i.add((10.0).mul(x_iplus1)))),
                     in_rotated_quadratic_cone(1+n/2));
    // t[j] >= 5*(x[i+2] - x[i+3])^2
    model.constraint(None,
                     &((0.5).hstack(t).hstack((0.5).sqrt().mul(x_iplus2.sub(x_iplus3)))),
                     in_rotated_quadratic_cone(1+n/2));
    // r[j] >= (x[i+1] - 2*x[i+2])^2
    model.constraint(None,
                     &((0.5).hstack(r).hstack(x_iplus1.sub((2.0).mul(x_iplus2)))),
                     in_rotated_quadratic_cone(1+n/2));
    // u[j] >= sqrt(10)*(x[i] - 10*x[i+3])^2
    model.constraint(None,
                     &((0.5/(10.0).sqrt()).hstack(u).hstack(x_i.sub((10.0).mul(x_iplus3)))),
                     in_rotated_quadratic_cone(1+n/2));
    // p[j] >= r[j]^2
    model.constraint(None,
                     &(c.hstack(p).hstack(r)),
                     in_rotated_quadratic_cone(3));
    // q[j] >= u[j]^2
    model.constraint(None,
                     &(c.hstack(q).hstack(u)),
                     in_rotated_quadratic_cone(3));
    // 0.1 <= x[j] <= 1.1
    model.constraint(None,&x,greater_than(0.1));
    model.constraint(None,&x,less_than(1.1));

    model.objective(Sense::Minimize, &Variable::vstack(&[&s, &t, &p, &q]).sum());
    model
}


#[bench]
pub fn chainsing1_large(b : & mut Bencher) {
    b.iter(|| let _ = chainsing1(10000))
}

// #[bench]
// pub fn chainsing2_large(b : & mut Bencher) {
//     b.iter(|| let _ = chainsing1(10000))
// }
#[bench]
pub fn chainsing3_large(b : & mut Bencher) {
    b.iter(|| let _ = chainsing3(10000))
}
#[bench]
pub fn chainsing4_large(b : & mut Bencher) {
    b.iter(|| let _ = chainsing4(10000))
}
