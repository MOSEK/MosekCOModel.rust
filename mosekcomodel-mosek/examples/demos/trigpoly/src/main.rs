//!
//!  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//!  File:      trigpoly.rs
//!
//!  Purpose: 
//!  Example of an optimization problem over nonnegative 
//!  trigonometric polynomials.
//!
//!  We consider the nonnegative trigonometric polynomials
//!
//!  ```math 
//!  H(w) = x_0 + 2·sum_{k=1}^n [ Re(x_k)·cos(w·k) + Im(x_k)·sin(w·k) ].
//!  ```
//!
//!  The example shows how to construct a polynomial H(w) that satisfies,
//! 
//!  ```math
//!     1 - delta <=  H(w) <= 1 + delta,   forall w \in [0, wp]
//!  ```
//! 
//!  while minimizing `sup_{w ∊ [ws,pi]} H(w).`
//!
//!  In the signal processing literature, such a trigonometric polynomial
//!  is known as (the squared amplitude respons of) a Chebyshev lowpass filter. 
//!
//!  References:
//!  [1] "Squared Functional Systems and Optimization Problems",  
//!      Y. Nesterov, 2004.
//!
//!  [2] "Convex Optimization of Non-negative Polynomials:
//!      Structured Algorithms and Applications", Ph.D thesis, Y. Hachez, 2003.
//!
extern crate mosekcomodel;
use itertools::Itertools;
use mosekcomodel::*;
use mosekcomodel::experimental::*;
use mosekcomodel_mosek::Model;
use std::{cell::RefCell, cmp::Ordering, f64::consts::PI, rc::Rc};
use gtk::glib::ControlFlow;
use gtk::prelude::*;
use cairo::Context;
use gtk::{glib,Application, DrawingArea, ApplicationWindow};

/// Creates a complex semidefinite variable `(Xr + J*Xi) >= 0`, using the equivalent
/// representation
/// ```
/// [Xr, -Xi; Xi, Xr] >= 0   (implying that Xi is skew-symmetric).
/// ```
#[allow(non_snake_case)]
fn complex_sdpvar(m : & mut Model, n : usize) -> (Variable<2>,Variable<2>) {
    let X   = m.variable(None, in_psd_cone().with_dim(2*n));
    let Xr  = (&X).index([0..n, 0..n]);
    let Xi  = (&X).index([n..2*n, 0..n]);
    let X22 = (&X).index([n..2*n, n..2*n]);
    
    _ = m.constraint(None, Xr.sub(&X22), zeros(&[n,n]));
    _ = m.constraint(None, Xi.add(&Xi.transpose()), zeros(&[n,n]));
    
    (Xr, Xi)
}

/// Creates a Toeplitz matrix of dimension `n+1`, where 
/// ```math
/// T_lk = a if l-k=i, and 0 otherwise.
/// ```
fn toeplitz(n : usize, i : i64, a : f64 /*=1.0*/) -> NDArray<2> {
    if i >= 0 {
        let i = i as usize;
        matrix::sparse([n+1,n+1], 
                       (i..n+1).zip(0..n+1-i).map(|(i,j)| [i,j]).collect::<Vec<[usize;2]>>(),
                       vec![a; n+1-i])
    }
    else {
        let i = (-i) as usize;
        matrix::sparse([n+1,n+1],
                       (0..n+1+i).zip(i..n+1).map(|(i,j)|[i,j]).collect::<Vec<[usize;2]>>(), 
                       vec![a; n+1-i])
    }
}

fn toeplitz_ext(n : usize, indx : &[i64], aa : &[f64]) -> NDArray<2> {
    let mut sp : Vec<[usize;2]> = Vec::new();
    let mut cof : Vec<f64> = Vec::new();

    // n = 5 
    // indx = [ 4,6,5 ]
    // 
    // [ 4,5,  5 ]
    // [ 0,1,  0 ]

    for (&i,&a) in indx.iter().zip(aa.iter()) {
        let i = if i >= 0 { i as usize } else { (-i) as usize };
        if i < n+1 {
            for (k0,k1) in (i..n+1).zip(0..n+1-i) { sp.push([k0,k1]); }
            for _ in 0..n+1-i { cof.push(a); }
        }
    }

    matrix::sparse([n+1,n+1], sp, cof)
}


/// Models the equation
/// ```math
/// x[i] = <T(n+1,i),X>
/// ```
/// where `x = (xr,xi)` is a complex variable vector, and `X = (Xr,Xi)` is a 
/// complex PSD variable.
#[allow(non_snake_case)]
fn trigpoly_0_pi(m : & mut Model, xr : & Variable<1>, xi : & Variable<1>) {
    let n = xi.len()-1;
    assert_eq!(xi.len(),xr.len());

    let (Xr, Xi) = complex_sdpvar(m, n+1);

    _ = m.constraint(None, 
                     xr.sub((0..n+1).genexpr(|_,i| Some(Xr.dot(toeplitz(n,i as i64,1.0))))), 
                     equal_to(0.0).with_shape(&[n+1]));
    _ = m.constraint(None,
                     xi.sub((0..n+1).genexpr(|_,i| Some(Xi.dot(toeplitz(n,i as i64,1.0))))),
                     equal_to(0.0).with_shape(&[n+1]));
}

/// Models the equation
/// ```math
/// x[i] = <T(n+1,i),X1> + <T(n,i+1),X2> + <T(n,i-1),X2> -  2·cos(a) <T(n,i),X2>
/// ```
/// where `x = (xr,xi)` is a complex variable vector, and `X1 = (X1r,X1i)`, 
/// `X2 = (X2r,X2i)` are complex PSD variables.
#[allow(non_snake_case)]
fn trigpoly_0_a(m : & mut Model, xr : & Variable<1>, xi : & Variable<1>, a : f64) {
    assert_eq!(xi.len(),xr.len());
    let n = xi.len()-1;
    let (X1r, X1i) = complex_sdpvar(m, n+1);
    let (X2r, X2i) = complex_sdpvar(m, n);

    let Tn = (0..n+1).map(|i| toeplitz(n,i as i64,1.0));
    let Tnx = (0..n+1).map(|i| toeplitz_ext(n-1, &[i as i64+1,i as i64-1, i as i64], &[1.0,1.0,-2.0*a.cos()]));
    m.constraint(None, 
                 xr.sub(Tn.clone().zip(Tnx.clone()).genexpr(|_,(Tni,Tnix)| Some( X1r.dot(Tni).add(X2r.dot(Tnix))))),
                 equal_to(0.0).with_shape(&[n+1]));

    m.constraint(None,
                 xi.sub(Tn.clone().zip(Tnx.clone()).genexpr(|_,(Tni,Tnix)| Some( X1i.dot(Tni).add(X2i.dot(Tnix))))),
                 equal_to(0.0).with_shape(&[n+1]));
}

/// Models the equation
/// ```math
/// x[i] = <T(n+1,i),X1> - <T(n,i+1),X2> - <T(n,i-1),X2> + 2·cos(a) <T(n,i),X2>
/// ```
/// where `x = (xr,xi)` is a complex variable vector, and `X1 = (X1r,X1i)`,
/// `X2 = (X2r,X2i)` are complex PSD variables.
#[allow(non_snake_case)]
fn trigpoly_a_pi(m : & mut Model, xr : &Variable<1>, xi : &Variable<1>, a : f64) {
    assert_eq!(xi.len(),xr.len());
    let n = xr.len()-1;
    
    let (X1r, X1i) = complex_sdpvar(m, n+1);
    let (X2r, X2i) = complex_sdpvar(m, n);

    let Tn = (0..n+1).map(|i| toeplitz(n,i as i64,1.0));
    let Tnx = (0..n+1).map(|i| toeplitz_ext(n-1, &[i as i64+1,i as i64-1, i as i64], &[-1.0,-1.0,2.0*a.cos()]));

    m.constraint(None, 
                 xr.sub( Tn.clone().zip(Tnx.clone()).genexpr(|_,(Tni,Tnix)| Some(X1r.dot(Tni).add(X2r.clone().dot(Tnix))))),
                 equal_to(0.0).with_shape(&[n+1]));
    m.constraint(None,
                 xi.clone().sub( Tn.zip(Tnx).genexpr(|_,(Tni,Tnix)| Some(X1i.clone().dot(Tni).add(X2i.clone().dot(Tnix))))),
                 equal_to(0.0).with_shape(&[n+1]));
}

enum Either<A,B> {
    Left(A),
    Right(B)
}

/// Models the epigraph 
/// ```math
/// 0 ≤ H(w) ≤ t, for all w ∊ [a, b], 
/// ```
/// where
/// ```math
/// H(w) = x0 + 2*Re{ x1*exp(-jw) + ... + xn*exp(-jwn) }
/// ```
/// and allowed intervals are 
/// ```
/// [a,b] ∊ { [a,pi], [0,b] }
/// ```
fn epigraph(m : & mut Model, xr : &Variable<1>, xi : &Variable<1>, t : Either<&Variable<0>,f64>, a : f64, b : f64) {
    let n = xr.len()-1;
    let ur = m.variable(None,n+1);
    let ui = m.variable(None,n+1);

    match &t {
        Either::Left(ref t)  => m.constraint(None,t.sub(xr.index(0).add(ur.index(0))), zero()),
        Either::Right(ref t) => m.constraint(None,t.into_expr().sub(xr.index(0).add(ur.index(0))), zero())
    };
    m.constraint(None, xr.index(1..n+1).add(ur.index(1..n+1)), zeros(&[n]));
    m.constraint(None, xi.add(&ui), zeros(&[n+1]));

    if a.abs() < 1e-12 && (b-PI).abs() < 1e-12 {
        trigpoly_0_pi(m, &ur, &ui);
    }
    else if a.abs() < 1e-12 && b < PI {
        trigpoly_0_a(m, &ur, &ui, b);
    }
    else if a < PI && (b-PI).abs() < 1e-12 {
        trigpoly_a_pi(m, &ur, &ui, a);
    }
    else {
        panic!("Invalid interval.");
    }
}


/// Models the hypograph 
/// ```math 
/// 0 ≤ t ≤ H(w), for all w ∊ [a, b]
/// ```
/// where
/// ```math
/// H(w) = x0 + 2*Re{ x1*exp(-jw) + ... + xn*exp(-jwn) }
/// ```
/// and
/// allowed intervals are 
/// ```
/// [a,b] ∊ { [a,pi], [0,b] }
/// ```
fn hypograph(m : & mut Model, xr : &Variable<1>, xi : &Variable<1>, t : Either<&Variable<0>,f64>, a : f64, b : f64) 
{
    let n = xr.len()-1;
    let u0 = m.variable(None,&[]);
    match t {
        Either::Left(t) => m.constraint(None,
                 t.sub(xr.index(0).sub(&u0)),
                 zero()),
        Either::Right(t) => m.constraint(None,
                 t.into_expr().sub(xr.index(0).sub(&u0)),
                 zero())
    };

    let ur = Variable::vstack(&[&u0.with_shape(&[1]), &xr.index(1..n+1)]);

    if a.abs() < 1e-12 && (b-PI).abs() < 1e-12 {
        trigpoly_0_pi(m, &ur, xi);
    }
    else if a.abs() < 1e-12 && b < PI {
        trigpoly_0_a(m, &ur, xi, b);
    }
    else if a < PI && (b-PI).abs() < 1e-12 {
        trigpoly_a_pi(m, &ur, xi, a);
    }
    else {
        panic!("Invalid interval.")
    }
}


const APP_ID : &str = "com.mosek.trigpoly";

fn main() -> glib::ExitCode {
    let mut m = Model::new(Some("trigpoly"));

    let n : usize = 10;

    let xr = m.variable(Some("xr"), n+1);
    let xi = m.variable(Some("xi"), n+1);
    
    let wp = PI/4.0;
    let ws = wp + PI/8.0;

    // H(w) >= 0
    trigpoly_0_pi(& mut m, &xr, &xi);

    let delta = 0.05;
    // H(w) <= (1+delta),  w \in [0, wp]
    epigraph(& mut m, &xr, &xi, Either::Right(1.0+delta), 0.0, wp);

    // (1-delta) <= H(w),  w \in [0, wp]
    hypograph(& mut m, &xr, &xi, Either::Right(1.0-delta), 0.0, wp);

    // H(w) < t,          w \in [ws, pi]
    let t = m.variable(Some("t"), nonnegative());
    epigraph(&mut m, &xr, &xi, Either::Left(&t), ws, PI);

    m.objective(None, Sense::Minimize, &t);

    // Enabled log output
    m.set_log_handler(|msg| print!("{}",msg) );

    m.solve();

    let xr = m.primal_solution(SolutionType::Default, &xr).unwrap();
    let xi = m.primal_solution(SolutionType::Default, &xi).unwrap();
    let t  = m.primal_solution(SolutionType::Default, &t).unwrap()[0];

    println!("xr: {:?}", xr);
    println!("xi: {:?}", xi);
    println!("t:  {}",t);

    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    let drawdata = (xr,xi,t,wp,ws);

    app.connect_activate(move |app : &Application| build_ui(app,&drawdata));
    let r = app.run_with_args::<&str>(&[]);
    println!("Main loop exit");
    r

}

#[allow(non_snake_case)]
fn build_ui(app   : &Application,
            ddata : &(Vec<f64>,Vec<f64>,f64,f64,f64))
{    
    let data = Rc::new(RefCell::new(ddata.clone()));
    
    let darea = DrawingArea::builder()
        .width_request(800) 
        .height_request(800)
        .build();

    // Redraw callback
    {
        let data = data.clone();
        darea.set_draw_func(move |widget,context,w,h| redraw_window(widget,context,w,h,&data.borrow()));
    }

    let window = ApplicationWindow::builder()
        .application(app)
        .title("Hello Trigpoly")
        .child(&darea)
        .build();
    
    window.present();
}


#[allow(non_snake_case)]
fn H(w : f64, xr : &[f64],xi : &[f64]) -> f64 {
    xr[0] + 2.0*(1..xr.len()).zip(xr[1..].iter().zip(xi[1..].iter()))
        .map(|(k,(&xrk,&xik))| xrk*(w*k as f64).cos() + xik*(w*k as f64).sin())
        .sum::<f64>()
}

#[allow(non_snake_case)]
fn redraw_window(_widget : &DrawingArea, context : &Context, w : i32, h : i32, data : &(Vec<f64>,Vec<f64>,f64,f64,f64)) {
    let (xr,xi,t,wp,ws) = data;
//   
//   p = graph.axis.painter.regular(basepathattrs=[deco.earrow.normal])
//
//   xticks = [ graph.axis.tick.tick(wp, label='$\omega_p$'),
//           graph.axis.tick.tick(ws, label='$\omega_s$'),
//           graph.axis.tick.tick(pi, label='$\pi$') ]
//
//   yticks = [ graph.axis.tick.tick(1+delta, label='$1+\delta$'),
//           graph.axis.tick.tick(1-delta, label='$1-\delta$'),
//           graph.axis.tick.tick(t, label='$t^\star$') ]
//
//   g = graph.graphxy(width=8, x2=None, y2=None,
//                   x=graph.axis.linear(title="$\omega$", min=0, max=pi+0.2,
//                                       manualticks=xticks,
//                                       painter=p,
//                                       parter=None),
//#                    y=graph.axis.linear(title="$H(\omega)$", min=0, max=1.2,
//#                                        manualticks=yticks,
//#                                        painter=p,
//#                                        parter=None))
//                   y=graph.axis.log(title="$H(\omega)$",
//                                    min=t/100,
//                                    #manualticks=yticks,
//                                    painter=p,
//                                    parter=None))
//   
//   g.plot(graph.data.function("y(x)=H(x)", context=locals(), points=500))
//
//   (x1, y1), (x2, y2) = g.pos(0.0, 1.0+delta), g.pos(wp,  1.0+delta)
//   g.stroke(path.line(x1, y1, x2, y2), [style.linestyle.dashed])
//
//   (x1, y1), (x2, y2) = g.pos(0.0, 1.0-delta), g.pos(wp,  1.0-delta)
//   g.stroke(path.line(x1, y1, x2, y2), [style.linestyle.dashed])
//
//   (x1, y1), (x2, y2) = g.pos(ws, t), g.pos(pi, t)
//   g.stroke(path.line(x1, y1, x2, y2), [style.linestyle.dashed])
//
//   g.writeEPSfile("trigpoly")
//   g.writePDFfile("trigpoly")
//   print("generated trigpoly.eps")
    context.set_source_rgb(1.0, 1.0, 1.0);
    _ = context.paint();

    let w : f64 = w.into();
    let h : f64 = h.into();
    let wscale = std::f64::consts::PI+0.2;


    const margin : f64 = 15.0;
    context.set_source_rgb(0.0, 0.0, 0.0);
    context.move_to(margin,margin);
    context.line_to(margin,h-margin);
    _ = context.stroke();
    
    context.move_to(margin,h/2.0);
    context.line_to(w-margin,h/2.0);
    _ = context.stroke();


    context.set_matrix(cairo::Matrix::identity());
    context.translate(margin,h/2.0);

    let Hw : Vec<(f64,f64)> = (0..500).map(|i| { let w = wscale/500.0*(i as f64); (w,H(w,xr.as_slice(),xi.as_slice())) } ).collect::<Vec<(f64,f64)>>();

    let (Hmin,Hmax) : (f64,f64) = match Hw.iter().minmax_by(|&(_,a),&(_,b)| if a < b { Ordering::Less } else if b < a { Ordering::Greater } else { Ordering::Equal } ) {
        itertools::MinMaxResult::NoElements => panic!("Broken"),
        itertools::MinMaxResult::OneElement((_,e)) => (*e,*e),
        itertools::MinMaxResult::MinMax((_,a),(_,b)) => (*a,*b)
    };

    let hscale = Hmin.abs().max(Hmax.abs()) * 2.0;
    context.scale((w-margin*2.0)/wscale, -(h-margin*2.0)/hscale);




    context.set_source_rgb(0.0, 0.0, 0.0);
    if let Some((x,y)) = Hw.first() {
        context.move_to(*x,*y);
        for (x,y) in Hw[1..].iter() {
            context.line_to(*x,*y);
        }
        context.set_matrix(cairo::Matrix::identity());
        _ = context.stroke();
    }
}
