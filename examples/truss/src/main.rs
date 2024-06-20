extern crate cairo;
extern crate glam;
extern crate mosekmodel;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, SystemTime};
use ellipsoids::Ellipsoid;
use glam::{DMat2,DVec2};
use gtk::glib::ControlFlow;
use gtk::prelude::*;
use itertools::izip;

use cairo::Context;
use gtk::{glib,Application, DrawingArea, ApplicationWindow};
use mosekmodel::expr::*;
use mosekmodel::matrix::SparseMatrix;
use mosekmodel::{hstack, in_rotated_quadratic_cones, unbounded, nonnegative,equal_to,zero, Model, Sense, SolutionType};

const APP_ID : &str = "com.mosek.truss-linear";

#[allow(non_snake_case)]
#[derive(Clone)]
struct DrawData {
    points       : Vec<[f64;2]>,
    node_type    : Vec<bool>,
    arcs         : Vec<(usize,usize)>,
    external_force : Vec<[f64;2]>,
    total_material_volume : f64,
    kappa        : f64,
    arc_volume   : Option<Vec<f64>>,
}
const D : usize = 2;

pub fn main() {
    #[allow(non_snake_case)]
    let mut drawdata = DrawData{
        points : vec![
            [0.0, 3.0], [0.0,1.0], [0.0,-1.0],[0.0,-3.0],
            [2.0, 2.0], [2.0, 0.0], [2.0, -2.0],
            [4.0,1.0],[4.0,-1.0],
            [7.0,0.0] ],
        node_type : vec![
            true,true,true,true,
            false,false,false,
            false,false,
            false],
        arcs : vec![
            (0,4),(0,5),
            (1,4),(1,5),(1,6),
            (2,4),(2,5),(2,6),
            (3,5),(3,6),
            (4,5),(4,7),(4,8),
            (5,6),(5,7),(5,8),
            (6,7),(6,8),
            (7,8),(7,9),
            (8,9)],

        external_force : vec![ [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,0.0], 
                               [0.0,-2.0] ],
        total_material_volume : 20.0,
        kappa : 1.0,

        arc_volume : None,
    };
    let numnodes = drawdata.points.len();
    let numarcs  = drawdata.arcs.len();

    // b is a parameter such that 
    // b is a (numarcs x (D*numnodes)) matrix. Rows are indexes by nodes, colunms are indexed by arcs,
    // so each column has an associated (i,j) ∊ A. The element b_{k,(i,j)} means row k, column
    // associated with (i,j). The matrix is build as
    //    b_{j,(ij)} = κ(p_j-p_i)/||p_j-p_i||^2  for (i,j) ∊ A
    //    b_{i,(ij)} = -κ(p_j-p_i)/||p_j-p_i||^2  for (i,j) ∊ A
    //    0 everywhere else.
    let sqrtkappa = drawdata.kappa.sqrt();
    let b = SparseMatrix::from_iterator(
        numarcs, 
        numnodes*D, 
        drawdata.arcs.iter().enumerate().flat_map(|(arci,&(i,j))| {
            let pi = drawdata.points[i];
            let pj = drawdata.points[j];
            let ti = drawdata.node_type[i];
            let tj = drawdata.node_type[j];

            
            let d = (pj[0]-pi[0], pj[1]-pi[1]);
            let sqrnormd = d.0.powi(2) + d.1.powi(2);
            
            std::iter::once(           (arci, j*D,   if !tj { sqrtkappa * d.0 / sqrnormd } else { 0.0 }))
                .chain(std::iter::once((arci, j*D+1, if !tj { sqrtkappa * d.1 / sqrnormd } else { 0.0 })))
                .chain(std::iter::once((arci, i*D,   if !ti { -sqrtkappa * d.0 / sqrnormd } else { 0.0 })))
                .chain(std::iter::once((arci, i*D+1, if !ti { -sqrtkappa * d.1 / sqrnormd } else { 0.0 })))
        }));

    let mut m = Model::new(Some("Truss"));
    let tau = m.variable(Some("tau"), unbounded());
    //let tau = m.variable(Some("tau"), equal_to(20.0));
    let sigma = m.variable(Some("sigma"), unbounded().with_shape(&[numarcs]));
    let t = m.variable(Some("t"),unbounded().with_shape(&[numarcs]));
    let s = m.variable(Some("s"),unbounded().with_shape(&[numarcs]));
    let w = m.variable(Some("w"),equal_to(drawdata.total_material_volume));

    // (1)
    m.objective(None, Sense::Minimize, &tau);
    //m.objective(None, Sense::Minimize, &w);

    // (2)
    m.constraint(Some("t_sigma_s"),
                 &hstack![t.clone().reshape(&[numarcs,1]),
                          sigma.clone().reshape(&[numarcs,1]),
                          s.clone().reshape(&[numarcs,1])],
                 in_rotated_quadratic_cones(&[numarcs,3], 1));
    // (3)
    m.constraint(Some("sum_sigma"),
                 &tau.clone().sub(sigma.clone().sum()),
                 zero());
        
    // (4) 
    m.constraint(Some("total_volume"),
                 &t.clone().sum().sub(w),
                 zero());
    // (5)
    let f : Vec<f64> = drawdata.external_force.iter().flat_map(|row| row.iter()).cloned().collect();
    m.constraint(Some("force_balance"), 
                 &s.clone().square_diag().mul(b).sum_on(&[1]),
                 equal_to(f));

    m.solve();

    m.write_problem("truss.ptf");

    let tsol = m.primal_solution(SolutionType::Default,&t).expect("No solution available");
    for (t,(i,j)) in tsol.iter().zip(drawdata.arcs.iter()) {
        println!("Arg ({},{}): volume = {:.3}",i,j,t);
    }

    drawdata.arc_volume = Some(tsol.to_vec());




    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    app.connect_activate(move | app : &Application | build_ui(app,&drawdata));

    let r = app.run_with_args::<&str>(&[]);
    println!("Main loop exit!");
}


#[allow(non_snake_case)]
fn build_ui(app   : &Application,
            ddata : &DrawData)
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
        .title("Hello Löwner-John")
        .child(&darea)
        .build();

//       if (false)
//       { // Time callback
//           let data = data.clone();
//           let darea = darea.clone();
//           glib::source::timeout_add_local(
//               Duration::from_millis(10),
//               move || {
//                   let mut data = data.borrow_mut();
//
//
//
//                   let dt = 0.001 * (SystemTime::now().duration_since(data.t0).unwrap().as_millis() as f64);
//
//                   data.Abs = izip!(data.radius.iter(),data.center.iter(),data.speed.iter())
//                       .map(|(&r,&c,&v)| {
//                           let theta_g = (2.0 * std::f64::consts::PI * v[0] * dt * SPEED_SCALE) % (2.0 * std::f64::consts::PI);
//                           let theta_l = (2.0 * std::f64::consts::PI * v[1] * dt * SPEED_SCALE) % (2.0 * std::f64::consts::PI);
//
//                           let (cost,sint) = ((theta_l/2.0).cos() , (theta_l/2.0).sin());
//                           let A = [ cost.powi(2)*r[0]+sint.powi(2)*r[1], cost*sint*(r[1]-r[0]),
//                                     cost*sint*(r[1]-r[0]), sint.powi(2) * r[0] + cost.powi(2) * r[1] ];
//                           let b = [ theta_g.cos()*c[0] - theta_g.sin()*c[1],
//                                     theta_g.sin()*c[0] + theta_g.cos()*c[1]];
//                           (A,b)
//                       }).collect();
//
//
//                   {
//                       // outer ellipsoid
//                       let mut m = Model::new(None);
//                       let t = m.variable(None, unbounded());
//                       let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
//                       let q = m.variable(None, unbounded().with_shape(&[2]));
//
//                       m.objective(None, mosekmodel::Sense::Maximize, &t);
//
//                       for (A,b) in data.Abs.iter() {
//                           let A = DMat2::from_cols_array(A).inverse();
//                           let b = A.mul_vec2(DVec2{x:b[0], y:b[1]}).to_array();
//
//                           let e : Ellipsoid<2> = ellipsoids::Ellipsoid::from_arrays(&A.to_cols_array(), &[-b[0],-b[1]]);
//
//                           ellipsoids::ellipsoid_contains(&mut m,&p,&q,&e);
//                       }
//
//                       m.solve();
//
//                       if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
//                                                     m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
//
//                           // A² = P => A = sqrt(P)
//                           // Ab = q => A\q
//                           let s = (psol[0]*psol[3]-psol[1]*psol[2]).sqrt();
//
//                           let A = DMat2::from_cols_array(&[psol[0],psol[1],psol[2],psol[3]]).add_mat2(&DMat2::from_cols_array(&[s,0.0,0.0,s])).mul_scalar(1.0/(psol[0]+psol[3] + 2.0*s).sqrt());
//                           let b = A.inverse().mul_vec2(DVec2::from_array([qsol[0],qsol[1]]));
//
//                           data.Pc = Some((A.to_cols_array(),b.to_array()));
//                       }
//                       else {
//                           data.Pc = None;
//                       }
//                   }
//
//
//                   {
//                       // inner ellipsoid
//                       let mut m = Model::new(None);
//
//                       let t = m.variable(None, unbounded());
//                       let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
//                       let q = m.variable(None, unbounded().with_shape(&[2]));
//
//                       m.objective(None, mosekmodel::Sense::Maximize, &t);
//
//                       for (A,b) in data.Abs.iter() {
//                           let A = DMat2::from_cols_array(A).inverse();
//                           let b = A.mul_vec2(DVec2{x:b[0], y:b[1]}).to_array();
//
//                           let e : Ellipsoid<2> = ellipsoids::Ellipsoid::from_arrays(&A.to_cols_array(), &[-b[0],-b[1]]);
//
//                           ellipsoids::ellipsoid_contained(&mut m,&p,&q,&e);
//                       }
//
//                       m.solve();
//
//                       if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
//                                                     m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
//                           let A = DMat2::from_cols_array(&[psol[0],psol[1],psol[2],psol[3]]).inverse();
//                           let b = A.mul_vec2(DVec2::from_array([qsol[0],qsol[1]])).to_array();
//
//                           data.Qd = Some((A.to_cols_array(),[-b[0],-b[1]]));
//                       }
//                       else {
//                           data.Qd = None;
//                       }
//                   }
//
//                   darea.queue_draw();
//                   ControlFlow::Continue
//               });
//       }

    window.present();
}

fn norm<const N : usize>(p : &[f64;N]) -> f64 {
    p.iter().cloned().map(f64::abs).sum::<f64>().sqrt()
}

#[allow(non_snake_case)]
fn redraw_window(_widget : &DrawingArea, context : &Context, w : i32, h : i32, data : &DrawData) {
    context.set_source_rgb(1.0, 1.0, 1.0);
    _ = context.paint();

    let w : f64 = w.into();
    let h : f64 = h.into();
    let s = w.min(h);

    context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
    context.translate(0.0,h/2.0);
    context.scale(s/10.0, -s/10.0);
    let mx = context.matrix();

    // ARCS
    context.set_source_rgb(0.0, 0.0, 0.0);    
    if let Some(ref volume) = data.arc_volume {
        for (&(i,j),&v) in data.arcs.iter().zip(volume.iter()) {
            let pi = data.points[i];
            let pj = data.points[j];
        
            println!("Arc volume = {:?}",volume);

            if v > 1.0e-4 {
                let w = v / norm(&[ pj[0]-pi[0], pj[1]-pi[1] ]);
                context.set_line_width(w*2.0);
                context.move_to(pi[0], pi[1]);
                context.line_to(pj[0], pj[1]);

                context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
                _ = context.stroke();
                context.set_matrix(mx);
            }
        }
    }
    // NODES
    context.set_source_rgb(0.0, 0.0, 0.5);
    for p in data.points.iter() {
        context.arc(p[0],p[1],0.1,0.0,std::f64::consts::PI*2.0);

        context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
        _ = context.fill();
        context.set_matrix(mx);
    }

    context.set_source_rgb(0.0, 0.0, 0.0);
    for (&f,p) in data.node_type.iter().zip(data.points.iter()) {
        if f {
            context.arc(p[0],p[1],0.1,0.0,std::f64::consts::PI*2.0);
            context.set_matrix(cairo::Matrix::new(3.0,0.0,0.0,3.0,0.0,0.0));
            _ = context.stroke();
            context.set_matrix(mx);
        }
    }

    context.set_source_rgb(1.0, 0.0, 0.0);
    for (f,p) in data.external_force.iter().zip(data.points.iter()) {
        if norm(f) > 0.0 {
            context.move_to(p[0],p[1]);
            context.line_to(p[0]+f[0],p[1]+f[1]);

            context.move_to(p[0]+f[0]-0.1,p[1]+f[1]+0.1);
            context.line_to(p[0]+f[0],p[1]+f[1]);
            context.line_to(p[0]+f[0]+0.1,p[1]+f[1]+0.1);

            context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
            _ = context.stroke();
            context.set_matrix(mx);
        }
    }



//       for (A,b) in data.Abs.iter() {
//           context.transform(cairo::Matrix::new(A[0],A[1],A[2],A[3],b[0],b[1]));
//           context.arc(0.0,0.0,1.0,0.0,std::f64::consts::PI*2.0);
//           context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
//           _ = context.stroke();
//           context.set_matrix(mx);
//       }
//
//       if let Some((p,q)) = data.Pc {
//           context.set_source_rgb(1.0, 0.0, 0.0);
//
//           let Z = DMat2::from_cols_array(&p).inverse().to_cols_array();
//           let w = [ -Z[0] * q[0] - Z[2] * q[1],
//                     -Z[1] * q[0] - Z[3] * q[1]];
//
//           context.transform(cairo::Matrix::new(Z[0], Z[1], Z[2], Z[3],w[0],w[1]));
//           context.arc(0.0, 0.0, 1.0, 0.0, std::f64::consts::PI*2.0);
//           context.set_matrix(cairo::Matrix::new(2.0,0.0,0.0,2.0,0.0,0.0));
//           _ = context.stroke();
//
//           context.set_matrix(mx);
//       }
//
//       if let Some((p,q)) = data.Qd {
//           context.set_source_rgb(0.0, 1.0, 0.0);
//
//           let Z = DMat2::from_cols_array(&p).inverse().to_cols_array();
//           let w = [ -Z[0] * q[0] - Z[2] * q[1],
//                     -Z[1] * q[0] - Z[3] * q[1]];
//
//           context.transform(cairo::Matrix::new(Z[0], Z[1], Z[2], Z[3],w[0],w[1]));
//           context.arc(0.0, 0.0, 1.0, 0.0, std::f64::consts::PI*2.0);
//           context.set_matrix(cairo::Matrix::new(2.0,0.0,0.0,2.0,0.0,0.0));
//           _ = context.stroke();
//
//           context.set_matrix(mx);
//       }
}

