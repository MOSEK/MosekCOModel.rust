extern crate cairo;
extern crate glam;
extern crate mosekcomodel;

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
use mosekcomodel::{unbounded, Model};

const APP_ID : &str = "com.mosek.lowner-john";
const SPEED_SCALE : f64 = 0.1;

#[allow(non_snake_case)]
#[derive(Clone)]
struct DrawData {
    radius : Vec<[f64;2]>,
    center : Vec<[f64;2]>,
    speed  : Vec<[f64;2]>,

    t0     : SystemTime,

    // Fixed ellipsoids
    Abs : Vec<([f64;4],[f64;2])>,
    // Bounding ellipsoid as { x : || Px+q || < 1 } 
    Pc : Option<([f64;4],[f64;2])>,
    Qd : Option<([f64;4],[f64;2])>,
}



pub fn main() -> glib::ExitCode {
    //let mut drawdata = Rc::new(RefCell::new(DrawData{
    #[allow(non_snake_case)]
    let drawdata = DrawData{
        radius : vec![[0.2,0.15],[0.3,0.2],[0.4, 0.2]],
        center : vec![[0.2,0.2],[-0.2,0.1],[0.2,-0.2]],
        speed  : vec![[0.1,0.3],[-0.3,0.5],[0.4,-0.3]],

        t0 : SystemTime::now(),

        Abs : vec![],
        Pc : None,
        Qd : None,
    };

    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    app.connect_activate(move | app : &Application | build_ui(app,&drawdata));

    let r = app.run_with_args::<&str>(&[]);
    println!("Main loop exit!");

    r
}

#[allow(non_snake_case)]
fn build_ui(app   : &Application,
            ddata : &DrawData)
{    
    // tx Send info from solver to GUI
    // rtx Send commands from GUI to solver
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
    
    { // Time callback
        let data = data.clone();
        let darea = darea.clone();
        glib::source::timeout_add_local(
            Duration::from_millis(10), 
            move || {
                let mut data = data.borrow_mut();
                let dt = 0.001 * (SystemTime::now().duration_since(data.t0).unwrap().as_millis() as f64);

                data.Abs = izip!(data.radius.iter(),data.center.iter(),data.speed.iter())
                    .map(|(&r,&c,&v)| {
                        let theta_g = (2.0 * std::f64::consts::PI * v[0] * dt * SPEED_SCALE) % (2.0 * std::f64::consts::PI);
                        let theta_l = (2.0 * std::f64::consts::PI * v[1] * dt * SPEED_SCALE) % (2.0 * std::f64::consts::PI);

                        let (cost,sint) = ((theta_l/2.0).cos() , (theta_l/2.0).sin());
                        let A = [ cost.powi(2)*r[0]+sint.powi(2)*r[1], cost*sint*(r[1]-r[0]),
                                  cost*sint*(r[1]-r[0]), sint.powi(2) * r[0] + cost.powi(2) * r[1] ];                            
                        let b = [ theta_g.cos()*c[0] - theta_g.sin()*c[1],
                                  theta_g.sin()*c[0] + theta_g.cos()*c[1]];
                        (A,b)
                    }).collect();

                      
                {
                    // outer ellipsoid
                    let mut m = Model::new(None);
                    let t = m.variable(None, unbounded());
                    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
                    let q = m.variable(None, unbounded().with_shape(&[2]));
  
                    m.objective(None, mosekcomodel::Sense::Maximize, &t);
                   
                    for (A,b) in data.Abs.iter() {
                        let A = DMat2::from_cols_array(A).inverse();
                        let b = A.mul_vec2(DVec2{x:b[0], y:b[1]}).to_array();

                        let e : Ellipsoid<2> = ellipsoids::Ellipsoid::from_arrays(&A.to_cols_array(), &[-b[0],-b[1]]);

                        ellipsoids::ellipsoid_contains(&mut m,&p,&q,&e);
                    }

                    m.solve();
  
                    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekcomodel::SolutionType::Default,&p),
                                                  m.primal_solution(mosekcomodel::SolutionType::Default,&q)) {
                        
                        // A² = P => A = sqrt(P)
                        // Ab = q => A\q
                        let s = (psol[0]*psol[3]-psol[1]*psol[2]).sqrt();

                        let A = DMat2::from_cols_array(&[psol[0],psol[1],psol[2],psol[3]]).add_mat2(&DMat2::from_cols_array(&[s,0.0,0.0,s])).mul_scalar(1.0/(psol[0]+psol[3] + 2.0*s).sqrt());
                        let b = A.inverse().mul_vec2(DVec2::from_array([qsol[0],qsol[1]]));

                        data.Pc = Some((A.to_cols_array(),b.to_array()));
                    }
                    else {
                        data.Pc = None;
                    }
                }


                {
                    // inner ellipsoid
                    let mut m = Model::new(None);

                    let t = m.variable(None, unbounded());
                    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
                    let q = m.variable(None, unbounded().with_shape(&[2]));

                    m.objective(None, mosekcomodel::Sense::Maximize, &t);

                    for (A,b) in data.Abs.iter() {
                        let A = DMat2::from_cols_array(A).inverse();
                        let b = A.mul_vec2(DVec2{x:b[0], y:b[1]}).to_array();

                        let e : Ellipsoid<2> = ellipsoids::Ellipsoid::from_arrays(&A.to_cols_array(), &[-b[0],-b[1]]);

                        ellipsoids::ellipsoid_contained(&mut m,&p,&q,&e);
                    }

                    m.solve();

                    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekcomodel::SolutionType::Default,&p),
                                                  m.primal_solution(mosekcomodel::SolutionType::Default,&q)) {
                        let A = DMat2::from_cols_array(&[psol[0],psol[1],psol[2],psol[3]]).inverse();
                        let b = A.mul_vec2(DVec2::from_array([qsol[0],qsol[1]])).to_array();

                        data.Qd = Some((A.to_cols_array(),[-b[0],-b[1]]));
                    }
                    else {
                        data.Qd = None;
                    }
                }

                darea.queue_draw();
                ControlFlow::Continue
            });
    }    

    window.present();
}

#[allow(non_snake_case)]
fn redraw_window(_widget : &DrawingArea, context : &Context, w : i32, h : i32, data : &DrawData) {
    context.set_source_rgb(1.0, 1.0, 1.0);
    _ = context.paint();

    let w : f64 = w.into();
    let h : f64 = h.into();
    let s = w.min(h);

    context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
    context.translate(s/2.0, s/2.0);
    context.scale(0.8*s, 0.8*s);
    let mx = context.matrix();


    context.set_source_rgb(0.0, 0.0, 0.0);
    for (A,b) in data.Abs.iter() {
        context.transform(cairo::Matrix::new(A[0],A[1],A[2],A[3],b[0],b[1]));
        context.arc(0.0,0.0,1.0,0.0,std::f64::consts::PI*2.0);
        context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
        _ = context.stroke();
        context.set_matrix(mx);
    }

    if let Some((p,q)) = data.Pc {
        context.set_source_rgb(1.0, 0.0, 0.0);

        let Z = DMat2::from_cols_array(&p).inverse().to_cols_array();
        let w = [ -Z[0] * q[0] - Z[2] * q[1],
                  -Z[1] * q[0] - Z[3] * q[1]];

        context.transform(cairo::Matrix::new(Z[0], Z[1], Z[2], Z[3],w[0],w[1]));
        context.arc(0.0, 0.0, 1.0, 0.0, std::f64::consts::PI*2.0);
        context.set_matrix(cairo::Matrix::new(2.0,0.0,0.0,2.0,0.0,0.0));
        _ = context.stroke();
   
        context.set_matrix(mx);
    }

    if let Some((p,q)) = data.Qd {
        context.set_source_rgb(0.0, 1.0, 0.0);

        let Z = DMat2::from_cols_array(&p).inverse().to_cols_array();
        let w = [ -Z[0] * q[0] - Z[2] * q[1],
                  -Z[1] * q[0] - Z[3] * q[1]];

        context.transform(cairo::Matrix::new(Z[0], Z[1], Z[2], Z[3],w[0],w[1]));
        context.arc(0.0, 0.0, 1.0, 0.0, std::f64::consts::PI*2.0);
        context.set_matrix(cairo::Matrix::new(2.0,0.0,0.0,2.0,0.0,0.0));
        _ = context.stroke();
   
        context.set_matrix(mx);
    }
}
