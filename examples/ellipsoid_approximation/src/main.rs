extern crate cairo;
extern crate glam;
extern crate mosekmodel;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, SystemTime};
use glam::{DMat2,DVec2};
use gtk::glib::ControlFlow;
use gtk::prelude::*;
use itertools::izip;

use cairo::Context;
use gtk::{glib,Application, DrawingArea, ApplicationWindow};
use mosekmodel::{unbounded, Model};

const APP_ID : &str = "com.mosek.lowner-john";

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
}



pub fn main() -> glib::ExitCode {
    //let mut drawdata = Rc::new(RefCell::new(DrawData{
    let drawdata = DrawData{
        radius : vec![[0.2,0.15],[0.3,0.2],[0.4,0.2]],
        center : vec![[0.2,0.2],[-0.2,0.1],[0.2,-0.2]],
        speed  : vec![[0.1,0.3],[-0.3,0.5],[0.4,-0.1]],

        t0 : SystemTime::now(),

        Abs : vec![],
        Pc : None,
    };

    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    app.connect_activate(move | app : &Application | build_ui(app,&drawdata));

    let r = app.run_with_args::<&str>(&[]);
    println!("Main loop exit!");

    r
}


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
                        let theta_g = (2.0 * std::f64::consts::PI * v[0] * dt) % (2.0 * std::f64::consts::PI);
                        let theta_l = (2.0 * std::f64::consts::PI * v[1] * dt) % (2.0 * std::f64::consts::PI);

                        let rmxg = DMat2::from_cols_array( &[theta_g.cos(), theta_g.sin(), -theta_g.sin(), theta_g.cos()]);

                        let rmxl = DMat2::from_cols_array( &[(theta_l/2.0).cos(), -(theta_l/2.0).sin(), (theta_l/2.0).sin(), (theta_l/2.0).cos()]);

                        let A = rmxl.mul_mat2(&DMat2::from_cols_array(&[r[0],0.0,0.0,r[1]])).mul_mat2(&rmxl.transpose());
                        let b = rmxg.mul_vec2(DVec2::from_array(c));

                        println!("A = {:?}",A);

                        assert!((A.col(0)[0] - A.col(1)[1]).abs() < 1.0e-10);

                        (A.to_cols_array(),b.to_array())
                    }).collect();

                      
                {
                    let mut m = Model::new(None);
                    let t = m.variable(None, unbounded());
                    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
                    let q = m.variable(None, unbounded().with_shape(&[2]));
  
                    m.objective(None, mosekmodel::Sense::Maximize, &t);
                   
                    for (A,b) in data.Abs.iter() {
                        let A = DMat2::from_cols_array(A).inverse();
                        let b = A.mul_vec2(DVec2{ x : -b[0], y : -b[1]});
                        let mut Adata = [[0.0;2];2]; 
                        Adata.iter_mut().flat_map(|v| v.iter_mut()).zip(A.transpose().to_cols_array().iter())
                            .for_each(|(t,&s)| *t = s);

                        let e = ellipsoids::Ellipsoid::new(&Adata, &b.to_array()); 
                        ellipsoids::ellipsoid_contains(&mut m,&p,&q,&e);
                    }

                    m.solve();
  
                    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
                                                  m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
                        data.Pc = Some(([psol[0],psol[1],psol[2],psol[3]],[qsol[0],qsol[1]]));
                     }
                }

                darea.queue_draw();
                ControlFlow::Continue
            });
    }    

    window.present();
}



fn redraw_window(_widget : &DrawingArea, context : &Context, w : i32, h : i32, data : &DrawData) {
    context.set_source_rgb(1.0, 1.0, 1.0);
    _ = context.paint();

    let w : f64 = w.into();
    let h : f64 = h.into();
    let s = w.min(h);

    context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
    context.translate(s/2.0, s/2.0);
    context.scale(0.8*s, 0.8*s);
    context.set_source_rgb(0.0, 0.0, 0.0);

    for (A,b) in data.Abs.iter() {
        let mx = context.matrix();
        context.transform(cairo::Matrix::new(A[0],A[1],A[2],A[3],b[0],b[1]));
        context.arc(0.0,0.0,1.0,0.0,std::f64::consts::PI*2.0);
        context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
        _ = context.stroke();
        context.set_matrix(mx);
    }


    if let Some((p,q)) = data.Pc {
        context.set_source_rgb(1.0, 0.0, 0.0);
        let mx = context.matrix();
        let a = DMat2::from_cols_array(&p).transpose().inverse();
        let b = a.mul_vec2(q.into());
   
        let adata = a.to_cols_array();
        //context.transform(cairo::Matrix::new(adata[0], adata[1], adata[2], adata[3], b.x,b.y));
        context.transform(cairo::Matrix::new(adata[0], adata[1], adata[2], adata[3], -b.x,-b.y));
   
        context.arc(0.0, 0.0, 1.0, 0.0, std::f64::consts::PI*2.0);
   
        context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
        _ = context.stroke();
   
        context.set_matrix(mx);
    }
}
