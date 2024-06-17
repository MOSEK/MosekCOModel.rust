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
//use cairo::glib::controlflow;
use gtk::{glib,Application, DrawingArea, ApplicationWindow};
use mosekmodel::{unbounded, Model};
//use gtk::prelude::*;
//use rand::random;

const APP_ID : &str = "com.mosek.lowner-john";

#[allow(non_snake_case)]
#[derive(Clone)]
struct DrawData {
    points         : Vec<[f64;2]>,
    polygons       : Vec<usize>,
    polygon_center : Vec<[f64;2]>,
    polygon_speed  : Vec<[f64;2]>,

    t0     : SystemTime,

    // Bounding ellipsoid as { x : || Ax+b || < 1 } 
    tpoints : Vec<[f64;2]>,
    Pc : Option<([f64;4],[f64;2])>,

}



pub fn main() -> glib::ExitCode {
    //let mut drawdata = Rc::new(RefCell::new(DrawData{
    let mut drawdata = DrawData{
        points : vec![ 
            [-0.25,0.0],[-0.05,-0.1],[-0.1,0.4],
            [0.1,0.1],[0.4,0.1],[0.4,0.4],[0.1,0.4],
            [0.0,-0.1],[0.2,-0.1],[0.4,-0.2],[0.1,-0.4],[-0.2,-0.2]],
        polygons : vec![0,3,7,12],
        polygon_center : vec![[-0.15,0.05],[0.25,0.25],[0.1,-0.2]],
        polygon_speed : vec![[0.1,0.3],[-0.3,0.5],[0.4,-0.1]],

        t0 : SystemTime::now(),

        tpoints : vec![[0.0;2]; 12],

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

                let tpoints = izip!(data.polygons.iter(),
                                    data.polygons[1..].iter(),
                                    data.polygon_speed.iter(),
                                    data.polygon_center.iter())
                    .flat_map(|(&pb,&pe,v,c)| {
                        let theta_g = (2.0 * std::f64::consts::PI * v[0] * dt) % (2.0 * std::f64::consts::PI);
                        let theta_l = (2.0 * std::f64::consts::PI * v[1] * dt) % (2.0 * std::f64::consts::PI);

                        let rmxg = DMat2::from_cols_array( &[theta_g.cos(), theta_g.sin(), -theta_g.sin(), theta_g.cos()]);
                        let rmxl = DMat2::from_cols_array( &[theta_l.cos(), theta_l.sin(), -theta_l.sin(), theta_l.cos()]);

                        data.points[pb..pe].iter().map(move |p| {
                            let p = DVec2{ x:p[0], y:p[1] };
                            let p = rmxl.mul_vec2(DVec2{ x:p.x-c[0], y:p.y-c[1] });
                            let p = rmxg.mul_vec2(DVec2{ x:p.x+c[0], y:p.y+c[1]  });
                            [p.x,p.y]
                        })
                    }).collect();
                data.tpoints = tpoints;

                
                {
                    let mut m = Model::new(None);
                    let t = m.variable(None, unbounded());
                    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
                    let q = m.variable(None, unbounded().with_shape(&[2]));

                    m.objective(None, mosekmodel::Sense::Maximize, &t);
                    ellipsoids::ellipsoid_contains_points(& mut m, &p, &q, data.tpoints.as_ref());

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

    for (&pb,&pe) in izip!(data.polygons.iter(),data.polygons[1..].iter()) {
        let pts = &data.tpoints[pb..pe];
        if let Some(p) = pts.last() {
            context.move_to(p[0], p[1]);
            for p in pts.iter() { context.line_to(p[0], p[1]); }

            let mx = context.matrix();
            context.set_matrix(cairo::Matrix::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0));
            _ = context.stroke();
            context.set_matrix(mx);
        }
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
