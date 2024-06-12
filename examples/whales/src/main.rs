extern crate mosekmodel;
extern crate cairo;
mod whales;
mod utils2d;
mod matrix;
mod ellipsoids;

use whales::{Ellipsoid,minimal_bounding_ellipsoid};
use utils2d::{det,matscale,matmul,matadd,trace,inv};

use cairo::Context;
use cairo::glib::ControlFlow;
use gtk::{glib,Application, DrawingArea, ApplicationWindow, GestureClick,Orientation};
use gtk::prelude::*;
use glam::{DMat2,DVec2};
//use rand::random;

const APP_ID : &str = "com.mosek.whales";

#[allow(non_snake_case)]
#[derive(Clone)]
struct DrawData {
    ellipses : Vec<Ellipsoid<2>>,
    bnd : Ellipsoid<2>,
}


fn ellipse_from_stheta(s : [f64;2], d : [f64;2], theta : f64) -> Ellipsoid<2> {
    let Rinv = [[  theta.cos()/s[0], - theta.sin()/s[1] ],
                [  theta.sin()/s[0],   theta.cos()/s[1] ]];
    Ellipsoid::new( &Rinv,
                    &[ - (d[0]*Rinv[0][0]+d[1]*Rinv[0][1]), -(d[0]*Rinv[1][0]) + d[1]*Rinv[1][1]])
}

#[allow(non_snake_case)]
fn main() -> Result<(),String> {
    let mut drawdata = DrawData {
        ellipses : vec![
            //Ellipsoid::new(&[[1.09613, -0.236851], [-0.236851, 0.539075]], &[ 0.596594, 1.23438] ),
            //Ellipsoid::new(&[[1.01769, -0.613843], [-0.613843, 2.52996 ]], &[-1.74633, -0.568805]),
            //Ellipsoid::new(&[[1.26487,  0.319239], [ 0.319239, 1.28526 ]], &[-0.856775, 1.29365] ),
            //Ellipsoid::new(&[[0.926849,-0.339339], [-0.339339, 1.19551 ]], &[ 0.452287, 0.575005]),
            //Ellipsoid::new(&[[0.819939,-0.0866013],[-0.0866013,0.824379]], &[-0.985105,-1.6824]  ),
            //Ellipsoid::new(&[[0.417981,-0.0699427],[-0.0699427,1.61654 ]], &[-1.73581,  0.118404]),

            ellipse_from_stheta([10.0,1.0], [0.0,3.0], 0.0/*std::f64::consts::PI / 10.0*/),
            ellipse_from_stheta([ 2.0,5.0], [0.0,0.0], 0.0/*std::f64::consts::PI / 10.0*/),
            ellipse_from_stheta([ 1.0,4.0], [-4.0,-2.0], 6.0/*std::f64::consts::PI / 10.0*/),

            //Ellipsoid::new(&[[1.2576, -0.3873], [-0.3873,0.3467]], &[ 0.2722,  0.1969], 0.1831),
            //Ellipsoid::new(&[[1.4125, -2.1777], [-2.1777,6.7775]], &[-1.228,  -0.0521], 0.3295),
            //Ellipsoid::new(&[[1.7018,  0.8141], [ 0.8141,1.7538]], &[-0.4049,  1.5713], 0.2077),
            //Ellipsoid::new(&[[0.9742, -0.7202], [-0.7202,1.5444]], &[ 0.0265,  0.5623], 0.2362),
            //Ellipsoid::new(&[[0.6798, -0.1424], [-0.1424,0.6871]], &[-0.4301, -1.0157], 0.3284),
            //Ellipsoid::new(&[[0.1796, -0.1423], [-0.1423,2.6181]], &[-0.3286,  0.557 ], 0.4931) 
            ],
        bnd : Ellipsoid::new(&[[1.0,0.0],[0.0,1.0]],&[0.0, 0.0])
    };

    let (P,q) = minimal_bounding_ellipsoid(drawdata.ellipses.as_slice())?;
    // A² = P => A = sqrt(P)
    // Ab = q => A\q
    let s = det(&P).sqrt();
    let A = matscale(&matadd(&P,&[[s,0.0],[0.0,s]]), 1.0/(trace(&P) + 2.0*s).sqrt());
    let b = matmul(&inv(&A),&q);

    drawdata.bnd = Ellipsoid::new(&A,&b);

    let app = Application::builder()
        .application_id(APP_ID)
        .build();
    
    app.connect_activate(move | app : &Application | build_ui(app,&drawdata));
    let _r = app.run_with_args::<&str>(&[]);
    Ok(())
}












fn build_ui(app : &Application, drawdata : & DrawData) {
    let darea = DrawingArea::builder()
        .width_request(800) 
        .height_request(800)
        .build();
    {
        let drawdata = drawdata.clone();
        darea.set_draw_func(move |widget,context,w,h| redraw_window(widget,context,w,h,&drawdata));
    }

    let window = ApplicationWindow::builder()
        .application(app)
        .title("Whalesong")
        .child(&darea)
        .build();
    window.present();
}

#[allow(non_snake_case)]
fn context_ellipsis(context : &Context, A : &[[f64;2];2], b : &[f64;2]) {
    //let Ainv = inv(A);

    let old_mx = context.matrix();
    context.translate(-b[0],-b[1]);
    context.transform(cairo::Matrix::new(A[0][0],A[0][1],A[1][0],A[1][1],0.0,0.0));

    println!("A = {:?}, b = {:?}",A,b);
    //context.arc(-b[0],-b[1],1.0,0.0,std::f64::consts::PI*2.0);
    context.arc(0.0,0.0,1.0,0.0,std::f64::consts::PI*2.0);
    context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
    _ = context.stroke();
    context.set_matrix(old_mx);
}

#[allow(non_snake_case)]
fn redraw_window(_widget : &DrawingArea, context : &Context, w : i32, h : i32, data : &DrawData) {
    context.set_source_rgb(1.0, 1.0, 1.0);
    _ = context.paint();

    let w : f64 = w.into();
    let h : f64 = h.into();
    let s = w.min(h)/50.0;
    
    context.translate(300.0,300.0);
    context.scale(s,s);

    context.set_source_rgb(1.0, 0.0, 0.0);
    {
        let (A,b) = data.bnd.get_Pq();        
        //let Ainv = inv(&data.P);
        let Z = inv(&A);
        let w = [ -Z[0][0] * b[0] - Z[0][1] * b[1], 
                  -Z[1][0] * b[0] - Z[1][1] * b[1] ];

        println!("A,b = ({:?},{:?}) -> ({:?},{:?})",A,b,Z,w);

        context_ellipsis(context,&Z,&w);
    }
    //let old_mx = context.matrix();
    //context.scale(s,s);
    //context.translate(1.0,1.0);
    //context.translate(-b[0],-b[1]);
    //context.transform(cairo::Matrix::new(Ainv[0][0],Ainv[0][1],Ainv[1][0],Ainv[1][1],0.0,0.0));

    //context.arc(-b[0],-b[1],1.0,0.0,std::f64::consts::PI*2.0);
    //context.arc(0.0,0.0,1.0,0.0,std::f64::consts::PI*2.0);
    //context.set_matrix(old_mx);
    //_ = context.stroke();

    context.set_source_rgb(0.0, 0.0, 0.0);

    if true {
        for e in data.ellipses.iter() {
            let (A,b) = e.get_Pq();
            let Z = inv(&A);
            let w = [ -Z[0][0] * b[0] - Z[0][1] * b[1], 
                      -Z[1][0] * b[0] - Z[1][1] * b[1] ];

            println!("A,b = ({:?},{:?}) -> ({:?},{:?})",A,b,Z,w);
            context_ellipsis(context,&Z,&w);
            
            //let old_mx = context.matrix();            
            //context.transform(cairo::Matrix::new(Ainv[0][0],Ainv[0][1],Ainv[1][0],Ainv[1][1],0.0,0.0));
            //context.scale(0.1,0.1);
            ////context.arc((1.0-b[0])*s, (1.0-b[1])*s,s,0.0,std::f64::consts::PI*2.0);
            //context.arc((2.0-b[0])*s, (2.0-b[1])*s,s,0.0,std::f64::consts::PI*2.0);
            //context.set_matrix(old_mx);
            //_ = context.stroke();
        }
    }
}
