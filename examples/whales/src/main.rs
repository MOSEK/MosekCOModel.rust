extern crate mosekmodel;
extern crate ellipsoids;
extern crate cairo;
mod whales;
mod utils2d;
//mod matrix;
//mod ellipsoids;

use ellipsoids::*;
use whales::{minimal_bounding_ellipsoid,maximal_contained_ellipsoid};
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
    outer : Ellipsoid<2>,
    inner : Ellipsoid<2>,
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

            Ellipsoid::new(&[[0.21957739348193858, 0.12360679774997899], [0.12360679774997899,0.9804226065180615]],&[0.0, 0.0]),
            Ellipsoid::new(&[[0.21957739348193858, -0.12360679774997899],[ -0.12360679774997899,0.9804226065180615]],&[0.5627615847138562, -1.2276362020180194]),


            //Ellipsoid::new(&[[2.3246108597249653, -0.5023002219069703],  [-0.5023002219069703, 1.143239126869145]], &[ 1.6262243355626635,   2.3572628105100137]),
            //Ellipsoid::new(&[[0.7493116238095875, -0.4519632176898713],  [-0.4519632176898713, 1.8627720007697564]],&[-0.7741774083306816,  -0.19900209213742667]),
            //Ellipsoid::new(&[[0.8582679328352112,  0.21661818880463501], [ 0.21661818880463501,0.8721042500171801]],&[-0.30881539733571506,  0.6395981801584628]),
            //Ellipsoid::new(&[[2.9440718650596165, -1.0778871720849044],  [-1.0778871720849044, 3.797461570034363]], &[ 2.2609091903635528,   5.387366401851428]),
            //Ellipsoid::new(&[[0.6104500401008942, -0.06447520306755025], [-0.06447520306755025,0.6137552998087111]],&[-0.36695528785675785, -0.7214779986292089]),
            //Ellipsoid::new(&[[1.1044036516422946, -0.18480500741119338], [-0.18480500741119338,4.271283557279645]], &[-5.123066175367,       2.1838724317503617]),
            ],
        outer : Ellipsoid::new(&[[1.0,0.0],[0.0,1.0]],&[0.0, 0.0]),
        inner : Ellipsoid::new(&[[1.0,0.0],[0.0,1.0]],&[0.0, 0.0]),
    };

    let (P,q) = minimal_bounding_ellipsoid(drawdata.ellipses.as_slice())?;
    // A² = P => A = sqrt(P)
    // Ab = q => A\q
    let s = det(&P).sqrt();
    let A = matscale(&matadd(&P,&[[s,0.0],[0.0,s]]), 1.0/(trace(&P) + 2.0*s).sqrt());
    let b = matmul(&inv(&A),&q);

    

    // u = Zx+w, ||u||<1 
    // Z^{-1}(u-w), ||u||<1
    let (Z,w) = maximal_contained_ellipsoid(drawdata.ellipses.as_slice())?;
    let Zinv = inv(&Z);
    let winv = matmul(&Zinv,&w);
    let winv = [ -winv[0], -winv[1]];
        
    drawdata.outer = Ellipsoid::new(&A,&b);
    drawdata.inner  = Ellipsoid::new(&Zinv,&winv);

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
    let s = w.min(h) / 20.0;
    
    context.translate(400.0,400.0);
    context.scale(s,s);

    context.set_source_rgb(1.0, 0.0, 0.0);
    {
        let (A,b) = data.outer.get_Pq();        
        //let Ainv = inv(&data.P);
        let Z = inv(&A);
        let w = [ -Z[0][0] * b[0] - Z[0][1] * b[1], 
                  -Z[1][0] * b[0] - Z[1][1] * b[1] ];

        context_ellipsis(context,&Z,&w);
    }
    context.set_source_rgb(0.0, 1.0, 0.0);
    {
        let (A,b) = data.inner.get_Pq();        
        //let Ainv = inv(&data.P);
        let Z = inv(&A);
        let w = [ -Z[0][0] * b[0] - Z[0][1] * b[1], 
                  -Z[1][0] * b[0] - Z[1][1] * b[1] ];

        context_ellipsis(context,&Z,&w);
    }
    context.set_source_rgb(0.0, 0.0, 0.0);

    if true {
        for e in data.ellipses.iter() {
            let (A,b) = e.get_Pq();
            let Z = inv(&A);
            let w = [ -Z[0][0] * b[0] - Z[0][1] * b[1], 
                      -Z[1][0] * b[0] - Z[1][1] * b[1] ];

            context_ellipsis(context,&Z,&w);
        }
    }
}
