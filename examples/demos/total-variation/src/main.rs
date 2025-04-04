extern crate mosekcomodel; extern crate rand;
extern crate rand_distr;
extern crate image;

use std::cell::RefCell;
use std::rc::Rc;
use gtk::gdk::Texture;
use gtk::gdk_pixbuf::Pixbuf;
use gtk::glib::Bytes;
use gtk::{prelude::*, Image, Orientation};
use gtk::{Application, DrawingArea, ApplicationWindow};
use itertools::{iproduct, izip};
use mosekcomodel::*;
use rand::*;
use rand_distr::*;

use image::{Rgb,ImageBuffer,ImageReader};


const APP_ID : &str = "com.mosek.total-variation";
const DEFAULT_WIDTH  : u32 = 200;
const DEFAULT_HEIGHT : u32 = 200;
const NOISE_SCALE : f32 = 0.3;




struct Data {
    img             : Option<ImageBuffer<Rgb<u8>,Vec<u8>>>,
    noisy_img       : ImageBuffer<Rgb<u8>,Vec<u8>>,
    width : u32,
    height : u32,
    model_red       : Model,
    ucore_red       : Variable<2>,
    sigma_con_red   : Constraint<0>,
    model_green     : Model, 
    ucore_green     : Variable<2>, 
    sigma_con_green : Constraint<0>,
    model_blue      : Model,  
    ucore_blue      : Variable<2>,  
    sigma_con_blue  : Constraint<0>,

    sol_red   : Option<Vec<f64>>,
    sol_green : Option<Vec<f64>>,
    sol_blue  : Option<Vec<f64>>,
}

impl Data {
    fn new(img : Option<ImageBuffer<Rgb<u8>,Vec<u8>>>, 
           noisy_img : ImageBuffer<Rgb<u8>,Vec<u8>>, 
           sigma : f64) -> Data 
    {
        let width  = noisy_img.width();
        let height = noisy_img.height();

        let noisy_image_red   = NDArray::new([height as usize,width as usize], None, noisy_img.pixels().map(|rgb| rgb.0[0] as f64 / 255.0).collect()).unwrap();
        let noisy_image_green = NDArray::new([height as usize,width as usize], None, noisy_img.pixels().map(|rgb| rgb.0[1] as f64 / 255.0).collect()).unwrap();
        let noisy_image_blue  = NDArray::new([height as usize,width as usize], None, noisy_img.pixels().map(|rgb| rgb.0[2] as f64 / 255.0).collect()).unwrap();

        let (model_red,   ucore_red,   sigma_con_red)   = total_var(sigma, &noisy_image_red);
        let (model_green, ucore_green, sigma_con_green) = total_var(sigma, &noisy_image_green);
        let (model_blue,  ucore_blue,  sigma_con_blue)  = total_var(sigma, &noisy_image_blue);

        Data{
            img,
            noisy_img,
            width,height,
            model_red,
            ucore_red,
            sigma_con_red,
            model_green,
            ucore_green,
            sigma_con_green,
            model_blue,
            ucore_blue,
            sigma_con_blue,

            sol_red : None,
            sol_blue : None,
            sol_green : None
        }
    }
    fn solve(&mut self) {
        self.model_red.solve();
        self.sol_red = Some(self.model_red.primal_solution(SolutionType::Default,&self.ucore_red).unwrap());

        self.model_blue.solve();
        self.sol_blue = Some(self.model_blue.primal_solution(SolutionType::Default,&self.ucore_blue).unwrap());
        
        self.model_green.solve();
        self.sol_green = Some(self.model_green.primal_solution(SolutionType::Default,&self.ucore_green).unwrap());
    }
}

fn get_image(filename : Option<String>) -> Result<ImageBuffer<Rgb<u8>,Vec<u8>>,String> {
    if let Some(filename) = filename {
        Ok(ImageReader::open(filename)
            .map_err(|err| err.to_string())?
            .decode().map_err(|err| err.to_string())?.to_rgb8())
    }
    else {
        let rgbdata = iproduct!(0..DEFAULT_HEIGHT, 0..DEFAULT_WIDTH,0..3)
            .map(|(i,j,k)| {
                let ii = i as f64 / DEFAULT_HEIGHT as f64;
                let jj = j as f64 / DEFAULT_WIDTH as f64;

                match k {
                    0 => ((1.0-ii)*(1.0-jj)*255.0) as u8,
                    1 => ((1.0-ii)*jj*255.0) as u8, 
                    2 => (ii*jj*255.0) as u8,
                    _ => 0
                }
        }).collect();
        ImageBuffer::from_raw(DEFAULT_WIDTH, DEFAULT_HEIGHT, rgbdata).ok_or(format!("Failed to create image"))
    }
}

fn main() { 
    let mut args = std::env::args(); args.next();   
    let sigma = args.next().expect("Missing argument: sigma").parse::<f64>().expect("First argument must be a float");

    let (img,noisy_img) = if let Some(filename) = args.next() {
        let img = get_image(Some(filename)).map_err(|e| format!("Failed to read image: {:?}",e)).unwrap();
        (None,img)
    }
    else {
        let img = get_image(None).map_err(|e| format!("Failed to read image: {:?}",e)).unwrap();

        let mut r = rand::rng();
        // create noisy image
        let noisy_img : ImageBuffer<Rgb<u8>,Vec<u8>> =
            ImageBuffer::from_raw(
                DEFAULT_WIDTH,
                DEFAULT_HEIGHT, 
                img.pixels()
                    .flat_map(|rgb| rgb.0.iter())
                    .map(|&v| ((v as f32 / 255.0 + NOISE_SCALE * (r.random::<f32>()-0.5)).max(0.0).min(1.0)*255.0) as u8)
                    .collect()).unwrap();
        (Some(img),noisy_img)
    };

    let data = Rc::new(RefCell::new(Data::new(img,noisy_img,sigma)));
    data.borrow_mut().solve();

    let app = Application::builder()
        .application_id(APP_ID)
        .build();
    app.connect_activate(move | app : &Application | build_ui(app,data.clone()));

    let r = app.run_with_args::<&str>(&[]);
    println!("Exit: {:?}",r);
}

fn bracket(f:f64,l:f64,u:f64) -> f64 { if f < l { l } else if f > u { u } else { f } } 
fn build_ui(app  : &Application,
            data : Rc<RefCell<Data>>)
{
    // tx Send info from solver to GUI
    // rtx Send commands from GUI to solver
    let hbox = gtk::Box::builder()
        .orientation(Orientation::Horizontal)
        .build();
    if let Some(img) = data.borrow().img.as_ref() {
        let imgarea = {
            let data = data.borrow();
            let rgbdata : Vec<u8> = img.pixels().flat_map(|rgb| rgb.0.iter().cloned()).collect();
            let rgbdata_bytes = Bytes::from(&rgbdata);
            let img_pixbuf = Pixbuf::from_bytes(&rgbdata_bytes, gtk::gdk_pixbuf::Colorspace::Rgb, false, 8, data.width as i32, data.height as i32, (data.width*3) as i32);
            let img_texture = Texture::for_pixbuf(&img_pixbuf);

            Image::builder()
                .width_request(data.width as i32) 
                .height_request(data.height as i32)
                .paintable(&img_texture)
                .build()
        };
        hbox.append(&imgarea);
    }

    let noisy_imgarea = {
        let data = data.borrow();
        let rgbdata : Vec<u8> = data.noisy_img.pixels().flat_map(|rgb| rgb.0.iter().cloned()).collect();
        let rgbdata_bytes = Bytes::from(&rgbdata);
        let img_pixbuf = Pixbuf::from_bytes(&rgbdata_bytes, gtk::gdk_pixbuf::Colorspace::Rgb, false, 8, data.width as i32, data.height as i32, (data.width*3) as i32);
        let img_texture = Texture::for_pixbuf(&img_pixbuf);

        Image::builder()
            .width_request(data.width as i32) 
            .height_request(data.height as i32)
            .paintable(&img_texture)
            .build()
    };


    let sol_imgarea = {

        let data = data.borrow();
        let rgbdata : Vec<u8> = 
            izip!(data.sol_red.as_ref().unwrap().iter(),data.sol_green.as_ref().unwrap().iter(),data.sol_blue.as_ref().unwrap().iter())
            .map(|(&r,&g,&b)| [ (bracket(r,0.0,1.0)*255.0) as u8, (bracket(g,0.0,1.0)*255.0) as u8, (bracket(b,0.0,1.0)*255.0) as u8])
            .flat_map(|rgb| rgb.into_iter())
            .collect();
        let rgbdata_bytes = Bytes::from(&rgbdata);
        let img_pixbuf = Pixbuf::from_bytes(&rgbdata_bytes, gtk::gdk_pixbuf::Colorspace::Rgb, false, 8, data.width as i32, data.height as i32, (data.width*3) as i32);
        let img_texture = Texture::for_pixbuf(&img_pixbuf);

        Image::builder()
            .width_request(data.width as i32) 
            .height_request(data.height as i32)
            .paintable(&img_texture)
            .build()
    };
    hbox.append(&noisy_imgarea);
    hbox.append(&sol_imgarea);

    // Redraw callback
    //{
    //    let data = data.clone();
    //    .set_draw_func(move |widget,context,w,h| redraw_window(widget,context,w,h,&data.borrow()));
    //}


    let window = ApplicationWindow::builder()
        .application(app)
        .title("Total variation")
        .child(&hbox)
        .build();
    
    window.present();
}




#[allow(non_snake_case)]
fn total_var(sigma : f64, f : &NDArray<2>) -> (Model,Variable<2>,Constraint<0>) {
    let mut M = Model::new(Some("TV"));
    M.set_log_handler(|msg| print!("{}",msg));
    let n = f.height();
    let m = f.width();

    let u = M.variable(Some("u"), nonnegative().with_shape(&[n+1,m+1]));
    _ = M.constraint(None, &u, less_than(1.0).with_shape(&[n+1,m+1]));
    let t = M.variable(Some("t"), unbounded().with_shape(&[n,m]));

    // In this example we define sigma and the input image f as parameters
    // to demonstrate how to solve the same model with many data variants.
    // Of course they could simply be passed as ordinary arrays if that is not needed.

    let ucore  = u.index([0..n,0..m]);
    let deltax = u.index([1..n+1,0..m]).sub(ucore.clone()).reshape(&[n,m,1]);
    let deltay = u.index([0..n,1..m+1]).sub(ucore.clone()).reshape(&[n,m,1]);

    M.constraint( Some("Delta"), stack![2; t.reshape(&[n,m,1]), deltax, deltay], in_quadratic_cones(&[n,m,3], 2));

    let c = M.constraint(Some("TotalVar"), 
                         sigma.into_expr().reshape(&[1,1])
                            .vstack(f.to_expr().sub(ucore).reshape(&[n*m,1]))
                            .flatten(),
                         in_quadratic_cone());

    M.objective( None, Sense::Minimize, t.sum());

    (M,u.index([0..n,0..m]),c.index([0..1]).reshape(&[]))
}

