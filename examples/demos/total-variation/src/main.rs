extern crate mosekcomodel; extern crate rand;
extern crate rand_distr;
extern crate image;

use std::cell::RefCell;
use std::rc::Rc;
use gtk::gdk::Texture;
use gtk::gdk_pixbuf::Pixbuf;
use gtk::glib::Bytes;
use gtk::{prelude::*, Image, Orientation};
use gtk::{Application, ApplicationWindow};
use itertools::{iproduct, izip};
use mosekcomodel::*;
use rand::*;

use image::{Rgb,ImageBuffer,ImageReader};


const APP_ID : &str = "com.mosek.total-variation";



fn main() { 
    let mut args = std::env::args(); args.next();
    let mut noise = 25.5/255.0;
    let mut sigma = 0.0004;
    let mut filename = None;
    let mut default_dim = 200;

    while let Some(v) = args.next() {
        match v.as_str() {
            "--help"|"-h" => {
                println!("total-variation [ OPTIONS ] [ --help | -h ] [ FILENAME ]");
                println!("  --sigma|-s VAL (default: 0.0004), average correction per cell, most useful in the range 0.0001 to 0.0006.");
                println!("  --noise|-n VAL (default: 0.1, valid range : [0.0;1.0]), noise added to generated image");
                println!("  --dim|-d N Dimension of generated image, this should probably not be significantly larger than 200");
                println!("  FILENAME Name of the file to use instead. Dimensions should probably not be significantly larger than 200x200");
                return;
            },
            "--noise"|"-n" => 
                if let Some(s) = args.next() {
                    noise = s.parse::<f32>().expect("First argument must be a float");
                },
            "--dim"|"-d" => 
                if let Some(s) = args.next() {
                    default_dim = s.parse::<u32>().expect("First argument must be an integer");
                },
            "--sigma"|"-s" =>
                if let Some(s) = args.next() {
                    sigma = s.parse::<f64>().expect("First argument must be a float");
                },
            s => {
                filename = Some(s.to_string());
            }
        }
    }

    let (img,noisy_img) = if let Some(filename) = filename {
        let img = get_image(Some(filename),default_dim).map_err(|e| format!("Failed to read image: {:?}",e)).unwrap();
        (None,img)
    }
    else {
        let img = get_image(None,default_dim).map_err(|e| format!("Failed to read image: {:?}",e)).unwrap();

        let mut r = rand::rng();
        // create noisy image
        let noisy_img : ImageBuffer<Rgb<u8>,Vec<u8>> =
            ImageBuffer::from_raw(
                default_dim,
                default_dim, 
                img.pixels()
                    .flat_map(|rgb| rgb.0.iter())
                    .map(|&v| ((v as f32 / 255.0 + noise * (r.random::<f32>()-0.5)).max(0.0).min(1.0)*255.0) as u8)
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



struct Data {
    img             : Option<ImageBuffer<Rgb<u8>,Vec<u8>>>,
    noisy_img       : ImageBuffer<Rgb<u8>,Vec<u8>>,
    width : u32,
    height : u32,
    model_red       : Model,
    ucore_red       : Variable<2>,
    model_green     : Model, 
    ucore_green     : Variable<2>, 
    model_blue      : Model,  
    ucore_blue      : Variable<2>,  

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

        let (model_red,   ucore_red)   = total_var(sigma, &noisy_image_red);
        let (model_green, ucore_green) = total_var(sigma, &noisy_image_green);
        let (model_blue,  ucore_blue)  = total_var(sigma, &noisy_image_blue);

        Data{
            img,
            noisy_img,
            width,height,
            model_red,
            ucore_red,
            model_green,
            ucore_green,
            model_blue,
            ucore_blue,

            sol_red : None,
            sol_blue : None,
            sol_green : None
        }
    }
    fn solve(&mut self) {
        self.model_red.solve();
        self.model_red.write_problem("red.ptf");
        self.sol_red = Some(self.model_red.primal_solution(SolutionType::Default,&self.ucore_red).unwrap());

        self.model_blue.write_problem("blue.ptf");
        self.model_blue.solve();
        self.sol_blue = Some(self.model_blue.primal_solution(SolutionType::Default,&self.ucore_blue).unwrap());
        
        self.model_green.write_problem("green.ptf");
        self.model_green.solve();
        self.sol_green = Some(self.model_green.primal_solution(SolutionType::Default,&self.ucore_green).unwrap());
    }
}

fn get_image(filename : Option<String>,default_dim : u32) -> Result<ImageBuffer<Rgb<u8>,Vec<u8>>,String> {
    if let Some(filename) = filename {
        Ok(ImageReader::open(filename)
            .map_err(|err| err.to_string())?
            .decode().map_err(|err| err.to_string())?.to_rgb8())
    }
    else {
        let rgbdata = iproduct!(0..default_dim, 0..default_dim,0..3)
            .map(|(i,j,k)| {
                let ii = i as f64 / default_dim as f64;
                let jj = j as f64 / default_dim as f64;

                match k {
                    0 => ((1.0-ii)*(1.0-jj)*255.0) as u8,
                    1 => ((1.0-ii)*jj*255.0) as u8, 
                    2 => (ii*jj*255.0) as u8,
                    _ => 0
                }
        }).collect();
        ImageBuffer::from_raw(default_dim, default_dim, rgbdata).ok_or(format!("Failed to create image"))
    }
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

    let window = ApplicationWindow::builder()
        .application(app)
        .title("Total variation")
        .child(&hbox)
        .build();
    
    window.present();
}




#[allow(non_snake_case)]
fn total_var(sigma : f64, f : &NDArray<2>) -> (Model,Variable<2>) {
    let mut M = Model::new(Some("TV"));
    M.set_log_handler(|msg| print!("{}",msg));
    let n = f.height();
    let m = f.width();

    let u = M.ranged_variable(Some("u"), in_range(0.0, 1.0).with_shape(&[n+1,m+1])).0;
    let t = M.variable(Some("t"), unbounded().with_shape(&[n,m]));

    // In this example we define sigma and the input image f as parameters
    // to demonstrate how to solve the same model with many data variants.
    // Of course they could simply be passed as ordinary arrays if that is not needed.

    let ucore  = u.index((..n,..m));

    M.constraint(Some("Delta"),
                 stack![2; t.reshape(&[n,m,1]),
                           u.index((1..,..m)).sub(&ucore).reshape(&[n,m,1]),
                           u.index((..n,1..)).sub(&ucore).reshape(&[n,m,1])],
                 in_quadratic_cone());

    M.constraint(Some("TotalVar"), 
                 ((n*m) as f64 * sigma).into_expr().flatten().vstack(ucore.sub(f).flatten()),
                 in_quadratic_cone());

    M.objective(None, Sense::Minimize, t.sum());

    (M,u.index([0..n,0..m]))
}

