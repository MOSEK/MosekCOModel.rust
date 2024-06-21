extern crate glam;
extern crate mosekmodel;
extern crate bevy;

use bevy::prelude::*;
use bevy::reflect::Reflect;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, SystemTime};
use ellipsoids::Ellipsoid;
use glam::{DMat2,DVec2};
use itertools::izip;
use mosekmodel::{unbounded, Model};



const APP_ID : &str = "com.mosek.lowner-john-3d";
const SPEED_SCALE : f64 = 0.1;

struct RotatingSphere<const N : usize> {
    center : [f64;N],
    radius : [f64;N],
    global_axis : [f64;N],
    global_rotation_speed : f64,
    local_axis : [f64;N],
    local_rotation_speed : f64,
}

#[allow(non_snake_case)]
#[derive(Default,Reflect,GizmoConfigGroup)]
struct DrawData {

    radius : Vec<[f64;3]>,
    center : Vec<[f64;3]>,
    speed  : Vec<[f64;2]>,

    // Fixed ellipsoids
    Abs : Vec<([f64;4],[f64;3])>,
    // Bounding ellipsoid as { x : || Px+q || < 1 } 
    Pc : Option<([f64;4],[f64;3])>,
    Qd : Option<([f64;4],[f64;3])>
}


fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}


/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // circular base
    commands.spawn(PbrBundle {
        mesh: meshes.add(Circle::new(4.0)),
        material: materials.add(Color::WHITE),
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        ..default()
    });
    // cube
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        material: materials.add(Color::rgb_u8(124, 144, 255)),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..default()
    });

    // cube
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        material: materials.add(Color::rgb_u8(124, 144, 255)),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..default()
    });

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });
    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}











//     pub fn main() {
//         let drawdata = DrawData{
//             radius : vec![[0.2,0.15],[0.3,0.2],[0.4, 0.2]],
//             center : vec![[0.2,0.2],[-0.2,0.1],[0.2,-0.2]],
//             speed  : vec![[0.1,0.3],[-0.3,0.5],[0.4,-0.3]],
//     
//             ..default()
//         };
//     
//         App::new()
//             .add_plugins(DefaultPlugins)
//             .init_gizmo_group::<DrawData>()
//             .add_systems(Startup, setup)
//             .add_systems(Update, ( draw, update ))
//             .run()
//             ;
//     
//         println!("Main loop exit!");
//     }
//     
//     
//     fn setup(
//         mut commands: Commands,
//         mut meshes: ResMut<Assets<Mesh>>,
//         mut materials: ResMut<Assets<StandardMaterial>>)
//     {
//     }
//     
//     fn draw(
//         mut gizmos: Gizmos,
//         mut my_gizmos: Gizmos<DrawData>,
//         time: Res<Time>) 
//     {
//     }
//     fn update(
//         mut config_store: ResMut<GizmoConfigStore>,
//         keyboard: Res<ButtonInput<KeyCode>>,
//         time: Res<Time>)
//     {
//     }

//#[allow(non_snake_case)]
//fn build_ui(app   : &Application,
//            ddata : &DrawData)
//{    
//    // tx Send info from solver to GUI
//    // rtx Send commands from GUI to solver
//    let data = Rc::new(RefCell::new(ddata.clone()));
//    
//    let darea = GLArea::builder()
//        .width_request(800) 
//        .height_request(800)
//        .build();
//
//    // Redraw callback
//    {
//        let data = data.clone();
//        darea.connect_render(move |widget,context| redraw_window(widget,context,(&data).borrow()) );
//        //darea.set_draw_func(move |widget,context,w,h| redraw_window(widget,context,w,h,&data.borrow()));
//    }
//
//    let window = ApplicationWindow::builder()
//        .application(app)
//        .title("Hello Löwner-John")
//        .child(&darea)
//        .build();
//    
//    { // Time callback
//        let data = data.clone();
//        let darea = darea.clone();
//        glib::source::timeout_add_local(
//            Duration::from_millis(10), 
//            move || {
//                let mut data = data.borrow_mut();
//                let dt = 0.001 * (SystemTime::now().duration_since(data.t0).unwrap().as_millis() as f64);
//
//                data.Abs = izip!(data.radius.iter(),data.center.iter(),data.speed.iter())
//                    .map(|(&r,&c,&v)| {
//                        let theta_g = (2.0 * std::f64::consts::PI * v[0] * dt * SPEED_SCALE) % (2.0 * std::f64::consts::PI);
//                        let theta_l = (2.0 * std::f64::consts::PI * v[1] * dt * SPEED_SCALE) % (2.0 * std::f64::consts::PI);
//
//                        let (cost,sint) = ((theta_l/2.0).cos() , (theta_l/2.0).sin());
//                        let A = [ cost.powi(2)*r[0]+sint.powi(2)*r[1], cost*sint*(r[1]-r[0]),
//                                  cost*sint*(r[1]-r[0]), sint.powi(2) * r[0] + cost.powi(2) * r[1] ];                            
//                        let b = [ theta_g.cos()*c[0] - theta_g.sin()*c[1],
//                                  theta_g.sin()*c[0] + theta_g.cos()*c[1]];
//                        (A,b)
//                    }).collect();
//
//                      
//                {
//                    // outer ellipsoid
//                    let mut m = Model::new(None);
//                    let t = m.variable(None, unbounded());
//                    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
//                    let q = m.variable(None, unbounded().with_shape(&[2]));
//  
//                    m.objective(None, mosekmodel::Sense::Maximize, &t);
//                   
//                    for (A,b) in data.Abs.iter() {
//                        let A = DMat2::from_cols_array(A).inverse();
//                        let b = A.mul_vec2(DVec2{x:b[0], y:b[1]}).to_array();
//
//                        let e : Ellipsoid<2> = ellipsoids::Ellipsoid::from_arrays(&A.to_cols_array(), &[-b[0],-b[1]]);
//
//                        ellipsoids::ellipsoid_contains(&mut m,&p,&q,&e);
//                    }
//
//                    m.solve();
//  
//                    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
//                                                  m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
//                        
//                        // A² = P => A = sqrt(P)
//                        // Ab = q => A\q
//                        let s = (psol[0]*psol[3]-psol[1]*psol[2]).sqrt();
//
//                        let A = DMat2::from_cols_array(&[psol[0],psol[1],psol[2],psol[3]]).add_mat2(&DMat2::from_cols_array(&[s,0.0,0.0,s])).mul_scalar(1.0/(psol[0]+psol[3] + 2.0*s).sqrt());
//                        let b = A.inverse().mul_vec2(DVec2::from_array([qsol[0],qsol[1]]));
//
//                        data.Pc = Some((A.to_cols_array(),b.to_array()));
//                    }
//                    else {
//                        data.Pc = None;
//                    }
//                }
//
//
//                {
//                    // inner ellipsoid
//                    let mut m = Model::new(None);
//
//                    let t = m.variable(None, unbounded());
//                    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 2);
//                    let q = m.variable(None, unbounded().with_shape(&[2]));
//
//                    m.objective(None, mosekmodel::Sense::Maximize, &t);
//
//                    for (A,b) in data.Abs.iter() {
//                        let A = DMat2::from_cols_array(A).inverse();
//                        let b = A.mul_vec2(DVec2{x:b[0], y:b[1]}).to_array();
//
//                        let e : Ellipsoid<2> = ellipsoids::Ellipsoid::from_arrays(&A.to_cols_array(), &[-b[0],-b[1]]);
//
//                        ellipsoids::ellipsoid_contained(&mut m,&p,&q,&e);
//                    }
//
//                    m.solve();
//
//                    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
//                                                  m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
//                        let A = DMat2::from_cols_array(&[psol[0],psol[1],psol[2],psol[3]]).inverse();
//                        let b = A.mul_vec2(DVec2::from_array([qsol[0],qsol[1]])).to_array();
//
//                        data.Qd = Some((A.to_cols_array(),[-b[0],-b[1]]));
//                    }
//                    else {
//                        data.Qd = None;
//                    }
//                }
//
//                darea.queue_draw();
//                ControlFlow::Continue
//            });
//    }    
//
//    window.present();
//}
//
//#[allow(non_snake_case)]
//fn redraw_window(_widget : &GLArea, ctx : &GLContext, dd : &DrawData) -> Propagation {
//    gl33::
//    Propagation::Proceed
//}
