extern crate glam;
extern crate mosekmodel;
extern crate bevy;
extern crate rand;
extern crate ellipsoids;

use bevy::math::{DMat3, DVec3};
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::render::mesh::PrimitiveTopology;
use bevy::render::render_asset::RenderAssetUsages;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, SystemTime};
use ellipsoids::Ellipsoid;
use glam::{DMat2,DVec2};
use itertools::izip;
use mosekmodel::{unbounded, Model};

use std::f32::consts::PI;

const SPEED_SCALE : f64 = 0.1;

#[allow(non_snake_case)]
#[derive(Default,Component)]
struct MyObject {
    center : Vec3,
    scale  : Vec3,

    global_axis : Vec3,
    global_rps  : f32,

    local_axis  : Vec3,
    local_rps   : f32,
}

#[derive(Default,Component)]
struct BoundingEllipsoid;


impl MyObject {
    pub fn random(center_sphere : f32, local_rps_bracket : (f32,f32), global_rps_bracket : (f32,f32), scale_bracket : (f32,f32)) -> MyObject {
        let g = ((global_rps_bracket.1-global_rps_bracket.0).abs(),global_rps_bracket.0.min(global_rps_bracket.1));
        let l = ((local_rps_bracket.1-local_rps_bracket.0).abs(),  local_rps_bracket.0.min(local_rps_bracket.1));
        let s = ((scale_bracket.0-scale_bracket.1).abs(),          scale_bracket.0.min(scale_bracket.1).max(0.1));

        let grps = rand::random::<f32>() * g.0 + g.1;
        let lrps = rand::random::<f32>() * l.0 + l.1;
        MyObject{
            center : center_sphere.max(0.0) * rand_vec3(),
            scale : Vec3::new(s.1+rand::random::<f32>()*s.0,
                              s.1+rand::random::<f32>()*s.0,
                              s.1+rand::random::<f32>()*s.0),
            global_axis : rand_vec3(),
            global_rps : grps,
            local_axis : rand_vec3(),
            local_rps : lrps
        }
    }
}

#[derive(Default,Component)]
struct CameraTransform{
    rps : f32
}

fn rand_vec3() -> Vec3 {
    while true {
        let r = Vec3::new(rand::random::<f32>()*2.0-1.0,
                          rand::random::<f32>()*2.0-1.0,
                          rand::random::<f32>()*2.0-1.0);
        if r.length() <= 1.0 {
            return r.normalize();
        }
    }
    Vec3::X
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .add_systems(Update, update_camera)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // circular base
    /*
    commands.spawn(PbrBundle {
        mesh: meshes.add(Circle::new(4.0)),
        material: materials.add(Color::WHITE),
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
        ..default()
    });
    */
    // cube
    
    for _ in 0..5 {
        commands.spawn((PbrBundle {
            mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
            material: materials.add(Color::rgb_u8(rand::random::<u8>()/2,rand::random::<u8>()/2,rand::random::<u8>()/2)),
            ..default()},
            MyObject::random(2.0, (-1.0,1.0),(-1.0,1.0),(0.5,2.5))
            ));
    }


    let boundmat = materials.add(StandardMaterial{
        base_color : Color::rgba_u8(192,192,255,64),
        //metallic : 0.9,
        //reflectance : 0.9, 
        ..default()
    });
    commands.spawn((
        PbrBundle{
            mesh: meshes.add(Sphere::new(1.0)),
            material: materials.add(Color::rgba_u8(192,192,255,64)),
            ..default() },
        BoundingEllipsoid{}
        ));

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
    commands.spawn((Camera3dBundle {
        transform: Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default() },
        CameraTransform{ rps:1.0/30.0 }));
}



fn update_camera(time: Res<Time>, mut query: Query<(&mut Transform,&CameraTransform)>) {
    let t = time.elapsed_seconds();
    for (mut transform, c) in &mut query {
        let camloc = Quat::from_rotation_y(2.0*PI*c.rps*t).mul_vec3(Vec3::new(-2.5, 4.5, 9.0)) ;
        let tf = Transform::from_xyz(camloc.x,camloc.y,camloc.z).looking_at(Vec3::ZERO, Vec3::Y);
        transform.clone_from(&tf);
    }
    
}
fn update(time: Res<Time>, 
          mut query: Query<(&mut Transform, &MyObject)>, 
          mut qbound: Query<(&mut Transform, &BoundingEllipsoid), Without<MyObject>>, 
          mut gizmos: Gizmos) {
    //let t = (time.elapsed_seconds().sin() + 1.) / 2.;
    let t = time.elapsed_seconds();
/*
    gizmos.line(Vec3::new(0.0,0.0,0.0), 
                Vec3::new(5.0,0.0,0.0),
                Color::rgb_u8(255,255,255));
    gizmos.line(Vec3::new(0.0,0.0,0.0), 
                Vec3::new(0.0,5.0,0.0),
                Color::rgb_u8(255,255,255));
    gizmos.line(Vec3::new(0.0,0.0,0.0), 
                Vec3::new(0.0,0.0,5.0),
                Color::rgb_u8(255,255,255));
*/
    println!("transform :");

    let cube_points = [ Vec3::new(-0.5,-0.5,-0.5),
                        Vec3::new( 0.5,-0.5, 0.5),
                        Vec3::new( 0.5, 0.5,-0.5),
                        Vec3::new(-0.5, 0.5, 0.5),
                        Vec3::new(-0.5,-0.5,-0.5),
                        Vec3::new( 0.5,-0.5, 0.5),
                        Vec3::new( 0.5, 0.5,-0.5),
                        Vec3::new(-0.5, 0.5, 0.5) ];
    let mut points = Vec::new();
    for (mut transform, e) in &mut query {
        transform.scale = e.scale;
        transform.rotation = Quat::from_axis_angle(e.local_axis, e.local_rps * t);
        println!("\tcenter : {}, scale : {}",e.center,e.scale);

        transform.translation = Quat::from_axis_angle(e.global_axis, (e.global_rps*t) % (2.0*PI)).mul_vec3(e.center);
        for p in cube_points.iter() {
            let p = transform.rotation.mul_vec3(Mat3::from_diagonal(e.scale).mul_vec3(*p)) + transform.translation;
            points.push([p.x as f64,p.y as f64,p.z as f64]);
        }
    }

    {
        // outer ellipsoid
        let mut m = Model::new(None);
        let t = m.variable(None, unbounded());
        let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 3);
        let q = m.variable(None, unbounded().with_shape(&[3]));
  
        m.objective(None, mosekmodel::Sense::Maximize, &t);
       


        ellipsoids::ellipsoid_contains_points(& mut m, &p, &q, points.as_slice());

        m.solve();
  
        if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
                                      m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
            
            let A = DMat3::from_cols_array(&[psol[0],psol[1],psol[2],psol[3],psol[4],psol[5],psol[6],psol[7],psol[8]]).inverse();
            let b = A.mul_vec3(DVec3::new(qsol[0],qsol[1],qsol[2]));
            
            for (mut transform,_) in & mut qbound {
                let mut tf = Transform::from_matrix(Mat4::from_mat3(A.as_mat3()));
                tf.translation = b.as_vec3();
                transform.clone_from(&tf);
            }
        }
        else {
            for (mut transform,_) in & mut qbound {
                transform.scale = Vec3::ZERO;
            }
        }
    }


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
