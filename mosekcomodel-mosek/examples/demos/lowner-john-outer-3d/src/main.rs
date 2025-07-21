extern crate glam;
extern crate mosekcomodel;
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
use mosekcomodel::unbounded;
use mosekcomodel_mosek::Model;

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

    // cubes providing a point cloud
    for _ in 0..8 {
        commands.spawn((PbrBundle {
            mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
            material: materials.add(Color::rgb_u8(rand::random::<u8>()/2+128,rand::random::<u8>()/2+128,rand::random::<u8>()/2+128)),
            ..default()},
            MyObject::random(2.0, (-1.0,1.0),(-1.0,1.0),(0.5,2.5))
            ));
    }


    /*
    commands.spawn((
        PbrBundle{
            //mesh: meshes.add(Sphere::new(1.0)),
            mesh: meshes.add(Cuboid::new(2.0,2.0,2.0)),
            material: materials.add(Color::rgba_u8(192,192,255,64)),
            ..default() },
        BoundingEllipsoid{}
        ));
    */
    commands.spawn((
        PbrBundle{
            mesh: meshes.add(Sphere::new(1.0)),
            //mesh: meshes.add(Cuboid::new(2.0,2.0,2.0)),
            material: materials.add(Color::rgba_u8(255,255,255,32)),
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
#[allow(non_snake_case)]
fn update(time: Res<Time>, 
          mut query: Query<(&mut Transform, &MyObject)>, 
          mut qbound: Query<(&mut Transform, &BoundingEllipsoid), Without<MyObject>>, 
          mut gizmos: Gizmos) {
    //let t = (time.elapsed_seconds().sin() + 1.) / 2.;
    let t = time.elapsed_seconds();

    let cube_points = [ Vec3::new(-0.5,-0.5,-0.5),
                        Vec3::new( 0.5,-0.5,-0.5),
                        Vec3::new( 0.5, 0.5,-0.5),
                        Vec3::new(-0.5, 0.5,-0.5),
                        Vec3::new(-0.5,-0.5, 0.5),
                        Vec3::new( 0.5,-0.5, 0.5),
                        Vec3::new( 0.5, 0.5, 0.5),
                        Vec3::new(-0.5, 0.5, 0.5) ];
    let mut points = Vec::new();
    for (mut transform, e) in &mut query {
        transform.scale = e.scale*0.95;
        transform.rotation = Quat::from_axis_angle(e.local_axis, e.local_rps * t);
        transform.translation = Quat::from_axis_angle(e.global_axis, (e.global_rps*t) % (2.0*PI)).mul_vec3(e.center);
        for p in cube_points.iter() {
            let p = transform.rotation.mul_vec3(Mat3::from_diagonal(transform.scale).mul_vec3(*p)) + transform.translation;
            points.push([p.x as f64,p.y as f64,p.z as f64]);
        }

        //let pts : Vec<Vec3> = points.iter().map(|v| DVec3::new(v[0],v[1],v[2]).as_vec3()).collect();
        //gizmos.linestrip(pts,Color::rgb_u8(255,0,0));
    }

    {
        // outer ellipsoid
        let mut m = Model::new(None);
        let t = m.variable(None, unbounded());
        let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 3);
        let q = m.variable(None, unbounded().with_shape(&[3]));
  
        m.objective(None, mosekcomodel::Sense::Maximize, &t);

        ellipsoids::ellipsoid_contains_points(& mut m, &p, &q, points.as_slice());

        m.solve();
  
        if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekcomodel::SolutionType::Default,&p),
                                      m.primal_solution(mosekcomodel::SolutionType::Default,&q)) {
            
            let A = DMat3::from_cols_array(&[psol[0],psol[1],psol[2],psol[3],psol[4],psol[5],psol[6],psol[7],psol[8]]).inverse();
            let b = -A.mul_vec3(DVec3::new(qsol[0],qsol[1],qsol[2]));


            // NOTE: The Transform::from_mat4 method only works for a translated, scaled orthogonal
            // basis. The matrix A we got is a symmetric positive definite matrix. We need to
            // obtain a scaled orthogonal basis that maps the unit sphere into the same ellipsoid
            // as A:
            let (scale,rotation,translation) = linalg::axb_to_srt(&A,&b);
            let tf = Transform{scale:scale.as_vec3(),rotation:rotation.as_quat(),translation:translation.as_vec3()};

            for (mut transform,_) in & mut qbound {
                transform.clone_from(&tf);
            }

            let maxlen = points.iter().map(|p| Mat3::from_diagonal(tf.scale).inverse().mul_vec3(tf.rotation.inverse().mul_vec3(DVec3::new(p[0],p[1],p[2]).as_vec3()-tf.translation)).length()).fold(0.0,|m,l| if l > m { l } else { m });
            if maxlen > 1.000000001 {
                println!("Max violation: {:.2e}",maxlen-1.0);
            }

            //gizmos.linestrip(
            //    cube_points.iter().map(|p| 2.0 * tf.rotation.mul_vec3(Mat3::from_diagonal(tf.scale).mul_vec3(*p)) + tf.translation),
            //    Color::rgb_u8(0,255,0));
        }
        else {
            for (mut transform,_) in & mut qbound {
                transform.scale = Vec3::ZERO;
            }
        }
    }
}

mod linalg {
    use bevy::math::{DQuat,DMat3,DVec3};
    use mosek::syevd;

    pub fn axb_to_srt(A : &DMat3, b : &DVec3) -> (DVec3, DQuat, DVec3) {

        let mut evecs = A.to_cols_array();
        let mut evals = [0.0; 3];
        syevd(mosek::Uplo::LO, 3, & mut evecs, &mut evals).unwrap();
        let evecm = DMat3::from_cols_array(&evecs);

        let tlate = *b;
        let scl = DVec3::new(evals[0],evals[1],evals[2]);
        let rot = DQuat::from_mat3(&evecm);

        (scl,rot,tlate)
    }
}


