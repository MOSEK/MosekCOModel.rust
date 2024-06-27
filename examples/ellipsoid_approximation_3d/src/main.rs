extern crate mosekmodel;
extern crate bevy;
extern crate mosek;
extern crate ellipsoids;
extern crate rand;

use bevy::{prelude::*, math::{DMat3, DVec3}};
use linalg::symsqrt3;

use ellipsoids::Ellipsoid;
use mosekmodel::{unbounded, Model};

use std::f32::consts::PI;

const N : usize = 5;

#[derive(Default,Component)]
struct BoundingEllipsoid;

fn rand_dir3() -> Vec3 {
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
fn rand_vec3(range : f32,base : f32) -> Vec3 {
    Vec3::new(rand::random::<f32>()*range+base,
              rand::random::<f32>()*range+base,
              rand::random::<f32>()*range+base)
}

#[allow(non_snake_case)]
#[derive(Default,Component)]
struct EllipseTransform {
    center : Vec3,
    radii : Vec3,
    global_axis : Vec3,
    global_speed : f32,
    local_axis : Vec3,
    local_speed : f32,
}

impl EllipseTransform {
    pub fn rand(center_radius : f32, radius_range : (f32,f32), global_rps : (f32,f32), local_rps : (f32,f32)) -> EllipseTransform {
        let center_radius = center_radius.max(0.0);
        let radius_range  = ((radius_range.1-radius_range.0).abs(),radius_range.0.min(radius_range.1));
        let global_range  = ((global_rps.1-global_rps.0).abs(),global_rps.0.min(global_rps.1));
        let local_range   = ((local_rps.1-local_rps.0).abs(),local_rps.0.min(local_rps.1));
        EllipseTransform{
            center       : rand_dir3()*center_radius,
            radii        : rand_vec3(radius_range.0,radius_range.1),
            global_axis  : rand_dir3(),
            local_axis   : rand_dir3(),
            global_speed : rand::random::<f32>() * global_range.0 + global_range.1,
            local_speed  : rand::random::<f32>() * local_range.0 + local_range.1,
        }
    }
}


#[derive(Default,Component)]
struct CameraTransform{
    rps : f32
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
    mut materials: ResMut<Assets<StandardMaterial>>) 
{
    commands.spawn((PbrBundle{
        mesh : meshes.add(Sphere::new(1.0)),
        material : materials.add(Color::rgba_u8(255,255, 255, 128)),
        ..default()},
        BoundingEllipsoid{} 
        ));


    let color = Color::rgba_u8(192,128,0,255);
    let blobmat = materials.add(StandardMaterial{
        base_color : color,
        metallic : 0.8,
        reflectance : 0.5,
        ..default()
    });


    for _ in 0..N {
        commands.spawn((PbrBundle{
            mesh : meshes.add(Sphere::new(1.0)),
            material : materials.add(StandardMaterial{
                base_color : Color::rgb_u8(rand::random::<u8>()/2+127,rand::random::<u8>()/2+127, rand::random::<u8>()/2+127),
                metallic : 0.8,
                reflectance : 0.5,
                ..default()
            }),
            ..default()},

            EllipseTransform::rand(3.0, (0.2,1.5),(1.0/15.0,1.0/2.0), (1.0/10.0, 1.0) )
            ));
    }

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
        CameraTransform{ rps:1.0/15.0 }));
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
fn update(time       : Res<Time>, 
          mut query  : Query<(&mut Transform, &EllipseTransform)>, 
          mut qbound : Query<&mut Transform, Without<EllipseTransform>>,
          mut gizmos : Gizmos) {
    let t = time.elapsed_seconds();

    let mut matrixes = Vec::new();
    for (mut transform, e) in &mut query {
        transform.scale = e.radii;
        transform.rotation = Quat::from_axis_angle(e.local_axis, e.local_speed * t);
        transform.translation = Quat::from_axis_angle(e.global_axis, (e.global_speed*t) % (2.0*PI)).mul_vec3(e.center);

        let A = Mat3::from_cols(transform.rotation.mul_vec3(Vec3::new(e.radii.x,0.0,      0.0)),
                                transform.rotation.mul_vec3(Vec3::new(0.0,      e.radii.y,0.0)),
                                transform.rotation.mul_vec3(Vec3::new(0.0,      0.0,      e.radii.z)));

        // It should be possible to derive the symmetric transformation from scale and rotation,
        // but I'm too lazy to do the math. 
        
        matrixes.push((linalg::aatsymsqrt3(&A.as_dmat3()).unwrap(),transform.translation));
    }

   
    // outer ellipsoid
    let mut m = Model::new(None);
    let t = m.variable(None, unbounded());
    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 3);
    let q = m.variable(None, unbounded().with_shape(&[3]));

    m.objective(None, mosekmodel::Sense::Maximize, &t);
   
    for (A,b) in matrixes.iter() {
        // { Ax+b : ||x|| < 1 } = { u: || A\u - A\b | | < 1 }
        let Ainv = A.transpose().inverse();
        let b = -Ainv.mul_vec3(b.as_dvec3());

        let e : Ellipsoid<3> = ellipsoids::Ellipsoid::from_arrays(&Ainv.to_cols_array(), &b.to_array());

        ellipsoids::ellipsoid_contains(&mut m,&p,&q,&e);
    }

    m.solve();

    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
                                  m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
        // A² = P => A = sqrt(P)
        // Ab = q => b = A\q
        
        let Psq = DMat3::from_cols_array(&[psol[0],psol[1],psol[2],psol[3],psol[4],psol[5],psol[6],psol[7],psol[8]]);
        let q   = DVec3::new(qsol[0],qsol[1],qsol[2]);

        let A   = symsqrt3(&Psq).unwrap();
        let b   = A.inverse().mul_vec3(q);

        let (scl,rot,tlate) = linalg::axb_to_srt(&A, &b);

        for mut transform in & mut qbound {
            transform.scale = scl.as_vec3();
            transform.rotation = rot.as_quat();
            transform.translation = tlate.as_vec3();
        }


        //gizmos.linestrip(matrixes.iter().map(|(m,z)| DMat3::from_diagonal(scl).inverse().mul_vec3(rot.inverse().mul_vec3(z.as_dvec3()-tlate)).as_vec3()), Color::rgb_u8(255,0,0));
        gizmos.linestrip(matrixes.iter().map(|(_,z)| *z), Color::rgb_u8(255,0,0));

    } 
    else {
        for mut transform in & mut qbound {
            transform.scale = Vec3::new(0.001,0.001,0.001);
        }
        
    }

}

mod linalg {
    use bevy::{prelude::*, math::{DMat3, DVec3,DQuat}};
    use mosek::syevd;
    /// Compute the symmetric square root of AA' by using eigenvector decomposition as 
    /// ```math
    /// AA' = UDU' = (UD^{1/2}U')^2
    /// ```
    /// where `U` is the matrix of eigenvectors and `D` is the diagonal matrix of eigenvalues. Then
    ///
    /// # Arguments
    /// - `A` a positive definite matrix
    #[allow(non_snake_case)]
    pub fn aatsymsqrt3(A : &DMat3) -> Result<DMat3,String> {
        symsqrt3(&A.mul_mat3(&A.transpose()))
    }
    
    #[allow(non_snake_case)]
    pub fn symsqrt3(A : &DMat3) -> Result<DMat3,String> {
        let mut A = A.to_cols_array();
        let mut w = [0f64;3]; 
        syevd(mosek::Uplo::LO,3,&mut A,&mut w)?;
        if w.iter().any(|&v| v <= 0.0) { return Err("A is not positive definite".to_string()) }
        w.iter_mut().for_each(|v| *v = v.sqrt());
        
        let d = DVec3::new(w[0],w[1],w[2]);
        let U = DMat3::from_cols_array(&A);
        let D = DMat3::from_diagonal(d);
    
        Ok(U.mul_mat3(&D).mul_mat3(&U.transpose()))
    }
    
    #[allow(non_snake_case)]
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

