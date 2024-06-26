extern crate mosekmodel;
extern crate bevy;
extern crate mosek;
extern crate ellipsoids;

use bevy::{prelude::*, math::{DMat3, DVec3}};
use linalg::symsqrt3;

use ellipsoids::Ellipsoid;
use mosekmodel::{unbounded, Model};

use std::f32::consts::PI;

#[derive(Default,Component)]
struct BoundingEllipsoid;


#[allow(non_snake_case)]
#[derive(Default,Component)]
struct EllipseTransform {
    center : Vec3,
    radii : Vec3,
    global_axis : Vec3,
    global_speed : f32,
    local_axis : Vec3,
    local_speed : f32,
    
    Ab : Option<(DMat3,DVec3)>,
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
//    commands.spawn(PbrBundle {
//        mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
//        //material: materials.add(Color::rgb_u8(124, 144, 255)),
//        material: materials.add(Color::rgb_u8(192,0,0)),
//        transform: Transform::from_xyz(0.0, 0.5, 0.0),
//        ..default()
//    });

    // cube
/*
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        material: materials.add(Color::rgba_u8(124, 144, 255,92)),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..default()
    });
*/


    commands.spawn((PbrBundle{
        mesh : meshes.add(Sphere::new(1.0)),
        material : materials.add(Color::rgba_u8(92, 92, 192, 92)),
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

    commands.spawn((PbrBundle{
        mesh : meshes.add(Sphere::new(1.0)),
        material : blobmat.clone(),
        ..default()},

        EllipseTransform{ 
            center : Vec3::new(0.0,1.5,1.0),
            radii  : Vec3::new(0.4, 0.6, 1.2),
            global_axis : Vec3::new(1.0,1.0,1.0).normalize(),
            global_speed : 2.0*PI/10.0,
            local_axis : Vec3::new(1.0,1.0,0.0).normalize(),
            local_speed : 2.0*PI/15.0,
            ..default() }
        ));
    
    commands.spawn((PbrBundle{
        mesh : meshes.add(Sphere::new(1.0)),
        material : blobmat.clone(),
        ..default()},

        EllipseTransform{ 
            center : Vec3::new(0.0,0.0,-2.0),
            radii  : Vec3::new(1.2, 1.5, 1.0),
            global_axis : Vec3::new(1.0,-1.0,0.8).normalize(),
            global_speed : 2.0*PI/15.0,
            local_axis : Vec3::new(1.0,-1.0,-0.5).normalize(),
            local_speed : 2.0*PI/10.0,
            ..default() }
        ));
    
    commands.spawn((PbrBundle{
        mesh : meshes.add(Sphere::new(1.0)),
        material : blobmat.clone(),
        ..default()},

        EllipseTransform{ 
            center : Vec3::new(-1.0,-0.5,-0.5),
            radii  : Vec3::new(1.3, 1.2, 0.8),
            global_axis : Vec3::new(-1.0,1.0,0.8).normalize(),
            global_speed : 2.0*PI/13.0,
            local_axis : Vec3::new(1.0,0.0,1.0).normalize(),
            local_speed : 2.0*PI/11.0,
            ..default() }
        ));

    commands.spawn((PbrBundle{
        mesh : meshes.add(Sphere::new(1.0)),
        material : blobmat.clone(),
        ..default()},

        EllipseTransform{ 
            center : Vec3::new(1.0,0.0,0.0),
            radii  : Vec3::new(1.0, 1.5, 0.8),
            global_axis : Vec3::new(0.0,1.0,0.2).normalize(),
            global_speed : 2.0*PI/12.5,
            local_axis : Vec3::new(0.0,1.0,1.0).normalize(),
            local_speed : 2.0*PI/17.0,
            ..default() }
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
        let Ainv = A.transpose().inverse();
        let b = -Ainv.mul_vec3(b.as_dvec3());

        let e : Ellipsoid<3> = ellipsoids::Ellipsoid::from_arrays(&Ainv.to_cols_array(), &b.to_array());

        ellipsoids::ellipsoid_contains(&mut m,&p,&q,&e);
    }

    m.solve();

    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekmodel::SolutionType::Default,&p),
                                  m.primal_solution(mosekmodel::SolutionType::Default,&q)) {
        // A² = P => A = sqrt(P)
        // Ab = q => A\q
        
        let Psq = DMat3::from_cols_array(&[psol[0],psol[1],psol[2],psol[3],psol[4],psol[5],psol[6],psol[7],psol[8]]);
        let q   = DVec3::new(qsol[0],qsol[1],qsol[2]);

        let A   = symsqrt3(&Psq).unwrap();
        let b   = A.inverse().mul_vec3(q);

        let A = Mat4::from_mat3(A.as_mat3());

        for mut transform in & mut qbound {
            transform.clone_from(&Transform::from_matrix(A));
            transform.scale *= 6.0;
            transform.translation = b.as_vec3();
        }
    } 
    else {
        for mut transform in & mut qbound {
            transform.scale = Vec3::new(0.001,0.001,0.001);
        }
        
    }

}

mod linalg {
    use bevy::{prelude::*, math::{DMat3, DVec3}};
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
}

