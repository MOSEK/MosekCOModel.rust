extern crate mosekcomodel;
extern crate bevy;
extern crate mosek;
extern crate ellipsoids;
extern crate rand;

use rand::SeedableRng;

use bevy::{prelude::*, math::{DMat3, DVec3,DQuat}};
use linalg::symsqrt3;

use ellipsoids::Ellipsoid;
use mosekcomodel::{unbounded, Model};
use rand::Rng;

use bevy::render::view::screenshot::ScreenshotManager;
use bevy::window::PrimaryWindow;

use std::{f32::consts::PI, ops::Range};

const N : usize = 10;

#[derive(Default,Component)]
struct BoundingEllipsoid;

trait RandVec3 {
    fn ddir(& mut self) -> DVec3;
    fn dvec(& mut self,range : Range<f64>) -> DVec3;
    fn dir(& mut self) -> Vec3;
    fn vec(& mut self,range : Range<f32>) -> Vec3;
    fn drot(& mut self) -> DQuat;
    fn dball(& mut self) -> DVec3;
    fn ball(& mut self) -> Vec3;
}

impl RandVec3 for rand::rngs::StdRng {
    fn ddir(& mut self) -> DVec3 {
        self.dball().normalize()
    }

    fn dball(&mut self) -> DVec3 {
        loop {
            let r = DVec3::new(self.gen_range(-1.0..1.0),self.gen_range(-1.0..1.0),self.gen_range(-1.0..1.0));
            let l = r.length(); if l <= 1.0 && l > 0.0 { return r }
        }
    }

    fn ball(&mut self) -> Vec3 {
        loop {
            let r = Vec3::new(self.gen_range(-1.0..1.0),self.gen_range(-1.0..1.0),self.gen_range(-1.0..1.0));
            let l = r.length(); if l <= 1.0 && l > 0.0 { return r }
        }
    }

    fn dvec(& mut self, range : Range<f64>) -> DVec3 {
        DVec3::new(self.gen_range(range.clone()),
                   self.gen_range(range.clone()),
                  self.gen_range(range.clone()))
    }

    fn dir(& mut self) -> Vec3 {
        self.ball().normalize()
    }

    fn vec(& mut self, range : Range<f32>) -> Vec3 {
        Vec3::new(self.gen_range(range.clone()),
                  self.gen_range(range.clone()),
                  self.gen_range(range.clone()))
    }

    fn drot(& mut self) -> DQuat {
        DQuat::from_axis_angle(self.ddir(), self.gen_range(0.0..2.0*std::f64::consts::PI))
    }    
}


fn rand_dir3() -> Vec3 {
    loop {
        let r = Vec3::new(rand::random::<f32>()*2.0-1.0,
                          rand::random::<f32>()*2.0-1.0,
                          rand::random::<f32>()*2.0-1.0);
        if r.length() <= 1.0 {
            return r.normalize();
        }
    }
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

    #[allow(non_snake_case)]
    pub fn rand_from(R : & mut rand::rngs::StdRng, center_radius : f32, radius_range : Range<f32>, global_rps : Range<f32>, local_rps : Range<f32>) -> EllipseTransform {
        let center_radius = center_radius.max(0.0);
        EllipseTransform{
            center       : R.ddir().as_vec3()*center_radius,
            radii        : R.vec(radius_range),
            global_axis  : R.dir(),
            local_axis   : R.dir(),
            global_speed : R.gen_range(global_rps),
            local_speed  : R.gen_range(local_rps)
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
        //.add_systems(Update, screen_shot)
        .run();
}

//fn screen_shot( time       : Res<Time>,
//                main_window: Query<Entity, With<PrimaryWindow>>,
//                mut screenshot_manager: ResMut<ScreenshotManager>,
//                mut counter: Local<u32>)
//{
//    let sec = time.elapsed_seconds();
//    let path = format!("./screenshot-{}.png", *counter);
//    *counter += 1;
//    let _ = screenshot_manager
//        .save_screenshot_to_disk(main_window.single(), path)
//        .unwrap();
//}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>) 
{
    commands.spawn((PbrBundle{
        mesh : meshes.add(Sphere::new(1.0)),
        material : materials.add(Color::srgba_u8(255,255, 255, 128)),
        ..default()},
        BoundingEllipsoid{} 
        ));


    let color = Color::srgba_u8(192,128,0,255);
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
                base_color : Color::srgb_u8(rand::random::<u8>()/2+127,rand::random::<u8>()/2+127, rand::random::<u8>()/2+127),
                metallic : 0.8,
                reflectance : 0.5,
                ..default()
            }),
            ..default()},

            //EllipseTransform::rand(3.0, (0.2,1.5),(1.0/15.0,1.0/2.0), (1.0/10.0, 1.0) )
            EllipseTransform::rand(3.0, (0.5,1.5),(1.0/15.0,1.0/2.0), (1.0/10.0, 1.0) )
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
    let dt = t;

    let ellipses : Vec<Ellipsoid<3>> = (&mut query).iter_mut().map(|(mut transform,e)| {
    //for (mut transform, e) in &mut query {
        transform.scale = e.radii;
        transform.rotation = Quat::from_axis_angle(e.local_axis, e.local_speed * t);
        transform.translation = Quat::from_axis_angle(e.global_axis, (e.global_speed*t) % (2.0*PI)).mul_vec3(e.center);
        //transform.translation = Vec3::ZERO;

        let D = Mat3::from_diagonal(e.radii);
        let A = Mat3::from_cols(transform.rotation.mul_vec3(D.col(0)),
                                transform.rotation.mul_vec3(D.col(1)),
                                transform.rotation.mul_vec3(D.col(2))).as_dmat3();
        // Symmetric square root is also:
        // symA = A^T √D A
        let symA = linalg::symsqrt3(&A.mul_mat3(&A.transpose())).unwrap();

        //matrixes.push((symA,transform.translation));

        // { Ax+b : ||x|| < 1 } = { u: || A\u - A\b | | < 1 }
        let Ainv = symA.transpose().inverse();
        let b = -Ainv.mul_vec3(transform.translation.as_dvec3());

        ellipsoids::Ellipsoid::from_arrays(&Ainv.to_cols_array(), &b.to_array())
    }).collect();

    // outer ellipsoid
    let mut m = Model::new(None);
    let t = m.variable(None, unbounded());
    let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 3);
    let q = m.variable(None, unbounded().with_shape(&[3]));

    m.objective(None, mosekcomodel::Sense::Maximize, &t);

    for e in ellipses.iter() {
        ellipsoids::ellipsoid_contains(&mut m,&p,&q,e);
    }

    m.solve();

    if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekcomodel::SolutionType::Default,&p),
                                  m.primal_solution(mosekcomodel::SolutionType::Default,&q)) {
        // A² = P => A = sqrt(P)
        // Ab = q => b = A\q
        
        let Psq = DMat3::from_cols_array(&[psol[0],psol[1],psol[2],psol[3],psol[4],psol[5],psol[6],psol[7],psol[8]]);
        let Pq  = DVec3::new(qsol[0],qsol[1],qsol[2]);

        let P   = symsqrt3(&Psq).unwrap();
        let q   = P.inverse().mul_vec3(Pq); 

        let A   = P.inverse();
        let b   = -A.mul_vec3(q);

        let (scl,rot,tlate) = linalg::axb_to_srt(&A, &b);

        for mut transform in & mut qbound {
            transform.scale = scl.as_vec3();
            transform.rotation = rot.as_quat();
            transform.translation = tlate.as_vec3();
        }
    } 
    else {
        for mut transform in & mut qbound {
            transform.scale = Vec3::new(0.001,0.001,0.001);
        }
        
    }

}

mod linalg {
    use bevy::math::{DMat3, DVec3,DQuat};
    use mosek::syevd;
    /// Compute the symmetric square root of (symmetric) A by using eigenvector decomposition as 
    /// ```math
    /// A = B'DB = (B'D^{1/2}B)^2 => √A = B' D^{1/2} B
    /// ```
    /// where `B` is the matrix of eigenvectors (orthogonal basis) and `D` is the diagonal matrix
    /// of eigenvalues. 
    ///
    /// # Arguments
    /// - `A` a positive definite matrix
    #[allow(non_snake_case)]
    pub fn symsqrt3(A : &DMat3) -> Result<DMat3,String> {
        let mut A = A.to_cols_array();
        let mut w = [0f64;3]; 
        syevd(mosek::Uplo::LO,3,&mut A,&mut w)?;
        if w.iter().any(|&v| v <= 0.0) { return Err("A is not positive definite".to_string()) }
        w.iter_mut().for_each(|v| *v = v.sqrt());
        //println!("-- eig vals : {:?}",w);
        
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


#[cfg(test)]
mod test {
    use bevy::math::{Quat, Mat3, DVec3, DMat3, Vec3};
    use rand::{self, SeedableRng, Rng};
    use ellipsoids::Ellipsoid;
    use mosekcomodel::{Model,unbounded,Variable};
    use super::{linalg,RandVec3};
    use std::f32::consts::PI;

    const NSAMPLE : usize = 1000;


    #[allow(non_snake_case)]
    #[test]
    fn test_sim() {
        use super::EllipseTransform;

        let mut R = rand::rngs::StdRng::seed_from_u64(123456);

        let ellipsis_transforms : Vec<EllipseTransform> = (0..5).map(|_| EllipseTransform::rand_from(&mut R,3.0, 0.5..1.5,1.0/15.0..1.0/2.0, 1.0/10.0..1.0 ) ).collect();

        for i in 0..100 {
            let dt = 0.35315 * i as f32;

            let ellipses : Vec<(Ellipsoid<3>,Vec3,Quat,Vec3)> = ellipsis_transforms.iter().map(|e| {
                let e_scale = e.radii;
                let e_rot   = Quat::from_axis_angle(e.local_axis, e.local_speed * dt);
                let e_tlate = Quat::from_axis_angle(e.global_axis, (e.global_speed*dt) % (2.0*PI)).mul_vec3(e.center);

                let D = Mat3::from_diagonal(e.radii);
                let A = Mat3::from_cols(e_rot.mul_vec3(D.col(0)),
                                        e_rot.mul_vec3(D.col(1)),
                                        e_rot.mul_vec3(D.col(2))).as_dmat3();
                // Symmetric square root is also:
                // symA symA' = U D U' = A'A
                let symA = linalg::symsqrt3(&A.mul_mat3(&A.transpose())).unwrap();

                // { Ax+b : ||x|| < 1 } = { u: || A\u - A\b | | < 1 }
                let Ainv = symA.transpose().inverse();
                let b = -Ainv.mul_vec3(e_tlate.as_dvec3());

                let e = ellipsoids::Ellipsoid::from_arrays(&Ainv.to_cols_array(), &b.to_array());

                let (P,q) = e.get_Pq();
                let P = DMat3::from_cols(DVec3::from_array(P[0]),DVec3::from_array(P[1]),DVec3::from_array(P[2]));
                let q = DVec3::from_array(q);

                // TEST
                // Now {x:||Px+q||<1} should map to the same ellipsoid as e_scale,e_rot,e_tlate
                // Verify that they do
                for v in (0..NSAMPLE).map(|_| R.ddir()) {
                    let w = (e_rot.mul_vec3(Mat3::from_diagonal(e_scale).mul_vec3(v.as_vec3())) + e_tlate).as_dvec3();
                    assert!(((q + P.mul_vec3(w)).length()-1.0).abs() < 0.00001);
                }

                (e,e_scale,e_rot,e_tlate)
            }).collect();

            // outer ellipsoid
            let mut m = Model::new(None);
            let t = m.variable(None, unbounded());
            let p = ellipsoids::det_rootn(None, & mut m, t.clone(), 3);
            let q = m.variable(None, unbounded().with_shape(&[3]));

            m.objective(None, mosekcomodel::Sense::Maximize, &t);

            let taus : Vec<Variable<0>>= ellipses.iter().map(|(e,_,_,_)| ellipsoids::ellipsoid_contains(&mut m,&p,&q,e)).collect();

            m.solve();

            if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekcomodel::SolutionType::Default,&p),
                                          m.primal_solution(mosekcomodel::SolutionType::Default,&q)) {

                // A² = P => A = sqrt(P)
                // Ab = q => b = A\q
                
                let Psq = DMat3::from_cols_array(&[psol[0],psol[1],psol[2],psol[3],psol[4],psol[5],psol[6],psol[7],psol[8]]);
                let Pq  = DVec3::new(qsol[0],qsol[1],qsol[2]);
                
                let P   = linalg::symsqrt3(&Psq).unwrap();
                let q   = P.inverse().mul_vec3(Pq);

                let A    = P.inverse();
                let b    = - A.mul_vec3(q);
                
                let Pinv = P.inverse();

                // Verify that the solution satisfies constraint for all ellipsoids
                for (tau,(e,_,_,_)) in taus.iter().zip(ellipses.iter()) {
                    let tau = m.primal_solution(mosekcomodel::SolutionType::Default,tau).unwrap();
                    let (a,b,c) = e.get_Abc();
                    let b = DVec3::from_array(b);
                    let a = DMat3::from_cols(DVec3::from_array(a[0]),DVec3::from_array(a[1]),DVec3::from_array(a[2]));
                    let s11 = P.mul_mat3(&P) - a.mul_scalar(tau[0]);
                    let s21 = P.mul_vec3(q) - b * tau[0];
                    let s22 = -(tau[0] * c + 1.0);
                    let s32 = P.mul_vec3(q);
                    let s33 = - P.mul_mat3(&P);

                    let mut mx : [f64;49] = [
                        s11.col(0)[0],s11.col(1)[0],s11.col(2)[0], s21[0], 0.0, 0.0, 0.0,
                        s11.col(0)[1],s11.col(1)[1],s11.col(2)[1], s21[1], 0.0, 0.0, 0.0,
                        s11.col(0)[2],s11.col(1)[2],s11.col(2)[2], s21[2], 0.0, 0.0, 0.0,
                        s21[0],       s21[1],       s21[2],        s22,    s32[0],s32[1],s32[2],
                        0.0,          0.0,          0.0,           s32[0], s33.col(0)[0],s33.col(1)[0],s33.col(2)[0],
                        0.0,          0.0,          0.0,           s32[1], s33.col(0)[1],s33.col(1)[1],s33.col(2)[1],
                        0.0,          0.0,          0.0,           s32[2], s33.col(0)[2],s33.col(1)[2],s33.col(2)[2] ];
                    let mut ev = [0.0; 7];

                    // This matrix must be negative semi definite
                    mosek::syevd(mosek::Uplo::LO, 7, &mut mx, & mut ev).unwrap();
                    assert!(* ev.iter().max_by(|a,b| a.total_cmp(b)).unwrap() < 0.00001);
                    
                    let (ep,eq) = e.get_Pq();
                    let eq = DVec3::from_array(eq);
                    let ep = DMat3::from_cols(DVec3::from_array(ep[0]),DVec3::from_array(ep[1]),DVec3::from_array(ep[2]));

                    // Verify that the ellipse e is contained in computed ellipsoid
                    for v in (0..NSAMPLE).map(|_| R.dball()) {
                        let r = P.mul_mat3(&ep.inverse()).mul_vec3(v - eq) + q;
                        assert!(r.length() < 1.00001);
                    }
                }



                let (scl,rot,tlate) = linalg::axb_to_srt(&A, &b);

                // TEST: Does A surround every ellipse e
                {
                    println!("---------------------------");
                    for v in (0..1000).map(|_| R.ddir()) {
                        assert!((DMat3::from_diagonal(scl).inverse().mul_vec3(rot.inverse().mul_vec3(A.mul_vec3(v))).length()-1.0).abs() < 0.00001);

                        for (e,e_scale,e_rot,e_tlate) in ellipses.iter() {
                            let (eP,eq) = e.get_Pq();
                            let eP = DMat3::from_cols(DVec3::from_array(eP[0]),DVec3::from_array(eP[1]),DVec3::from_array(eP[2]));
                            let eq = DVec3::from_array(eq);

                            {
                                let p = eP.inverse().mul_vec3(v-eq);

                                // Px+q maps the ellipsoid {x:||eP x + eq|| <= 1} into the interior of the
                                // unit ball
                                assert!((P.mul_vec3(p)+q).length() <= 1.00001);
                                assert!((A.inverse().mul_vec3(p-b)).length() < 1.00001);
                            }

                            {
                                // the rotation,scale,translate should also map into the interior of
                                // the the solution sphere {x:||Px+q||<=1}
                                
                                let p = rot.mul_vec3(DMat3::from_diagonal(scl).mul_vec3(v))+tlate;
                                assert!((P.mul_vec3(p)+q).length() <= 1.00001);
                                assert!(A.inverse().mul_vec3(p-b).length() <= 1.00001);
                            }
                        }
                    }
                }
            }
            else {
                assert!(false);
            }
        }
    }






    #[allow(non_snake_case)]
    #[test]
    fn test() {
        let mut R = rand::rngs::StdRng::seed_from_u64(123456);

        for _ in 0..10 {
            let escl = R.dvec(0.2..2.0);
            let erot = R.drot();

            let D = DMat3::from_diagonal(escl);
            let Rt = DMat3::from_cols(erot.mul_vec3(DVec3::new(1.0,0.0,0.0)),
                                      erot.mul_vec3(DVec3::new(0.0,1.0,0.0)),
                                      erot.mul_vec3(DVec3::new(0.0,0.0,1.0)));
            let A = DMat3::from_cols(erot.mul_vec3(D.col(0)),
                                     erot.mul_vec3(D.col(1)),
                                     erot.mul_vec3(D.col(2)));

            let symA = super::linalg::symsqrt3(&A.mul_mat3(&A.transpose())).unwrap();

            // Test that AA' == symA symA
            assert!((A.mul_mat3(&A.transpose()) - symA.mul_mat3(&symA)).to_cols_array().iter().map(|v| v.abs()).sum::<f64>() < 0.00001);
                
            // Check that A and symA maps to the same ellipsoid, i.e. 
            // || A^{-1} * symA * v || = 1 for all v:||v|| = 1, or equivalently
            //
            // A^{-1} * symA defines an orthogonal basis
            let Ainv = A.inverse();
            let symAinv = symA.inverse();
            {
                let B = Ainv * symA;
                //println!("B = {:?}",B);
                //println!("  : {},{},{}",B.col(0).length(),B.col(1).length(),B.col(2).length());
                //println!("  : {},{},{}",B.col(0).dot(B.col(1)),B.col(0).dot(B.col(2)),B.col(1).dot(B.col(2)));

                assert!(B.col(0).dot(B.col(1)).abs() < 0.00001);
                assert!(B.col(0).dot(B.col(2)).abs() < 0.00001);
                assert!(B.col(1).dot(B.col(2)).abs() < 0.00001);
            }
            {
                let B = symAinv * A;
                //println!("B = {:?}",B);
                assert!(B.col(0).dot(B.col(1)).abs() < 0.00001);
                assert!(B.col(0).dot(B.col(2)).abs() < 0.00001);
                assert!(B.col(1).dot(B.col(2)).abs() < 0.00001);
            }

            // Test that rot*scl maps into the same ellipsoid as symA
            for v in (0..NSAMPLE).map(|_| R.ddir()) {
                //println!("v = {:?}, ||A^(-1) S v|| = {:?}, ||symA^(-1) A v|| = {}",
                //         v, A.inverse().mul_vec3(symA.mul_vec3(v)).length(),
                //         symA.inverse().mul_vec3(A.mul_vec3(v)).length());

                //// Check that A and symA maps to the same ellipsoid, i.e. 
                //// A^{-1} * symA * v
                //assert!((A.inverse().mul_vec3(symA.mul_vec3(v)).length()-1.0).abs() < 0.00001);
                //assert!((symA.inverse().mul_vec3(A.mul_vec3(v)).length()-1.0).abs() < 0.00001);

                // Check identity of A and erot*escl
                assert!((Rt.mul_vec3(v) - erot.mul_vec3(v)).length() < 0.00001);
                assert!((A.mul_vec3(v) - erot.mul_vec3(D.mul_vec3(v))).length().abs() < 0.00001);

                // Check equivalence of symA and erot*escl, i.e. for v:||v||=1:
                // 1. || symA^{-1} * erot * escl * v || = 1
                // 2. || escl^{-1} * erot^{-1} * symA * v ||

                assert!( (Ainv.mul_vec3(erot.mul_vec3(D.mul_vec3(v))).length()-1.0).abs() < 0.00001 );
                assert!( (D.inverse().mul_vec3(erot.inverse().mul_vec3(A.mul_vec3(v))).length()-1.0).abs() < 0.00001 );

                assert!( (symAinv.mul_vec3(erot.mul_vec3(D.mul_vec3(v))).length()-1.0).abs() < 0.00001 );
                assert!( (D.inverse().mul_vec3(erot.inverse().mul_vec3(symA.mul_vec3(v))).length()-1.0).abs() < 0.00001 );
            }

            // outer ellipsoid
            let mut m = Model::new(None);
            let t = m.variable(None, unbounded());
            let Psq = ellipsoids::det_rootn(None, & mut m, t.clone(), 3);
            let q = m.variable(None, unbounded().with_shape(&[3]));

            m.objective(None, mosekcomodel::Sense::Maximize, &t);
           
            let (tau,e) = {
                let A = symA;
                let b = DVec3::ZERO;
                // { Ax+b : ||x|| < 1 } = { u: || A\u - A\b | | < 1 }
                let Z = A.inverse();
                let w = -Z.mul_vec3(b);

                let e : Ellipsoid<3> = Ellipsoid::from_arrays(&Z.to_cols_array(), &w.to_array());

                {
                    let (p,q) = e.get_Pq();
                    let (a,b,c) = e.get_Abc();
                    let p = DMat3::from_cols(DVec3::from_array(p[0]),DVec3::from_array(p[1]),DVec3::from_array(p[2]));
                    let a = DMat3::from_cols(DVec3::from_array(a[0]),DVec3::from_array(a[1]),DVec3::from_array(a[2]));

                    let ptp = p.transpose().mul_mat3(&p);
                    assert!((a-ptp).to_cols_array().iter().map(|v| v.abs()).sum::<f64>() < 0.00001);
                }

                (ellipsoids::ellipsoid_contains(&mut m,&Psq,&q,&e),e)
            };

            m.solve();

            if let (Ok(psol),Ok(qsol)) = (m.primal_solution(mosekcomodel::SolutionType::Default,&Psq),
                                          m.primal_solution(mosekcomodel::SolutionType::Default,&q)) {
                let P = linalg::symsqrt3(&DMat3::from_cols_array(&[psol[0],psol[1],psol[2],psol[3],psol[4],psol[5],psol[6],psol[7],psol[8]])).unwrap();
                let Pinv = P.inverse();
                let q = Pinv.mul_vec3(DVec3::new(qsol[0],qsol[1],qsol[2]));
                //assert!(Pq.length() < 0.0001);
               

                // Verify that the solution actually satisfies the PSD constraint
                if let Ok(tau) = m.primal_solution(mosekcomodel::SolutionType::Default,&tau) 
                {
                    let (a,b,c) = e.get_Abc();
                    let b = DVec3::from_array(b);
                    let a = DMat3::from_cols(DVec3::from_array(a[0]),DVec3::from_array(a[1]),DVec3::from_array(a[2]));
                    let s11 = P.mul_mat3(&P) - a.mul_scalar(tau[0]);
                    let s21 = P.mul_vec3(q) - b * tau[0];
                    let s22 = -(tau[0] * c + 1.0);
                    let s32 = P.mul_vec3(q);
                    let s33 = - P.mul_mat3(&P);

                    let mut mx : [f64;49] = [
                        s11.col(0)[0],s11.col(1)[0],s11.col(2)[0], s21[0], 0.0, 0.0, 0.0,
                        s11.col(0)[1],s11.col(1)[1],s11.col(2)[1], s21[1], 0.0, 0.0, 0.0,
                        s11.col(0)[2],s11.col(1)[2],s11.col(2)[2], s21[2], 0.0, 0.0, 0.0,
                        s21[0],       s21[1],       s21[2],        s22,    s32[0],s32[1],s32[2],
                        0.0,          0.0,          0.0,           s32[0], s33.col(0)[0],s33.col(1)[0],s33.col(2)[0],
                        0.0,          0.0,          0.0,           s32[1], s33.col(0)[1],s33.col(1)[1],s33.col(2)[1],
                        0.0,          0.0,          0.0,           s32[2], s33.col(0)[2],s33.col(1)[2],s33.col(2)[2] ];
                    let mut ev = [0.0; 7];

                    // This matrix must be negative semi definite
                    mosek::syevd(mosek::Uplo::LO, 7, &mut mx, & mut ev).unwrap();
                    assert!(* ev.iter().max_by(|a,b| a.total_cmp(b)).unwrap() < 0.00001);
                    
                    let (ep,eq) = e.get_Pq();
                    let eq = DVec3::from_array(eq);
                    let ep = DMat3::from_cols(DVec3::from_array(ep[0]),DVec3::from_array(ep[1]),DVec3::from_array(ep[2]));

                    // Verify that the ellipse e is contained in computed ellipsoid
                    for v in (0..NSAMPLE).map(|_| R.dball()) {
                        //println!("P.inverse().mul_vec3(A.mul_vec3(v)).length() = {}",P.inverse().mul_vec3(symA.mul_vec3(v)).length());
                        let r = P.mul_mat3(&ep.inverse()).mul_vec3(v - eq) + q;
                        assert!(r.length() < 1.00001);
                    }
                    
                    println!("ep   = {:?}\nsymA = {:?}",ep,symA);
                }
                


                let Z = Pinv;
                let w = Z.mul_vec3(q);
               
                // Test that P (nearly) contains A

                for v in (0..NSAMPLE).map(|_| R.dball()) {
                    //println!("P.inverse().mul_vec3(A.inverse().mul_vec3(v)).length() = {}",P.inverse().mul_vec3(symA.inverse().mul_vec3(v)).length());
                    assert!( P.inverse().mul_vec3(symA.inverse().mul_vec3(v)).length() < 1.00001);
                }

                // Test decomposition of P into rotation asnd scale

                let (scl,rot,_tlate) = linalg::axb_to_srt(&Z, &w);
                let scale = DMat3::from_diagonal(scl);
                let scale_inv = scale.inverse();
                let rot_inv = rot.inverse();

                for v in (0..NSAMPLE).map(|_| R.ddir()) {
                    // Check that the decomposition of P is correct. If P maps a point on the unit ball
                    // to a point on the ellipsoid, then the decomposition into scale and rotation
                    // should map it back on the unit ball (although to a different point).
                    
                    let mut evecs = Z.to_cols_array();
                    let mut evals = [0.0; 3];
                    mosek::syevd(mosek::Uplo::LO, 3, & mut evecs, &mut evals).unwrap();
                    assert!(* evals.iter().min_by(|a,b| a.total_cmp(b)).unwrap() >= 0.0); // Z is PSD
                    let evecm = DMat3::from_cols_array(&evecs);
                    let scl   = DMat3::from_diagonal(DVec3::new(evals[0],evals[1],evals[2]));

                    // Verify that decomposition is as expected
                    assert!(scl.inverse().mul_vec3(evecm.inverse().mul_vec3(Z.mul_vec3(v))).length() < 1.00001 );
                    
                    // Verify that scale, rotation maps to the same ellipsoid as Z

                    assert!((scale_inv.mul_vec3(rot_inv.mul_vec3(Z.mul_vec3(v))).length()-1.0).abs() <= 0.00001);
                    assert!((Z.inverse().mul_vec3(rot.mul_vec3(scale.mul_vec3(v))).length()-1.0).abs() <= 0.00001);
                }


                for v in (0..NSAMPLE).map(|_| R.ddir()) {
                    // Check that the decomposition of P is correct. If P maps a point on the unit ball
                    // to a point on the ellipsoid, then the decomposition into scale and rotation
                    // should map it back on the unit ball (although to a different point).

                    assert!((Z.inverse().mul_vec3(rot.mul_vec3(scale.mul_vec3(v))).length()-1.0).abs() <= 0.00001);
                }
                for v in (0..NSAMPLE).map(|_| R.ddir()) {
                    let e_scl = DMat3::from_diagonal(escl);
                    let e_rot = erot;

                    let r = scale_inv.mul_vec3(rot.inverse().mul_vec3( e_rot.mul_vec3(e_scl.mul_vec3(v)) )).length();
                    assert!(scale_inv.mul_vec3(rot.inverse().mul_vec3( e_rot.mul_vec3(e_scl.mul_vec3(v)) )).length() <= 1.1);
                }
            }
        }
    }
}

