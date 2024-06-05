// Copied from Azsek, whales.js

#[allow(non_snake_case)]
pub fn matmul(A : &[[f64;2];2], b : &[f64;2]) -> [f64;2] { [A[0][0]*b[0]+A[0][1]*b[1], A[1][0]*b[0]+A[1][1]*b[1]] }
#[allow(non_snake_case)]
pub fn mat2mul(A : &[[f64;2];2], B : &[[f64;2];2])  -> [[f64;2];2] { [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]], [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]] }
#[allow(non_snake_case)]
pub fn det(A : &[[f64;2];2])  -> f64 { A[0][0]*A[1][1] - A[0][1]*A[1][0] }
#[allow(non_snake_case)]
pub fn trans(A : &[[f64;2];2]) -> [[f64;2];2] { [[A[0][0], A[1][0]], [A[0][1], A[1][1]]] }
#[allow(non_snake_case)]
pub fn inv(A : &[[f64;2];2])  -> [[f64;2];2] { 
    let da = det(&A);
    [[A[1][1]/da, -A[0][1]/da], [-A[1][0]/da, A[0][0]/da]]
}
#[allow(non_snake_case)]
pub fn scale(a : &[f64;2], c : f64) -> [f64;2] { [c*a[0], c*a[1]] }
#[allow(non_snake_case)]
pub fn matscale(A : &[ [f64;2];2], c : f64) -> [ [f64;2];2] { [[c*A[0][0], c*A[0][1]], [c*A[1][0], c*A[1][1]]] }
#[allow(non_snake_case)]
pub fn vecadd(a : &[f64;2], b : &[f64;2]) -> [f64;2] { [a[0]+b[0], a[1]+b[1]] }
#[allow(non_snake_case)]
pub fn vecsub(a : &[f64;2], b : &[f64;2]) -> [f64;2] { [a[0]-b[0], a[1]-b[1]] }
#[allow(non_snake_case)]
pub fn matadd(A : &[ [f64;2];2], B : &[ [f64;2];2]) -> [[f64;2];2] { [[A[0][0]+B[0][0], A[0][1]+B[0][1]],[A[1][0]+B[1][0], A[1][1]+B[1][1]]] }
#[allow(non_snake_case)]
pub fn matsub(A : &[ [f64;2];2], B : &[ [f64;2];2]) -> [[f64;2];2] { [[A[0][0]-B[0][0], A[0][1]-B[0][1]],[A[1][0]-B[1][0], A[1][1]-B[1][1]]] }
#[allow(non_snake_case)]
pub fn trace(A : &[ [f64;2];2]) -> f64 { A[0][0] + A[1][1] } 
#[allow(non_snake_case)]
pub fn rot(alpha : f64) -> [[f64;2];2] { [[alpha.cos(), -alpha.sin()], [alpha.sin(), alpha.cos()]] }

