use itertools::izip;
use crate::expr::Expr;


pub trait Matrix  {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn transpose(&self) -> Self;

    fn shape(&self) -> [usize; 2];
    fn reshape(self,shape : [usize; 2]) -> Result<Self,()> where Self:Sized;
    fn nnz(&self) -> usize;
    fn data(&self) -> &[f64];
    fn sparsity(&self) -> Option<&[usize]>; 
    fn inplace_mul_scalar(&mut self, s : f64);
    fn dissolve(self) -> ([usize;2],Option<Vec<usize>>,Vec<f64>);
    fn to_dense(&self) -> Self;
}

///////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct NDArray<const N : usize> {
    shape : [usize; N],
    sp    : Option<Vec<usize>>,
    data  : Vec<f64>,
}

impl Matrix for NDArray<2> {
    fn width(&self) -> usize { self.shape()[1] }
    fn height(&self) -> usize { self.shape()[0] }
    fn transpose(&self) -> NDArray<2> {
        let shape = [self.shape[1],self.shape[0]];
        if let Some(ref sp) = self.sp {
            let n = sp.len();

            let mut ptr = vec![0; self.shape[1]+1];
            sp.iter().for_each(|&i| unsafe{ *ptr.get_unchecked_mut(1 + i % self.shape[1]) += 1 });
            _ = ptr.iter_mut().fold(0,|c,p| {*p += c; *p });

            let mut rsp = vec![0usize; n];
            let mut rdata = vec![0.0; n];

            for (&k,&d) in sp.iter().zip(self.data.iter()) {
                let (i,j) = (k / self.shape[1], k % self.shape[1]); 
                let p = unsafe{ *ptr.get_unchecked(j) };
                unsafe {
                    *rsp.get_unchecked_mut(p) = j*self.shape[0] + i;
                    *rdata.get_unchecked_mut(p) = d;
                    *ptr.get_unchecked_mut(j) += 1;
                }
            }

            NDArray{
                shape,
                sp : Some(rsp),
                data:rdata
            }
        }
        else {
            let data : Vec<f64> = (0..self.shape[1]).map(|j| self.data[j..].iter().step_by(self.shape[1])).flat_map(|it| it.clone()).map(|&i| i).collect();
            NDArray { shape, sp : None, data }
        }
    }
    fn shape(&self) -> [usize; 2] { self.shape() } 
    fn reshape(self,shape : [usize; 2]) -> Result<NDArray<2>,()> { self.reshape(shape) }
    fn nnz(&self) -> usize { self.nnz() }
    fn data(&self) -> &[f64] { self.data() }
    fn sparsity(&self) -> Option<&[usize]> { self.sparsity() } 
    fn inplace_mul_scalar(&mut self, s : f64) { self.inplace_mul_scalar(s) }
    fn dissolve(self) -> ([usize;2],Option<Vec<usize>>,Vec<f64>) { self.dissolve() }
    fn to_dense(&self) -> Self { self.to_dense() }
}

impl<const N : usize> NDArray<N> {
    pub fn new(shape : [usize;N], sp : Option<Vec<usize>>, data : Vec<f64>) -> Result<NDArray<N>,String>  { 
        // validate data
        if let Some(sp) = sp {
            if sp.len() > 1 && sp.iter().zip(sp[1..].iter()).any(|(&i0,&i1)| i1 <= i0) {
                Err("Sparsity is unsorted or contains duplicates".to_string())
            }
            else if sp.len() != data.len() {
                Err("Mismatching sparsity and data lengths".to_string())
            }
            else if sp.len() > 0 && shape.iter().product::<usize>() <= *sp.last().unwrap() {
                Err("Mismatching sparsity and shape".to_string())
            }
            else {
                Ok(NDArray{ shape,sp : Some(sp),data })
            }
        }
        else {
            let nnz : usize = shape.iter().product();
            if nnz != data.len() {
                Err("Mismatching data and shape".to_string())
            }
            else {
                Ok(NDArray{shape,sp:None,data})
            }
        }
    }

    pub fn from_iter<I>(shape : [usize; N], it : I) -> Result<NDArray<N>,String> where I : Iterator<Item = ([usize;N],f64)>{
        let mut strides = [0usize;N];
        _ = strides.iter_mut().zip(shape.iter()).rev().fold(1usize, |c,(s,d)| { *s = c; c*d });

        let mut sp = Vec::new();
        let mut data = Vec::new();
        for (i,v) in it {
            if i.iter().zip(shape.iter()).any(|(j,d)| j >= d) {
                return Err("Index out of bounds".to_string());
            }
            sp.push( i.iter().zip(strides.iter()).map(|(a,b)| a*b).sum());
            data.push(v);
        }

        NDArray::from_flat_tuples_internal(shape, sp.as_slice(), data.as_slice())
    }

    pub fn from_tuples(shape : [usize; N], index : &[ [usize; N] ], data : &[f64]) -> Result<NDArray<N>,String>{
        if data.len() != index.len() {
            Err("Mismatching data and index lengths".to_string())
        }
        else if index.len() > 0 && index.iter().any(|i| i.iter().zip(shape.iter()).any(|(&j,&d)| j >= d)) {
            Err("Index out of bounds".to_string())
        }
        else if index.len() == 0 {
            Ok(NDArray{shape, sp : Some(Vec::new()), data : data.to_vec()})
        }
        else {
            let mut strides = [1usize; N]; _ = strides.iter_mut().zip(shape.iter()).rev().fold(1usize, |c,(s,d)| {*s = c; c*d} );

            let sp_unordered : Vec<usize> = index.iter().map(|i| i.iter().zip(strides.iter()).map(|(j,s)| j*s).sum() ).collect();

            NDArray::from_flat_tuples_internal(shape, sp_unordered.as_slice(), data)
        }
    }

    fn from_flat_tuples_internal(shape : [usize; N], sp_unordered : &[usize], data : &[f64]) -> Result<NDArray<N>,String>{
        if sp_unordered.iter().zip(sp_unordered[1..].iter()).any(|(a,b)| a >= b) {
            // sp is unordered
            let mut perm : Vec<usize> = (0..sp_unordered.len()).collect();
            perm.sort_by_key(|i| unsafe { *sp_unordered.get_unchecked(*i) });
            if perm.iter().zip(perm[1..].iter()).any(|(&i0,&i1)| unsafe{ *sp_unordered.get_unchecked(i0) == *sp_unordered.get_unchecked(i1) } ) {
                // eliminate duplicates
                let nunique = perm.len() - perm.iter().zip(perm[1..].iter()).filter(|(&i0,&i1)| unsafe{ *sp_unordered.get_unchecked(i0) == *sp_unordered.get_unchecked(i1) } ).count();
                let mut rsp = vec![0usize; nunique];
                let mut rdata = vec![0.0f64; nunique];

                rsp[0] = sp_unordered[perm[0]];
                rdata[0] = data[perm[0]];

                let mut i = 0usize;
                for (&p0,&p1) in izip!(perm.iter(),perm[1..].iter()) {
                    let i0 = unsafe { *sp_unordered.get_unchecked(p0) };
                    let i1 = unsafe { *sp_unordered.get_unchecked(p1) };
                    if i0 != i1 {
                        i += 1;
                        unsafe{ *rsp.get_unchecked_mut(i) = i1 };
                    }
                    unsafe { *rdata.get_unchecked_mut(i) += *data.get_unchecked(p1) };
                }
                Ok(NDArray{ shape, sp:Some(rsp), data: data.to_vec()})
            }
            else {
                let sp = perm.iter().map(|&i| unsafe{ *sp_unordered.get_unchecked(i)} ).collect();
                let data = perm.iter().map(|&i| unsafe{ *data.get_unchecked(i)} ).collect();

                Ok(NDArray{ shape, sp : Some(sp), data })
            }
        }
        else {
            Ok(NDArray{ shape, sp : Some(sp_unordered.to_vec()), data : data.to_vec() })
        }
    }

    pub fn shape(&self) -> [usize; N] { self.shape }
    pub fn reshape<const M : usize>(self,shape : [usize; M]) -> Result<NDArray<M>,()> {
        if shape.iter().product::<usize>() != self.shape.iter().product() {
            Err(())
        }
        else {
            Ok(NDArray{ shape,sp : self.sp, data : self.data })
        }
    }
    pub fn nnz(&self) -> usize { self.data.len() }
    pub fn data(&self) -> &[f64] { self.data.as_slice() }
    pub fn sparsity(&self) -> Option<&[usize]> { if let Some(ref sp) = self.sp { Some(sp.as_slice()) } else { None } }
    pub fn inplace_mul_scalar(&mut self, s : f64) { self.data.iter_mut().for_each(|v| *v *= s); }
    pub fn dissolve(self) -> ([usize;N],Option<Vec<usize>>,Vec<f64>) { (self.shape,self.sp,self.data) }
    pub fn to_dense(&self) -> NDArray<N> {
        if let Some(ref sp) = self.sp {
            let mut data = vec![0.0; self.shape.iter().product()];
            assert!(sp.iter().max().map(|&v| v < data.len()).unwrap_or(true));
            for (&i,&f) in izip!(sp.iter(),self.data.iter()) {
                unsafe { *data.get_unchecked_mut(i) = f };
            }
            NDArray{
                shape : self.shape,
                sp : None,
                data
            }
        }
        else {
            self.clone()
        }
    }
    pub fn to_expr(&self) -> super::expr::Expr<N> {
        if let Some(ref sp) = self.sp {
            Expr::new(
                &self.shape,
                Some(sp.clone()),
                (0..sp.len()+1).collect(),
                vec![0; sp.len()],
                self.data.clone())
        }
        else {            
            Expr::new(
                &self.shape,
                None,
                (0..self.nnz()+1).collect(),
                vec![0; self.nnz()],
                self.data.clone())
        }
    }
}


impl From<&[f64]> for NDArray<1> {
    fn from(v : &[f64]) -> NDArray<1> {
        NDArray{ shape : [ v.len() ], sp : None, data : v.to_vec() }
    }
}

impl From<Vec<f64>> for NDArray<1> {
    fn from(v : Vec<f64>) -> NDArray<1> {
        NDArray{ shape : [ v.len() ], sp : None, data : v }
    }
}

// Implement conversion

impl<const N : usize> Into<Expr<N>> for &NDArray<N> {
    fn into(self) -> Expr<N> {
        Expr::new(
            &self.shape,
            self.sparsity().map(|s| s.to_vec()),
            (0..self.nnz()+1).collect(), // ptr
            vec![0; self.nnz()], // subj
            self.data().to_vec())
    }
}

///////////////////////////////////////////////////////////////////////////////
// SparseMatrix
///////////////////////////////////////////////////////////////////////////////

/// Represents a sparse matrix.


//impl SparseMatrix {
//    pub fn from_iterator<T>(height : usize, width : usize, it : T) -> SparseMatrix
//        where T : IntoIterator<Item=(usize,usize,f64)>
//    {
//        let mut m = SparseMatrix::zeros(height,width);
//        m.extend(it);
//        m
//    }
//    pub fn zeros(height : usize, width : usize) -> SparseMatrix { SparseMatrix{ shape : [height,width], sp : Vec::new(), data : Vec::new() } }
//    pub fn diagonal(data : Vec<f64>) -> SparseMatrix {
//        let n = data.len();
//        SparseMatrix{ shape : [n,n], sp : (0..n*n).step_by(n+1).collect(), data}
//    }
//    pub fn from_ijv(height : usize, width : usize, subi : &[usize], subj : &[usize], coefficients : Vec<f64>) -> SparseMatrix {
//        if subi.len() != subj.len() || subi.len() != coefficients.len() {
//            panic!("Mismatching vector length");
//        } 
//        if let Some(&v) = subi.iter().max() { if v >= height { panic!("Invalid subi entry"); } }
//        if let Some(&v) = subj.iter().max() { if v >= width { panic!("Invalid subj entry"); } }
//
//        let sp = subi.iter().zip(subj.iter()).map(|(&i,&j)| i*width+j).collect();
//        SparseMatrix::from_sparsity_v(height, width, sp, coefficients)
//    }
//    pub fn from_sparsity_v(height : usize, width : usize, sp : Vec<usize>, coefficients : Vec<f64>) -> SparseMatrix {
//        if sp.len() != coefficients.len() { panic!("Mismatching data dimensions"); }
//
//        if let Some(&v) = sp.iter().max() { if v >= width*height { panic!("Invalid sparsity entry"); } };
//        if sp.iter().zip(sp[1..].iter()).all(|(&i0,&i1)| i0 < i1) {
//            SparseMatrix{
//                shape : [height,width],
//                data : coefficients,
//                sp
//            }
//        } else {
//            let nnz = sp.len();
//            let mut sparsity   = vec![0usize; nnz];
//            let mut data = vec![0.0; nnz];
//            let mut perm = vec![0usize;nnz];
//            let mut ptr  = vec![0usize; usize::max(height,width)+1];
//
//            sp.iter().for_each(|i| unsafe { *ptr.get_unchecked_mut(i%width+1) += 1; } );
//            _ = ptr.iter_mut().fold(0,|c,v| { *v += c; *v });
//            sp.iter().enumerate().for_each(|(i,&si)| unsafe{ *perm.get_unchecked_mut(*ptr.get_unchecked(si%width)) = i; *ptr.get_unchecked_mut(si%width) += 1; });
//
//            ptr.iter_mut().for_each(|p| *p = 0);
//            sp.iter().for_each(|i| unsafe { *ptr.get_unchecked_mut(i%width+1) += 1; } );
//            _ = ptr.iter_mut().fold(0,|c,v| { *v += c; *v });
//            for &p in perm.iter() {
//                let i = unsafe{ *sp.get_unchecked(p) };
//                let ti = unsafe{ *ptr.get_unchecked(i/width) };
//                unsafe { 
//                    *sparsity.get_unchecked_mut(ti) = (i/width) * width + i%width;
//                    *data.get_unchecked_mut(ti) = *coefficients.get_unchecked(p);
//                    *ptr.get_unchecked_mut(i/width) += 1;
//                }
//            }
//            SparseMatrix{ shape : [height,width], sp:sparsity,data}
//        }
//    }
//    pub fn new(height : usize, width : usize, sparsity : &[[usize;2]], coefficients : Vec<f64>) -> SparseMatrix {
//        if sparsity.len() != coefficients.len() { panic!("Mismatching data dimensions"); }
//        if sparsity.iter().any(|&i| i[0] >= height || i[1] >= width) {
//            panic!("Sparsity pattern out of bounds");
//        }
//        let nnz = coefficients.len();
//
//        if sparsity.iter().zip(sparsity[1..].iter()).all(|(i0,i1)| i0[0] < i1[0] || (i0[0] == i1[0] && i0[1] < i1[1])) {
//            //sorted
//            SparseMatrix{ shape : [height,width],
//                          sp    : sparsity.iter().map(|&i| i[0]*width+i[1]).collect(),
//                          data  : coefficients.to_vec() }
//        }
//        else {
//            let mut sp   = vec![0usize; nnz];
//            let mut data = vec![0.0; nnz];
//            let mut perm = vec![0usize;nnz];
//            let mut ptr  = vec![0usize; usize::max(height,width)+1];
//
//            sparsity.iter().for_each(|i| unsafe { *ptr.get_unchecked_mut(i[1]+1) += 1; } );
//            _ = ptr.iter_mut().fold(0,|c,v| { *v += c; *v });
//            sparsity.iter().enumerate().for_each(|(i,&si)| unsafe{ *perm.get_unchecked_mut(*ptr.get_unchecked(si[1])) = i; *ptr.get_unchecked_mut(si[1]) += 1; });
//
//            ptr.iter_mut().for_each(|p| *p = 0);
//            sparsity.iter().for_each(|i| unsafe { *ptr.get_unchecked_mut(i[0]+1) += 1; } );
//            _ = ptr.iter_mut().fold(0,|c,v| { *v += c; *v });
//            for &p in perm.iter() {
//                let i = unsafe{ *sparsity.get_unchecked(p) };
//                let ti = unsafe{ *ptr.get_unchecked(i[0]) };
//                unsafe { 
//                    *sp.get_unchecked_mut(ti) = i[0] * width + i[1];
//                    *data.get_unchecked_mut(ti) = *coefficients.get_unchecked(p);
//                    *ptr.get_unchecked_mut(i[0]) += 1;
//                }
//            }
//
//            SparseMatrix{ shape : [height,width], sp,data}
//        }
//    }
//    pub fn shape(&self) -> [usize; 2] { self.shape }
//    pub fn data(&self) -> &[f64] { self.data.as_slice() }
//    pub fn sparsity(&self) -> &[usize] { self.sp.as_slice() }
//
//    pub fn transpose(&self) -> SparseMatrix {
//        let n = self.sp.len();
//        let (height,width) = (self.shape[0],self.shape[1]);
//        let mut ptr = vec![0; width+1];
//        self.sp.iter().for_each(|&i| unsafe{ *ptr.get_unchecked_mut(1 + i % width) += 1 });
//        _ = ptr.iter_mut().fold(0,|c,p| {*p += c; *p });
//
//        let mut sp = vec![0usize; n];
//        let mut data = vec![0.0; n];
//
//        for (&k,&d) in self.sp.iter().zip(self.data.iter()) {
//            let (i,j) = (k / width, k % width); 
//            let p = unsafe{ *ptr.get_unchecked(j) };
//            unsafe {
//                *sp.get_unchecked_mut(p) = j*height + i;
//                *data.get_unchecked_mut(p) = d;
//                *ptr.get_unchecked_mut(j) += 1;
//            }
//        }
//
//        SparseMatrix{
//            shape : [width,height],
//            sp,
//            data
//        }
//    }
//
//    pub fn get_flat_data(self) -> (Vec<usize>,Vec<f64>) {
//        (self.sp,self.data)
//    }
//}


impl<const N : usize> std::ops::Mul<f64> for NDArray<N> {
    type Output = NDArray<N>;
    fn mul(mut self,rhs : f64) -> Self::Output {
        self.data.iter_mut().for_each(|v| *v *= rhs);
        self
    }
}

impl<const N : usize> std::ops::Mul<NDArray<N>> for f64 {
    type Output = NDArray<N>;
    fn mul(self,mut rhs : NDArray<N>) -> Self::Output {
        rhs.data.iter_mut().for_each(|v| *v *= self );
        rhs
    }
}

impl<const N : usize> std::ops::MulAssign<f64> for NDArray<N> {
    fn mul_assign(&mut self, rhs: f64) {
        self.data.iter_mut().for_each(|v| *v *= rhs);
    } 
}


//impl std::ops::Mul<DenseMatrix> for DenseMatrix {
//    type Output = DenseMatrix;
//
//    fn mul(self,rhs : DenseMatrix) -> DenseMatrix {
//        let lhsshape = self.shape();
//        let rhsshape = rhs.shape();
//        if lhsshape[1] != rhsshape[0] { panic!("Mismatching operand dimensions"); }
//        // naive implementation:
//        
//        let shape = [lhsshape[0],rhsshape[1]];
//
//        let data = iproduct!(0..lhsshape[0],0..rhsshape[1]).map(|(i,j)| self.data[i*lhsshape[1]..].iter().zip(rhs.data[j..].iter().step_by(rhsshape[1])).map(|(&v0,&v1)| v0*v1).sum() ).collect();
//
//        DenseMatrix{
//            shape,
//            data
//        }
//    }
//}
//



// GLOBAL FUNCTIONS

pub fn dense<const N : usize>(shape : [usize;N], data : Vec<f64>) -> NDArray<N> {
    NDArray::new(shape,None,data).unwrap()
}
pub fn sparse<const N : usize>(shape : [usize;N], sp : Vec<usize>, data : Vec<f64>) -> NDArray<N> {
    NDArray::new(shape,Some(sp),data).unwrap()
}

pub fn diag<V>(data : V) -> NDArray<2> where V:Into<Vec<f64>> {
    let data = data.into();
    let dim = data.len();
    NDArray::new([dim,dim],Some((0..dim*dim).step_by(dim+1).collect()),data).unwrap()
}
pub fn speye(dim : usize) -> NDArray<2> {
    NDArray::new([dim,dim],Some((0..dim*dim).step_by(dim+1).collect()),vec![1.0; dim]).unwrap()
}

pub fn ones<const N : usize>(shape : [usize; N]) -> NDArray<N> {
    NDArray::new(shape,None,vec![1.0; shape.iter().product()]).unwrap()
}

