//! Module for Variable object and related implementations

use super::*;
use itertools::{iproduct, izip};
use super::utils;

/// A Variable object is basically a wrapper around a variable index
/// list with a shape and a sparsity pattern.
#[derive(Clone)]
pub struct Variable<const N : usize> {
    idxs     : Vec<usize>,
    sparsity : Option<Vec<usize>>,
    shape    : [usize; N]
}

impl<const N : usize> Variable<N> {
    fn numnonzeros(&self) -> usize {
        if let Some(ref sp) = self.sparsity {
            sp.len()
        }
        else {
            self.len()
        }
    }

    fn sparse_primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64], idx : & mut [[usize;N]]) -> Result<usize,String> {
        let sz = self.numnonzeros();
        if res.len() < sz || idx.len() < sz { panic!("Result array too small") }
        else {
            m.primal_var_solution(solid,self.idxs.as_slice(),res)?;
            let mut strides = [0; N];
            _ = strides.iter_mut().zip(self.shape.iter()).rev().fold(1,|c,(s,&d)| { *s = c; *s * d} );
            if let Some(ref sp) = self.sparsity {
                for (&i,ix) in sp.iter().zip(idx.iter_mut()) {
                    let _ = strides.iter().zip(ix.iter_mut()).fold(i, |i,(&s,ix)| { *ix = i / s; i % s } );
                }
            }
            else {
                for (i,ix) in idx.iter_mut().enumerate() {
                    let _ = strides.iter().zip(ix.iter_mut()).fold(i, |i,(&s,ix)| { *ix = i / s; i % s } );
                }
            }
            Ok(sz)
        }
    }
}

impl<const N : usize> ModelItem<N> for Variable<N> {
    fn len(&self) -> usize { return self.shape.iter().product(); }
    fn shape(&self) -> [usize; N] { self.shape }
    
    fn sparse_primal(&self,m : &Model,solid : SolutionType) -> Result<(Vec<f64>,Vec<[usize;N]>),String> {
        let mut nnz = vec![0.0; self.numnonzeros()];
        let dflt = [0usize; N];
        let mut idx : Vec<[usize;N]> = vec![dflt;self.numnonzeros()];
        self.sparse_primal_into(m,solid,nnz.as_mut_slice(),idx.as_mut_slice())?;
        Ok((nnz,idx))
    }

    fn primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.primal_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; });
            }
            Ok(sz)
        }
    }
    fn dual_into(&self,m : &Model,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.dual_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; })
            }
            Ok(sz)
        }
    }
}

impl ModelItemIndex<usize> for Variable<1> {
    type Output = Variable<0>;
    fn index(&self, index: usize) -> Variable<0> {
        if self.shape.len() != 1 { panic!("Cannot index into multi-dimensional variable"); }
        if let Some(ref sp) = self.sparsity {
            if let Ok(i) = sp.binary_search(&index) {
                Variable{
                    idxs     : vec![self.idxs[i]],
                    sparsity : None,
                    shape    : []
                }
            }
            else {
                Variable{
                    idxs     : vec![],
                    sparsity : Some(vec![]),
                    shape    : []
                }
            }
        }
        else {
            Variable{
                idxs     : vec![self.idxs[index]],
                sparsity : None,
                shape    : []
            }
        }
    }
}

impl<const N : usize> ModelItemIndex<&[usize; N]> for Variable<N> {
    type Output = Variable<0>;
    fn index(&self, index: &[usize; N]) -> Variable<0> {
        let index = self.shape.iter().zip(index.iter()).fold(0,|v,(&d,&i)| v*d+i);
        if let Some(ref sp) = self.sparsity {
            if let Ok(i) = sp.binary_search(&index) {
                Variable{
                    idxs     : vec![self.idxs[i]],
                    sparsity : None,
                    shape    : []
                }
            }
            else {
                Variable{
                    idxs     : vec![],
                    sparsity : Some(vec![]),
                    shape    : []
                }
            }
        }
        else {
            Variable{
                idxs     : vec![self.idxs[index]],
                sparsity : None,
                shape    : []
            }
        }
    }
}

impl ModelItemIndex<std::ops::Range<usize>> for Variable<1> {
    type Output = Variable<1>;
    fn index(&self, index: std::ops::Range<usize>) -> Variable<1> {
        let n = index.len();
        if let Some(ref sp) = self.sparsity {
            let first = match sp.binary_search(&index.start) {
                Ok(i)  => i,
                Err(i) => i
            };
            let last = match sp.binary_search(&index.start) {
                Ok(i) => i+1,
                Err(i) => i
            };

            Variable{
                idxs     : self.idxs[first..last].to_vec(),
                sparsity : Some(sp[first..last].iter().map(|&i| i - index.start).collect()),
                shape    : [n]
            }
        }
        else {
            Variable{
                idxs     : self.idxs[index].to_vec(),
                sparsity : None,
                shape    : [n]
            }
        }
    }
}

impl<const N : usize> ModelItemIndex<&[std::ops::Range<usize>; N]> for Variable<N> {
    type Output = Variable<N>;
    fn index(&self, ranges: &[std::ops::Range<usize>;N]) -> Variable<N> {
        if !ranges.iter().zip(self.shape.iter()).any(|(r,&d)| r.start > r.end || r.end <= d ) { panic!("The range is out of bounds in the the shape: {:?} in {:?}",ranges,self.shape) }

        let mut rshape = [0usize;N]; rshape.iter_mut().zip(ranges.iter()).for_each(|(rs,ra)| *rs = ra.end-ra.start);
        let mut rstrides = rshape; let _ = rstrides.iter_mut().rev().fold(1,|v,s| { let prev = *s; *s = v; v*prev});

        if let Some(ref sp) = self.sparsity {
            let mut strides = rshape.to_vec();
            let _ = strides.iter_mut().rev().fold(1,|v,s| { let prev = *s; *s = v; v*prev });

            let mut rsp   = Vec::with_capacity(sp.len());
            let mut ridxs = Vec::with_capacity(self.idxs.len());

            sp.iter().zip(self.idxs.iter())
                .for_each(|(&s,&ix)|
                          if izip!(rshape.iter(),strides.iter(),ranges.iter()).all(|(&sh,&st,ra)| { let i = (s / st) % sh; i <= ra.start && i < ra.end }) {
                              rsp.push(izip!(rshape.iter(),
                                             strides.iter(),
                                             ranges.iter(),
                                             rstrides.iter()).map(|(&sh,&st,ra,&rst)| ((s / st) % sh - ra.start) * rst).sum());
                              ridxs.push(ix);
                          });
            Variable{idxs     : ridxs,
                     sparsity : Some(rsp),
                     shape    : rshape }
        }
        else {
            //let rnum :usize = rshape.iter().product();
            let ridxs : Vec<usize> = (0..rshape.iter().product())
                .map(|i| izip!(rshape.iter(),rstrides.iter(),ranges.iter(),rstrides.iter()).map(|(&rsh,&rst,ra,&st)| (((i / rst) % rsh)+ra.start)*st ).sum::<usize>() )
                .map(|i| self.idxs[i] /*TODO: unsafe get*/)
                .collect();

            Variable{idxs : ridxs,
                     sparsity : None,
                     shape : rshape}
        }
    }
}

impl<const N : usize> Variable<N> {
    pub fn new(idxs : Vec<usize>, sparsity : Option<Vec<usize>>, shape : &[usize; N]) -> Variable<N> {
        Variable{
            idxs,
            sparsity,
            shape:*shape }
    }
    pub fn idxs(&self) -> &[usize] { self.idxs.as_slice() }
    pub fn sparsity(&self) -> Option<&[usize]> { if let Some(ref sp) = self.sparsity { Some(sp.as_slice()) } else { None }}
    pub fn shape(&self) -> &[usize] { self.shape.as_slice() }
    pub fn with_shape<const M : usize>(self, shape : &[usize; M]) -> Variable<M> {
        match self.sparsity {
            None =>
                if self.idxs.len() != shape.iter().product() {
                    panic!("Shape does not match the size");
                },
            Some(ref sp) =>
                if ! sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
                    panic!("Shape does not match the sparsity pattern");
                }
        }

        Variable{
            idxs     : self.idxs,
            sparsity : self.sparsity,
            shape:*shape
        }
    }

    pub fn with_sparsity(self, sp : Vec<usize>) -> Variable<N> {
        if sp.len() != self.idxs.len() {
            panic!("Sparsity does not match the size");
        }
        if sp.len() > 1 && ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
            panic!("Sparsity pattern is not sorted or contains duplicates");
        }
        if ! sp.last().map_or_else(|| true, |&v| v < self.shape.iter().product()) {
            panic!("Sparsity pattern does not match the shape");
        }

        Variable {
            idxs : self.idxs,
            sparsity : Some(sp),
            shape : self.shape
        }
    }

    pub fn gather(self) -> Variable<1> {
        Variable {
            shape : [self.idxs.len()],
            idxs : self.idxs,
            sparsity : None,
        }
    }

    pub fn into_column(self) -> Variable<2> {
        Variable {
            shape : [self.shape.iter().product(),1],
            idxs : self.idxs,
            sparsity : None,
        }
    }

    pub fn with_shape_and_sparsity<const M : usize>(self,shape : &[usize; M], sp : Vec<usize>) -> Variable<M> {
        if sp.len() != self.idxs.len() {
            panic!("Sparsity does not match the size");
        }
        if sp.len() > 1  && ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
            panic!("Sparsity pattern is not sorted or contains duplicates");
        }
        if sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
            panic!("Sparsity pattern does not match the shape");
        }
        Variable {
            idxs : self.idxs,
            sparsity : Some(sp),
            shape : *shape
        }
    }

    pub fn flatten(self) -> Variable<1> {
        Variable {
            idxs : self.idxs,
            sparsity : self.sparsity,
            shape : [self.shape.iter().product()]
        }
    }

    pub fn diag(self) -> Variable<1> where Self : ExprTrait<2> {
        if self.shape[0] != self.shape[1] {
            panic!("Invalid shape for operation")
        }
        if let Some(sp) = self.sparsity {
            let n = self.shape[0];
            let idxs : Vec<usize> = sp.iter().zip(self.idxs.iter()).filter(|(&i,_)| i/n == i%n).map(|v| *v.1).collect();
            let rsp = if idxs.len() < n {
                Some(sp.iter().filter(|&i| i/n == i%n).map(|i| i / n).collect())
            } 
            else {
                None
            };

             Variable {
                idxs,
                sparsity : rsp,
                shape : [n]
            }
        }
        else {
            Variable {
                idxs : self.idxs.iter().step_by(self.shape[0]+1).cloned().collect(),
                sparsity : None,
                shape : [self.shape[0]]
            }
        }
    }

    // // Other functions to be implemented:
    // ///// Take the diagonal element of a square, cube,... variable
    // //pub fn diag(&self) -> Variable
    // //pub fn into_diag(&self) -> Variable
    pub fn slice(&self, ranges : &[std::ops::Range<usize>; N]) -> Variable<N> {
        if ranges.iter().zip(self.shape.iter()).find(|(r,&d)| r.start >= r.end || r.end > d).is_some() {
            panic!("Slice out of bounds");
        }

        let mut rshape = [0usize; N];
        let mut rstrides= [8usize; N];
        rshape.iter_mut().zip(ranges).for_each(|(res,r)| *res = r.end-r.start);

        rshape.iter().rev().scan(1,|v,d| { let w=*v; *v = *v*d; Some(w) } ).zip(rstrides.iter_mut().rev()).for_each(|(s,t)| *t = s);

        let mut strides = [0usize; N];
        strides.iter_mut().zip(self.shape.iter()).rev().fold(1,|v,(st,&d)| { *st = v; d*v });

        if let Some(ref sp) = self.sparsity {
            let mut rsp   = Vec::with_capacity(sp.len());
            let mut ridxs = Vec::with_capacity(self.idxs.len());

            sp.iter().zip(self.idxs.iter())
                .for_each(|(&s,&ix)|
                          if izip!(rshape.iter(),strides.iter(),ranges.iter()).all(|(&sh,&st,ra)| { let i = (s / st) % sh; i <= ra.start && i < ra.end }) {
                              rsp.push(izip!(rshape.iter(),
                                             strides.iter(),
                                             ranges.iter(),
                                             rstrides.iter()).map(|(&sh,&st,ra,&rst)| ((s / st) % sh - ra.start) * rst).sum());
                              ridxs.push(ix);
                          });
            Variable{idxs     : ridxs,
                     sparsity : Some(rsp),
                     shape    : rshape }
        }
        else {            

            let ridxs : Vec<usize> = 
                (0..rshape.iter().product())
                .map(|i| izip!(rshape.iter(),
                               rstrides.iter(),
                               ranges.iter(),
                               strides.iter()).map(|(&rsh,&rst,ra,&st)| (((i / rst) % rsh)+ra.start)*st ).sum::<usize>() )
                .map(|i| self.idxs[i] /*TODO: unsafe get*/)
                .collect();
            println!("Variable::slice({:?}): idxs={:?} -> {:?}, shape = {:?}",ranges,self.idxs,ridxs,rshape);
            Variable{idxs : ridxs,
                     sparsity : None,
                     shape : rshape}
        }
    }
    // pub fn index(&self, idx : &[usize]) -> Variable {
    //     if idx.len() != self.shape.len() { panic!("The range does not match the shape") }
    //     if idx.iter().zip(self.shape.iter()).any(|(&i,&d)| i >= d ) { panic!("The range does not match the shape") }

    //     let (index,_) = idx.iter().zip(self.shape.iter()).fold((0,1),|(r,stride),(&i,&d)| (r+i*stride,stride*d));
    //     if let Some(ref sp) = self.sparsity {
    //         if let Ok(i) = sp.binary_search(&index) {
    //             Variable{
    //                 idxs : vec![self.idxs[i]],
    //                 sparsity : None,
    //                 shape : vec![] }
    //         }
    //         else {
    //             Variable{
    //                 idxs : vec![],
    //                 sparsity : Some(vec![]),
    //                 shape : vec![] }
    //         }
    //     }
    //     else {
    //         Variable{
    //             idxs : vec![self.idxs[index]],
    //             sparsity : None,
    //             shape : vec![] }
    //     }

    // }
    pub fn stack(dim : usize, xs : &[&Variable<N>]) -> Variable<N> {
        if ! xs.iter().zip(xs[1..].iter())
            .all(|(v0,v1)| utils::shape_eq_except(v0.shape.as_slice(),v1.shape.as_slice(),dim)) {
                panic!("Operands have mismatching shapes");
            }

        let ddim : usize = xs.iter().map(|v| v.shape[dim]).sum();
        //let n      = xs.len();
        let rnelm  = xs.iter().map(|v| v.idxs.len()).sum();
        let mut rshape = xs[0].shape.clone(); rshape[dim] = ddim;
        let nd = rshape.len();

        if dim == 0 {
            let mut ridxs : Vec<usize> = Vec::with_capacity(rnelm);
            for v in xs {
                ridxs.extend(v.idxs.iter());
            }
            let rsp = if rnelm < rshape.iter().product() {
                let mut ofs : usize = 0;
                let mut rsp : Vec<usize> = Vec::with_capacity(rnelm);
                for v in xs {
                    if let Some(ref sp) = v.sparsity {
                        rsp.extend(sp.iter().map(|&i| i+ofs));
                    }
                    else {
                        rsp.extend(ofs..ofs+v.idxs.len());
                    }
                    ofs += v.shape.iter().product::<usize>();
                }

                Some(rsp)
            }
            else {
                None
            };

            Variable{
                idxs     : ridxs,
                sparsity : rsp,
                shape    : rshape }
        }
        else if rshape.iter().product::<usize>() == rnelm {
            let _d0 : usize = rshape[0..dim].iter().product();
            let d1 : usize = rshape[dim];
            let d2 : usize = if dim < nd - 1 { rshape[dim+1..].iter().product() } else { 1 };

            let mut ridxs : Vec<usize> = vec![0; rnelm];

            let stride = d1*d2;
            let mut ofs = 0;

            for v in xs {
                let vd1 = v.shape[dim];
                let chunksize = vd1*d2;
                for (src,dst) in v.idxs.chunks(chunksize).zip(ridxs.chunks_mut(stride)) {
                    dst[ofs..ofs+chunksize].clone_from_slice(src);
                }

                ofs += v.shape[dim];
            }
            Variable{idxs     : ridxs,
                     sparsity : None,
                     shape    : rshape}
        }
        else {
            let d0 : usize = rshape[0..dim].iter().product();
            let d1 : usize = rshape[dim];
            let d2 : usize = if dim < nd - 1 { rshape[dim+1..].iter().product() } else { 1 };

            let mut dofs : usize = 0;
            let mut ridxs = Vec::with_capacity(rnelm);
            let mut rsp   = Vec::with_capacity(rnelm);
            for v in xs {
                let vd1 = v.shape[dim];
                ridxs.extend(v.idxs.iter());
                if let Some(ref sp) = v.sparsity {
                    rsp.extend(sp.iter().map(|&i| { let (i0,i1,i2) = (i/(vd1*d2),(i/d2)%vd1,i%d2); (i0*d1+i1+dofs)*d2+i2 }))
                }
                else {
                    rsp.extend(iproduct!(0..d0,dofs..dofs+vd1,0..d2).map(|(i0,i1,i2)| (i0*d1+i1)*d2+i2))
                }
                dofs += v.shape[dim];
            }

            let mut perm : Vec<usize> = (0..rnelm).collect();
            perm.sort_by_key(|&p| *unsafe{rsp.get_unchecked(p)});
            Variable{
                idxs     : perm.iter().map(|&p| unsafe{*ridxs.get_unchecked(p)}).collect(),
                sparsity : Some(perm.iter().map(|&p| unsafe{*rsp.get_unchecked(p)}).collect()),
                shape    : rshape }
        }
    }
    pub fn vstack(xs : &[&Variable<N>]) -> Variable<N> { Self::stack(0,xs) }
    pub fn hstack(xs : &[&Variable<N>]) -> Variable<N> { Self::stack(1,xs) }

    pub fn transpose(self) -> Self where Self : ExprTrait<2> {
        let mut shape = [0usize; N];
        shape[0] = self.shape[1];
        shape[1] = self.shape[0];
        if let Some(sp) = self.sparsity {
            let mut xsp : Vec<(usize,usize)> = sp.iter().zip(self.idxs.iter()).map(|(&i,&ni)| (( i % self.shape[1]) * self.shape[0] + i / self.shape[1], ni) ).collect();
            xsp.sort();
            let rsp = xsp.iter().map(|v| v.0).collect();
            let rnidxs = xsp.iter().map(|v| v.1).collect();

            Variable{
                sparsity : Some(rsp),
                idxs : rnidxs,
                shape
            }
        } else {
            let mut idxs = vec![0usize; self.idxs.len()];

            for (t,&s) in izip!(idxs.iter_mut(),
                                (0..self.shape[1])
                                    .flat_map(|i| self.idxs[i..].iter().step_by(self.shape[1]))) {
                *t = s;
            }

            Variable{
                idxs,
                sparsity : None,
                shape
            }
        }
    }
}

impl<const N : usize> Variable<N> {
    fn eval_common(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&self.shape,
                                                  self.idxs.len(),
                                                  self.idxs.len());
        rptr.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        rsubj.clone_from_slice(self.idxs.as_slice());
        rcof.fill(1.0);
        if let (Some(rsp),Some(sp)) = (rsp,&self.sparsity) {
            rsp.clone_from_slice(sp.as_slice())
        }
    }
}

impl<const N : usize> super::ExprTrait<N> for Variable<N> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.eval_common(rs,ws,xs);
    }
}



