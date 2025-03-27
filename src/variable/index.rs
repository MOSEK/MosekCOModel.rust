//! Implements indexing into variables.
//!
//!
use std::rc::Rc;
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use itertools::izip;

use crate::{ModelItemIndex, Variable};
use crate::utils::{iter::*,*};

pub trait ModelItemIndexElement                 { fn expand(self, d : usize) -> Range<usize>; }
impl ModelItemIndexElement for usize            { fn expand(self, d : usize) -> Range<usize> { Range { start: self.min(d), end: self.min(d) } } }
impl ModelItemIndexElement for Range<usize>     { fn expand(self, d : usize) -> Range<usize> { Range{ start: self.start.min(d), end : self.end.min(d) } } }
impl ModelItemIndexElement for RangeFrom<usize> { fn expand(self, d : usize) -> Range<usize> { Range{ start: self.start.min(d), end : d } } }
impl ModelItemIndexElement for RangeTo<usize>   { fn expand(self, d : usize) -> Range<usize> { Range{ start : 0, end: self.end.min(d) } } }
impl ModelItemIndexElement for RangeFull        { fn expand(self, d : usize) -> Range<usize> { Range{ start : 0, end: d } } }


impl ModelItemIndex<Variable<1>> for Range<usize> {
    type Output = Variable<1>;
    fn index(self,obj : &Variable<1>) -> Self::Output { 
        [self].index(obj)
    }
}
impl ModelItemIndex<Variable<1>> for RangeTo<usize> {
    type Output = Variable<1>;
    fn index(self,obj : &Variable<1>) -> Self::Output { 
        [0..self.end].index(obj)
    }
}

impl ModelItemIndex<Variable<1>> for RangeFrom<usize> {
    type Output = Variable<1>;
    fn index(self,obj : &Variable<1>) -> Self::Output { 
        [self.start..obj.shape[0]].index(obj)
    }
}

impl ModelItemIndex<Variable<1>> for RangeFull {
    type Output = Variable<1>;
    fn index(self,obj : &Variable<1>) -> Self::Output { 
        obj.clone()
    }
}

impl<I1,I2> ModelItemIndex<Variable<2>> for (I1,I2)
    where 
        I1 : ModelItemIndexElement,
        I2 : ModelItemIndexElement 
{
    type Output = Variable<2>;
    fn index(self,obj : &Variable<2>) -> Self::Output { 
        obj.index([self.0.expand(obj.shape[0]),
                   self.1.expand(obj.shape[1])])
    }
}

impl<I1,I2,I3> ModelItemIndex<Variable<3>> for (I1,I2,I3)
    where 
        I1 : ModelItemIndexElement,
        I2 : ModelItemIndexElement,
        I3 : ModelItemIndexElement 
{
    type Output = Variable<3>;
    fn index(self,obj : &Variable<3>) -> Self::Output { 
        obj.index([self.0.expand(obj.shape[0]),
                  self.1.expand(obj.shape[1]),
                  self.2.expand(obj.shape[2])]) 
    }
}

impl<I1,I2,I3,I4> ModelItemIndex<Variable<4>> for (I1,I2,I3,I4)
    where 
        I1 : ModelItemIndexElement,
        I2 : ModelItemIndexElement,
        I3 : ModelItemIndexElement,
        I4 : ModelItemIndexElement 
{
    type Output = Variable<4>;
    fn index(self,obj : &Variable<4>) -> Self::Output { 
        obj.index([self.0.expand(obj.shape[0]),
                   self.1.expand(obj.shape[1]),
                   self.2.expand(obj.shape[2]),
                   self.3.expand(obj.shape[3])]) 
    }
}

impl<I1,I2,I3,I4,I5> ModelItemIndex<Variable<5>> for (I1,I2,I3,I4,I5)
    where 
        I1 : ModelItemIndexElement,
        I2 : ModelItemIndexElement,
        I3 : ModelItemIndexElement,
        I4 : ModelItemIndexElement,
        I5 : ModelItemIndexElement 
{
    type Output = Variable<5>;
    fn index(self,obj : &Variable<5>) -> Self::Output { 
        obj.index([self.0.expand(obj.shape[0]),
                   self.1.expand(obj.shape[1]),
                   self.2.expand(obj.shape[2]),
                   self.3.expand(obj.shape[3]),
                   self.4.expand(obj.shape[4])]) 
    }
}


impl ModelItemIndex<Variable<1>> for usize {
    type Output = Variable<0>;
    fn index(self, v : &Variable<1>) -> Variable<0> {
        if v.shape.len() != 1 { panic!("Cannot index into multi-dimensional variable"); }
        if let Some(ref sp) = v.sparsity {
            if let Ok(i) = sp.binary_search(&self) {
                Variable{
                    idxs: Rc::new(vec![v.idxs[i]]),
                    sparsity: None,
                    shape : []
                }
            }
            else {
                Variable{
                    idxs: Rc::new(vec![]),
                    sparsity: Some(Rc::new(vec![])),
                    shape : []
                }
            }
        }
        else {
            Variable{
                idxs : Rc::new(vec![v.idxs[self]]),
                sparsity : None,
                shape : []
            }
        }
    }
}

impl<const N : usize> ModelItemIndex<Variable<N>> for [usize; N] {
    type Output = Variable<0>;
    fn index(self, v : &Variable<N>) -> Variable<0> {
        let index = v.shape.iter().zip(self.iter()).fold(0,|v,(&d,&i)| v*d+i);
        if let Some(ref sp) = v.sparsity {
            if let Ok(i) = sp.binary_search(&index) {
                Variable{
                    idxs : Rc::new(vec![v.idxs[i]]),
                    sparsity : None,
                    shape : []
                }
            }
            else {
                Variable{
                    idxs : Rc::new(vec![]),
                    sparsity : Some(Rc::new(vec![])),
                    shape : []
                }
            }
        }
        else {
            Variable{
                idxs : Rc::new(vec![v.idxs[index]]),
                sparsity : None,
                shape : []
            }
        }
    }
}

impl<const N : usize> ModelItemIndex<Variable<N>> for [Range<usize>; N] {
    type Output = Variable<N>;
    fn index(self, v : &Variable<N>) -> Variable<N> {
        if !self.iter().zip(v.shape.iter()).any(|(r,&d)| r.start > r.end || r.end <= d ) { panic!("The range is out of bounds in the the shape: {:?} in {:?}",self,v.shape) }

        let mut rshape = [0usize;N]; rshape.iter_mut().zip(self.iter()).for_each(|(rs,ra)| *rs = ra.end-ra.start);
        let rstrides = rshape.to_strides();
        let strides = v.shape.to_strides();


        if let Some(ref sp) = v.sparsity {
            let mut rsp   = Vec::with_capacity(sp.len());
            let mut ridxs = Vec::with_capacity(v.idxs.len());

            sp.iter().zip(v.idxs.iter())
                .for_each(|(&s,&ix)|
                          if izip!(rshape.iter(),strides.iter(),self.iter()).all(|(&sh,&st,ra)| { let i = (s / st) % sh; i <= ra.start && i < ra.end }) {
                              rsp.push(izip!(rshape.iter(),
                                             strides.iter(),
                                             self.iter(),
                                             rstrides.iter()).map(|(&sh,&st,ra,&rst)| ((s / st) % sh - ra.start) * rst).sum());
                              ridxs.push(ix);
                          });
            Variable{
                idxs     : Rc::new(ridxs), 
                sparsity : Some(Rc::new(rsp)),
                shape : rshape }
        }
        else {
            fn addvec<const N : usize>(lhs : &[usize;N], rhs : &[usize;N]) -> [usize;N] {
                let mut r = [0usize;N]; 
                r.iter_mut().zip(lhs.iter().zip(rhs.iter())).for_each(|(d,(&s0,&s1))| *d = s0+s1 );
                r
            }

            let mut offset = [0usize; N]; offset.iter_mut().zip(self.iter()).for_each(|(o,i)| *o = i.start);
            let ridxs : Vec<usize> = 
                rshape.index_iterator()
                    .map(|index| strides.to_linear(&addvec(&index,&offset)))
                    .map(|i| v.idxs[i] /*TODO: unsafe get*/)
                    .collect();

            Variable{
                idxs     : Rc::new(ridxs),
                sparsity : None,
                shape    : rshape }
        }
    }
}

