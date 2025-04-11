use crate::*;
use crate::utils::{iter::IndexIteratorExt, ShapeToStridesEx};



/// A Constraint object is a wrapper around an array of constraint
/// indexes and a shape. Note that constraint objects are never sparse.
#[derive(Clone)]
pub struct Constraint<const N : usize> {
    pub(crate) idxs  : Vec<usize>,
    pub(crate) shape : [usize; N]
}

impl<const N : usize> Constraint<N> {
    pub fn index<I>(&self, idx : I) -> I::Output where I : ModelItemIndex<Self> {
        idx.index(self)
    }
    pub fn reshape<const M : usize>(self, shape : &[usize; M]) -> Constraint<M> {
        if shape.iter().product::<usize>() != self.shape.iter().product::<usize>() {
            panic!("Mismatching shapes");
        }
        Constraint{
            idxs : self.idxs,
            shape : *shape
        }
    }
    // TODO implement more stacking and reshaping functions here, similar to Variable
}

//======================================================
// Domain use
//======================================================

/// Represents something that can be used as a domain for a constraint.
pub trait ConstraintDomain<const N : usize,M> {
    type Result;
    fn add_constraint(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String>;
}

/// Implement LinearDomain as constraint domain
impl<const N : usize,M> ConstraintDomain<N,M> for LinearDomain<N> {
    fn add_constraint(self, m : & mut Model, name : Option<&str>) -> Result<Constraint<N>,String> {
        m.linear_constraint(name,self)
    }

}

/// Implement ConicDomain as a constraint domain
impl<const N : usize> ConstraintDomain<N> for ConicDomain<N> {
    /// Add a constraint with expression expected to be on the top of the rs stack.
    fn add_constraint(self, m : & mut Model, name : Option<&str>) -> Result<Constraint<N>,String> {
        m.conic_constraint(name,self)
    }
}

impl<const N : usize> ConstraintDomain<N> for PSDDomain<N> {
    fn add_constraint(self, m : & mut Model, name : Option<&str>) -> Result<Constraint<N>,String> {
        m.psd_constraint(name,self)
    }
}















impl ModelItemIndex<Constraint<1>> for usize {
    type Output = Constraint<0>;
    fn index(self, v : &Constraint<1>) -> Constraint<0> {
        if v.shape.len() != 1 { panic!("Cannot index into multi-dimensional variable"); }
        Constraint{
            idxs     : vec![v.idxs[self]],
            shape    : []
        }
    }
}

impl<const N : usize> ModelItemIndex<Constraint<N>> for [usize; N] {
    type Output = Constraint<0>;
    fn index(self, v : &Constraint<N>) -> Constraint<0> {
        let index = v.shape.iter().zip(self.iter()).fold(0,|v,(&d,&i)| v*d+i);
        Constraint{
            idxs     : vec![v.idxs[index]],
            shape    : []
        }
    }
}

impl ModelItemIndex<Constraint<1>> for std::ops::Range<usize> {
    type Output = Constraint<1>;
    fn index(self, v : &Constraint<1>) -> Constraint<1> {
        let n = self.len();
        Constraint{
            idxs     : v.idxs[self].to_vec(),
            shape    : [n]
        }
    }
}

impl<const N : usize> ModelItemIndex<Constraint<N>> for [std::ops::Range<usize>; N] {
    type Output = Constraint<N>;
    fn index(self, v : &Constraint<N>) -> Constraint<N> {
        if !self.iter().zip(v.shape.iter()).any(|(r,&d)| r.start > r.end || r.end <= d ) { panic!("The range is out of bounds in the the shape: {:?} in {:?}",self,v.shape) }

        let mut rshape = [0usize;N]; rshape.iter_mut().zip(self.iter()).for_each(|(rs,ra)| *rs = ra.end-ra.start);
        let strides = v.shape.to_strides();

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

        Constraint{idxs     : ridxs,
                 shape    : rshape}
    }
}


