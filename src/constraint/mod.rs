use crate::{utils::{iter::IndexIteratorExt, ShapeToStridesEx}, ConDomainTrait, ConicDomain, LinearDomain, Model, ModelItemIndex};



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
}

//======================================================
// Domain
//======================================================

/// Represents something that can be used as a domain for a constraint.
pub trait ConstraintDomain<const N : usize> {
    fn add_constraint(self, m : & mut Model, name : Option<&str>) -> Result<Constraint<N>,String>;
}
pub trait IntoConstraintDomain<const N : usize> {
    type Result : ConstraintDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String>;
}

impl IntoConstraintDomain<1> for usize {
    type Result = LinearDomain<1>;
    fn try_into_domain(self,shape : [usize;1]) -> Result<Self::Result,String> {
        [self].try_into_domain(shape)
    }
}

impl<const N : usize> IntoConstraintDomain<N> for [usize;N] {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        if shape[0] != self {
            Err(format!("Mismatching expression shape {:?} and domain shape [{:?}]",shape,self))
        }
        else {
            let totalsize = shape.iter().product();
            Ok(LinearDomain{
                shape,
                sp : None,
                ofs : crate::LinearDomainOfsType::M(vec![0; totalsize]),
                dt : crate::LinearDomainType::Free,
                is_integer : false
            })
        }
    }
}

impl<const N : usize> IntoConstraintDomain<N> for LinearDomain<N> {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        if self.shape
    }
}


impl<const N : usize> ConstraintDomain<N> for LinearDomain<N> {
    fn add_constraint(self, m : & mut Model, name : Option<&str>) -> Result<Constraint<N>,String> {
       m.linear_constraint(name, self)
    }
}

/// Implement LinearDomain as constraint domain
impl<const N : usize> ConDomainTrait<N> for LinearDomain<N> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
        m.linear_constraint(name,self)
    }

}

/// Implement ConicDomain as a constraint domain
impl<const N : usize> ConDomainTrait<N> for ConicDomain<N> {
    /// Add a constraint with expression expected to be on the top of the rs stack.
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
        m.conic_constraint(name,self)
    }
}

/// Implement a fixed-size integer array as domain for constraint, meaning unbounded with the array
/// as shape.
impl<const N : usize> ConDomainTrait<N> for &[usize;N] {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:LinearDomainOfsType::Scalar(0.0),
                                shape:*self,
                                sp:None,
                                is_integer: false})
    }
}

/// Implement integer as domain for constraint, producing a vector variable if the given size.
impl ConDomainTrait<1> for usize {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<1> {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:LinearDomainOfsType::Scalar(0.0),
                                shape:[self],
                                sp:None,
                                is_integer:false})
    }
}

impl<const N : usize> ConDomainTrait<N> for PSDDomain<N> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
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


