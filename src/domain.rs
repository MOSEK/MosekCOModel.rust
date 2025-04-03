use iter::PermuteByMutEx;
use itertools::Either;
use super::matrix::NDArray;
use crate::utils::*;

pub enum LinearDomainType {
    NonNegative,
    NonPositive,
    Zero,
    Free
}

pub enum ConicDomainType {
    QuadraticCone,
    RotatedQuadraticCone,
    SVecPSDCone,
    GeometricMeanCone,
    DualGeometricMeanCone,
    ExponentialCone,
    DualExponentialCone,
    PrimalPowerCone(Vec<f64>),
    DualPowerCone(Vec<f64>),
    // linear types
    NonNegative,
    NonPositive,
    Zero,
    Free
}

#[derive(Debug)]
pub enum LinearDomainOfsType {
    Scalar(f64),
    M(Vec<f64>)
}

pub trait DomainTrait<const N : usize> {
}

impl<const N : usize> DomainTrait<N> for LinearDomain<N> {}
impl<const N : usize> DomainTrait<N> for ConicDomain<N> {}
impl<const N : usize> DomainTrait<N> for PSDDomain<N> {}

/// Trait for structs that can be turned into a domain without imposing a shape.
pub trait IntoDomain {
    type Result;
    fn try_into_domain(self) -> Result<Self::Result,String>;
}

/// Trait for structs that given a shape can be turned into a domain and either checking the shape
/// or scaling to conform to the shape. 
pub trait IntoShapedDomain<const N : usize> {
    type Result : DomainTrait<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String>;
}

pub trait IntoConicDomain<const N : usize> {
    fn into_conic(self) -> ConicDomain<N>;
}

///////////////////////////////////////////////////////////////////////////////
// ScalableLinearDomain
///////////////////////////////////////////////////////////////////////////////


/// A struct defining a scalable proto-domain, i.e. a domain that can be scaled up to any shape.
pub struct ScalableLinearDomain {
    domain_type : LinearDomainType,
    offset : f64,
    is_integer : bool,
}

impl ScalableLinearDomain {
    pub fn integer(self) -> ScalableLinearDomain { ScalableLinearDomain{ is_integer : true, ..self } }
    pub fn continuous(self) -> ScalableLinearDomain { ScalableLinearDomain{ is_integer : false, ..self } }
    pub fn with_shape<const N : usize>(self,shape : &[usize;N]) -> LinearProtoDomain<N> {
        LinearProtoDomain{
            shape : *shape,
            domain_type : self.domain_type,
            offset : Either::Left(self.offset),
            sparsity : None,
            is_integer : self.is_integer
        }
    }
    pub fn with_offset(self,offset : f64) -> ScalableLinearDomain { ScalableLinearDomain{ offset, ..self } }
    // TODO: Check sparsity pattern indexes against shape?
    pub fn with_shape_and_sparsity<const N : usize>(self, shape : &[usize;N], sparsity : &[[usize;N]]) -> LinearProtoDomain<N> {
        let st = shape.to_strides();        
        LinearProtoDomain{
            shape : *shape,
            domain_type : self.domain_type,
            offset : Either::Left(self.offset),
            sparsity : Some(sparsity.iter().map(|i| st.to_linear(i)).collect()),
            is_integer : self.is_integer
        }
    }
}

impl<const N : usize> IntoShapedDomain<N> for ScalableLinearDomain {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        Ok(LinearDomain{
            shape,
            offset : vec![self.offset; shape.iter().product()],
            sparsity    : None,
            domain_type : self.domain_type,
            is_integer  : self.is_integer
        })        
    }
}
impl IntoDomain for ScalableLinearDomain {
    type Result = LinearDomain<0>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        Ok(LinearDomain{
            shape : [],
            offset : vec![self.offset],
            sparsity : None,
            domain_type    : self.domain_type,
            is_integer : self.is_integer
        })
    }
}

///////////////////////////////////////////////////////////////////////////////
// LinearProtoDomain 
///////////////////////////////////////////////////////////////////////////////

/// A domain structure that can be turned into a finalized domain. Note that turning it into a
/// finalized domain also performs consistency checks that may fail.
pub struct LinearProtoDomain<const N : usize> {
    shape : [usize;N],
    domain_type : LinearDomainType,
    offset : Either<f64,Vec<f64>>,
    sparsity : Option<Vec<usize>>,
    is_integer : bool
}

impl<const N : usize> LinearProtoDomain<N> {
    pub fn integer(self) -> Self { LinearProtoDomain{ is_integer : true, ..self } }
    pub fn continuous(self) -> Self { LinearProtoDomain{ is_integer : false, ..self } }
    pub fn with_offset(self,offset : Vec<f64>) -> Self { LinearProtoDomain{ offset : Either::Right(offset), ..self } }
    // TODO: Check sparsity pattern indexes against shape?
    pub fn with_shape<const M : usize>(self, shape : &[usize;M]) -> LinearProtoDomain<M> {
        LinearProtoDomain{ 
            shape : *shape,
            domain_type : self.domain_type,
            offset : self.offset,
            sparsity : self.sparsity,
            is_integer: self.is_integer
        }
    }

    pub fn with_sparsity(self, sparsity : &[[usize;N]]) -> Self {
        let st = self.shape.to_strides();
        let sparsity = sparsity.iter().map(|i| st.to_linear(i)).collect();

        LinearProtoDomain{
            sparsity : Some(sparsity),
            ..self
        }
    }

    pub fn with_shape_and_sparsity<const M : usize>(self, shape : &[usize;M], sparsity : &[[usize;M]]) -> LinearProtoDomain<M> {
        let st = shape.to_strides();
        let sparsity = sparsity.iter().map(|i| st.to_linear(i)).collect();
        LinearProtoDomain{ 
            shape : *shape, 
            sparsity : Some(sparsity),
            domain_type : self.domain_type,
            offset : self.offset,
            is_integer : self.is_integer
        }
    }
}

impl<const N : usize> IntoDomain for LinearProtoDomain<N> {
    type Result = LinearDomain<N>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        let st = self.shape.to_strides();                
        let totalsize : usize = self.shape.iter().product();
        if let Some(sp) = &self.sparsity {
            if let Some((a,b)) = sp.iter().zip(sp[1..].iter()).find(|(a,b)| a >= b) {
                return Err(format!("Sparsity pattern unsorted or contains duplicates: {:?} and {:?}", st.to_index(*a), st.to_index(*b)));
            }
            if let Some(i) = sp.iter().max() {
                if *i >= totalsize {
                    return Err(format!("Element in sparsity pattern is out of bounds: {:?}", st.to_index(*i)));
                }
            }
            if let Either::Right(offset) = &self.offset {
                if sp.len() != offset.len() {
                    return Err(format!("Sparsity and offset lengths do not match"));
                }
            }


        }
        else if let Either::Right(v) = &self.offset {
            if totalsize != v.len() {
                return Err(format!("Offset and shape lengths do not match"));
            }
        }

        let offset = 
            match self.offset {
                Either::Right(v) => v,
                Either::Left(v) => {
                    let n = self.sparsity.as_ref().map(|v| v.len()).unwrap_or(totalsize);
                    vec![v; n]
                }
            };

        Ok(LinearDomain{
            shape       : self.shape,
            offset,
            sparsity    : self.sparsity,
            domain_type : self.domain_type,
            is_integer  : self.is_integer
        })
    }
}

impl<const N : usize> IntoShapedDomain<N> for LinearProtoDomain<N> {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        let res = IntoDomain::try_into_domain(self)?;
        if res.shape != shape {
            Err(format!("Mismatched domain shape: {:?} vs {:?}",res.shape,shape))
        }
        else {
            Ok(res)
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
// ScalableConicDomain
///////////////////////////////////////////////////////////////////////////////

pub struct ScalableConicDomain {
    domain_type : ConicDomainType,
    cone_dim   : Option<usize>,
    is_integer : bool
}

impl ScalableConicDomain {
    pub fn with_shape<const N : usize>(self, shape : &[usize;N]) -> ConicProtoDomain<N> { 
        let cone_dim = if let Some(cd) = self.cone_dim { cd } else { N.max(1) - 1 };
        ConicProtoDomain{ 
            shape : *shape,             
            cone_dim,
            offset : vec![0.0; shape.iter().product()],
            domain_type : self.domain_type,
            is_integer : self.is_integer 
       } 
    }
    pub fn with_conedim(self,cone_dim : usize) -> Self { ScalableConicDomain{ cone_dim : Some(cone_dim), ..self } }
    pub fn integer(self)    -> Self { ScalableConicDomain{is_integer : true,  ..self} }
    pub fn continuous(self) -> Self { ScalableConicDomain{is_integer : false, ..self} }
}

impl IntoDomain for ScalableConicDomain {
    type Result = ConicDomain<0>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        Err(format!("Domain size or shape cannot be determined"))
    }
}

impl<const N : usize> IntoShapedDomain<N> for ScalableConicDomain {
    type Result = ConicDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        match self.domain_type {
            ConicDomainType::QuadraticCone          => if shape[N-1] < 1 { return Err(format!("Quadratic cones must be at least 1 element, got {}", shape[N-1])); },
            ConicDomainType::RotatedQuadraticCone   => if shape[N-1] < 2 { return Err(format!("Rotated quadratic cones must be at least 2 element, got {}", shape[N-1])); },
            ConicDomainType::GeometricMeanCone      => if shape[N-1] < 1 { return Err(format!("Geometric mean cones must be at least 1 element, got {}", shape[N-1])); },
            ConicDomainType::DualGeometricMeanCone  => if shape[N-1] < 1 { return Err(format!("Dual geometric mean cones must be at least 1 element, got {}", shape[N-1])); },
            ConicDomainType::ExponentialCone        => if shape[N-1] != 3 { return Err(format!("Exponential cones must be at least 1 element, got {}", shape[N-1])); },
            ConicDomainType::DualExponentialCone    => if shape[N-1] != 3 { return Err(format!("Dual exponential cones must be at least 1 element, got {}", shape[N-1])); },
            ConicDomainType::PrimalPowerCone(ref v) => if shape[N-1] < v.len() { return Err(format!("Power cones must be at least {} element, got {}", v.len(),shape[N-1])); },
            ConicDomainType::DualPowerCone(ref v)   => if shape[N-1] < v.len() { return Err(format!("Dual power cones must be at least {} element, got {}", v.len(),shape[N-1])); },
            ConicDomainType::SVecPSDCone            => {
                let d = shape[N-1];
                if d < 1 {
                    return Err(format!("Size of SVecPSDCone must be at least 1, got: {}", shape[N-1])); 
                }
                let n = ((((1 + d*8) as f64).sqrt()-1.0)/2.0) as usize;
                if n * (n+1)/2 != d {
                    return Err(format!("Size of SVecPSDCone must correspond to the lower triangular part of a square matrix, but got: {}", shape[N-1])); 
                }
            },
            _ => { /*rest can be whatever size*/ }
        }

        Ok(ConicDomain{
           domain_type : self.domain_type,
           offset: vec![0.0; shape.iter().product()],
           shape,
           conedim : N-1,
           is_integer : self.is_integer
        })
    }
}

///////////////////////////////////////////////////////////////////////////////
// ConicProtoDomain
///////////////////////////////////////////////////////////////////////////////

pub struct ConicProtoDomain<const N : usize> {
    shape       : [usize;N],
    domain_type : ConicDomainType,
    offset      : Vec<f64>,
    cone_dim    : usize,
    is_integer  : bool,
}

impl<const N : usize> ConicProtoDomain<N> {
    pub fn with_shape<const M : usize>(self, shape : &[usize;M]) -> ConicProtoDomain<M> { 
        ConicProtoDomain{ 
            shape : *shape, 
            domain_type : self.domain_type,
            offset : self.offset,
            cone_dim : self.cone_dim, 
            is_integer : self.is_integer
        }
    }
    pub fn with_conedim(self,cone_dim : usize) -> Self { ConicProtoDomain{ cone_dim, ..self }}
    pub fn with_offset(self,offset : Vec<f64>) -> Self { ConicProtoDomain{ offset, ..self }}
    pub fn integer(self) -> Self { ConicProtoDomain{ is_integer : true, ..self } }
    pub fn continuous(self) -> Self { ConicProtoDomain{ is_integer : false, ..self } }
}

impl<const N : usize> IntoDomain for ConicProtoDomain<N> {
    type Result = ConicDomain<N>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        if self.offset.len() != self.shape.iter().product() {
            return Err(format!("Domain offset length does not match shape"));
        }
        if self.cone_dim >= N {
            return Err(format!("Domain has invalid cone dimension, expected 0..{}, got {}",N-1,self.cone_dim));
        }
        let cd = self.shape[self.cone_dim];
        match self.domain_type {
            ConicDomainType::QuadraticCone          => if cd < 1       { return Err(format!("Quadratic cones must be at least 1 element, got {}", cd)); },
            ConicDomainType::RotatedQuadraticCone   => if cd < 2       { return Err(format!("Rotated quadratic cones must be at least 2 element, got {}", cd)); },
            ConicDomainType::GeometricMeanCone      => if cd < 1       { return Err(format!("Geometric mean cones must be at least 1 element, got {}", cd)); },
            ConicDomainType::DualGeometricMeanCone  => if cd < 1       { return Err(format!("Dual geometric mean cones must be at least 1 element, got {}", cd)); },
            ConicDomainType::ExponentialCone        => if cd != 3      { return Err(format!("Exponential cones must be at least 1 element, got {}", cd)); },
            ConicDomainType::DualExponentialCone    => if cd != 3      { return Err(format!("Dual exponential cones must be at least 1 element, got {}", cd)); },
            ConicDomainType::PrimalPowerCone(ref v) => if cd < v.len() { return Err(format!("Power cones must be at least {} element, got {}", v.len(),cd)); },
            ConicDomainType::DualPowerCone(ref v)   => if cd < v.len() { return Err(format!("Dual power cones must be at least {} element, got {}", v.len(),cd)); },
            ConicDomainType::SVecPSDCone            => {
                if cd < 1 {
                    return Err(format!("Size of SVecPSDCone must be at least 1, got: {}", cd)); 
                }
                let n = ((((1 + cd*8) as f64).sqrt()-1.0)/2.0) as usize;
                if n * (n+1)/2 != cd {
                    return Err(format!("Size of SVecPSDCone must correspond to the lower triangular part of a square matrix, but got: {}", cd)); 
                }
            },
            _ => { /*rest can be whatever size*/ }
        }
        Ok(ConicDomain{
           domain_type : self.domain_type,
           offset: self.offset,
           shape : self.shape,
           conedim : self.cone_dim,
           is_integer : self.is_integer
        })
    }
}
impl<const N : usize> IntoShapedDomain<N> for ConicProtoDomain<N> {
    type Result = ConicDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        let dom = IntoDomain::try_into_domain(self)?;
        if dom.shape != shape {
            Err(format!("Domain shape did not match the expected shape: {:?} vs {:?}",dom.shape,shape))
        }
        else {
            Ok(dom)
        }
    }
}


impl IntoDomain for usize {
    type Result = LinearDomain<1>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        Ok(LinearDomain { domain_type: LinearDomainType::Free, offset: vec![0.0;self], shape: [self], sparsity: None, is_integer: false })
    }
}

impl IntoShapedDomain<1> for usize {
    type Result = LinearDomain<1>;
    fn try_into_domain(self,shape : [usize;1]) -> Result<Self::Result,String> {
        if shape[0] != self {
            Err("Domain does not match the given shape".to_string())
        }
        else {
            Ok(LinearDomain { domain_type: LinearDomainType::Free, offset: vec![0.0;self], shape: [self], sparsity: None, is_integer: false })
        }
    }
}

impl<const N : usize> IntoDomain for [usize;N] {
    type Result = LinearDomain<N>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        let n = self.iter().product();
        Ok(LinearDomain { domain_type: LinearDomainType::Free, offset: vec![0.0; n], shape: self, sparsity: None, is_integer: false })
    }
}

impl<const N : usize> IntoShapedDomain<N> for [usize;N] {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        if shape != self {
            Err("Domain shape did not match the expected shape".to_string())
        }
        else {
            let n = self.iter().product();
            Ok(LinearDomain { domain_type: LinearDomainType::Free, offset: vec![0.0; n], shape: self, sparsity: None, is_integer: false })
        }
    }
}


impl<const N : usize> IntoDomain for &[usize;N] {
    type Result = LinearDomain<N>;
    fn try_into_domain(self) -> Result<Self::Result,String> { IntoDomain::try_into_domain(*self) }
}

impl<const N : usize> IntoShapedDomain<N> for &[usize;N] {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        IntoShapedDomain::try_into_domain(*self,shape)
    }
}

///////////////////////////////////////////////////////////////////////////////
// ScalablePSDDomain 
///////////////////////////////////////////////////////////////////////////////

pub struct ScalablePSDDomain {
    cone_dims : Option<(usize,usize)>
}

impl ScalablePSDDomain {
    pub fn with_conedims(self,conedim0 : usize, conedim1 : usize) -> Self { ScalablePSDDomain{ cone_dims : Some((conedim0,conedim1)) }}
    pub fn with_dim(self,dim : usize) -> PSDProtoDomain<2> { 
        PSDProtoDomain{
            shape : [dim,dim],
            cone_dims : self.cone_dims,
        }
    }
    pub fn with_shape<const N : usize>(self, shape : &[usize;N]) -> PSDProtoDomain<N> { 
        let cone_dims = if let Some(cd) = self.cone_dims { cd } else { (N.max(2)-2,N.max(2)-1) };
        PSDProtoDomain{shape:*shape, cone_dims : Some(cone_dims) } 
    }
}

impl IntoDomain for ScalablePSDDomain {
    type Result = PSDDomain<0>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        Err("PSD Domain has no shape".to_string())
    }
}
impl<const N : usize> IntoShapedDomain<N> for ScalablePSDDomain {
    type Result = PSDDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        if N < 2 {
            return Err("PSD Domains by be at least two-dimensional".to_string());
        }

        let conedims = 
            if let Some((d0,d1)) = self.cone_dims {
                if d0 == d1 {
                    return Err("PSD domains cone dimensions must be different".to_string());
                }
                else if d0.max(d1) >= N {
                    return Err("Invalid cone dimensions for PSD domain".to_string());
                }
                (d0,d1)
            }
            else {
                (N-2,N-1)
            };

        Ok(PSDDomain{ shape, conedims })
    }
}

///////////////////////////////////////////////////////////////////////////////
// PSDProtoDomain 
///////////////////////////////////////////////////////////////////////////////

pub struct PSDProtoDomain<const N : usize> {
    shape : [usize;N],
    cone_dims : Option<(usize,usize)>
}

impl<const N : usize> PSDProtoDomain<N> {
    pub fn dissolve(self) -> ([usize;N],Option<(usize,usize)>) { (self.shape,self.cone_dims) }
    pub fn with_conedims(self,conedim0 : usize, conedim1 : usize) -> Self { PSDProtoDomain{ cone_dims : Some((conedim0,conedim1)), ..self }}
    pub fn with_shape<const M : usize>(self, shape : [usize;M]) -> PSDProtoDomain<M> { 
        PSDProtoDomain{
            shape, 
            cone_dims : self.cone_dims
        } 
    }
}

impl<const N : usize> IntoDomain for PSDProtoDomain<N> {
type Result = PSDDomain<N>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        if N < 2 {
            return Err(format!("PSD Domains by be at least two-dimensional"));
        }

        let conedims = 
            if let Some((d0,d1)) = self.cone_dims {
                if d0 == d1 {
                    return Err(format!("PSD domains cone dimensions must be different"));
                }
                else if d0.max(d1) >= N {
                    return Err(format!("Invalid cone dimensions for PSD domain"));
                }
                else if self.shape[d0] != self.shape[d1] {
                    return Err(format!("Cone dimensions for PSD domain must have same size"));
                }
                (d0,d1)
            }
            else {
                (N-2,N-1)
            };

        Ok(PSDDomain{ shape : self.shape, conedims })
    }
}

impl<const N : usize> IntoShapedDomain<N> for PSDProtoDomain<N> {
    type Result = PSDDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        let res = IntoDomain::try_into_domain(self)?;
        if res.shape == shape {
            Ok(res)
        }
        else {
            Err(format!("Domain shape did not match the expected shape: {:?} vs {:?}",res.shape,shape))
        }
    }
}


impl IntoDomain for f64 {
    type Result = LinearDomain<0>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        Ok(LinearDomain { domain_type: LinearDomainType::Zero, offset: vec![self], shape: [], sparsity: None, is_integer: false })
    }
}

impl<const N : usize> IntoShapedDomain<N> for f64 {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        Ok(LinearDomain { domain_type: LinearDomainType::Zero, offset: vec![self; shape.iter().product()], shape, sparsity: None, is_integer: false })
    }
}

impl IntoDomain for Vec<f64> {
    type Result = LinearDomain<1>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        let n = self.len();
        Ok(LinearDomain { domain_type: LinearDomainType::Zero, offset: self, shape: [n], sparsity: None, is_integer: false })
    }
}

impl<const N : usize> IntoShapedDomain<N> for Vec<f64> {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        let n = self.len();
        if n != shape.iter().product() {
            Err(format!("Vector cannot be reshaped into a {:?} object",shape))
        }
        else {
            Ok(LinearDomain { domain_type: LinearDomainType::Zero, offset: self, shape, sparsity: None, is_integer: false })
        }
    }
}

impl IntoDomain for &[f64] {
    type Result = LinearDomain<1>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        IntoDomain::try_into_domain(self.to_vec())
    }
}

impl<const N : usize> IntoShapedDomain<N> for &[f64] {
    type Result = LinearDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        IntoShapedDomain::try_into_domain(self.to_vec(),shape)
    }
}








//-----------------------------------------------------------------------------
// RangedDomain
//-----------------------------------------------------------------------------

pub struct ScalableLinearRange {
    lower : f64,
    upper : f64,
    is_integer : bool,
}

impl ScalableLinearRange {
    pub fn with_shape<const N : usize>(self,shape: &[usize;N]) -> ProtoLinearRange<N> {
        ProtoLinearRange{
            shape : *shape,
            lower : Either::Left(self.lower),
            upper : Either::Left(self.upper),
            sparsity : None,
            is_integer : self.is_integer,
        }
    }
    pub fn with_shape_and_sparsity<const N : usize>(self, shape : &[usize;N], sparsity : &[[usize; N]]) -> ProtoLinearRange<N> {
        let st = shape.to_strides();
        let sparsity = sparsity.iter().map(|index| st.to_linear(index)).collect();
        ProtoLinearRange{
            shape : *shape,
            lower : Either::Left(self.lower),
            upper : Either::Left(self.upper),
            sparsity : Some(sparsity),
            is_integer : self.is_integer,
        }
    }
    pub fn with_sparsity<const N : usize>(self, sparsity : &[[usize; N]]) -> ProtoLinearRange<N> {
        let shape = 
            sparsity.iter()
                .fold([0usize;N],|mut shape,index| { 
                    shape.iter_mut().zip(index.iter()).for_each(|(s,&i)| *s = (*s).max(i)); shape 
                });

        let st = shape.to_strides();
        let sparsity = sparsity.iter().map(|index| st.to_linear(index)).collect();
        ProtoLinearRange{
            shape,
            lower      : Either::Left(self.lower),
            upper      : Either::Left(self.upper),
            sparsity   : Some(sparsity),
            is_integer : self.is_integer,
        }
    }

    pub fn integer(self) -> Self {
        ScalableLinearRange{
            is_integer : true,
            ..self
        }
    }
    pub fn continuous(self) -> Self {
        ScalableLinearRange{
            is_integer : false,
            ..self
        }
    }
}

pub struct ProtoLinearRange<const N : usize> {
    shape      : [usize;N],
    lower      : Either<f64,Vec<f64>>,
    upper      : Either<f64,Vec<f64>>,
    sparsity   : Option<Vec<usize>>,
    is_integer : bool,
}

impl<const N : usize> ProtoLinearRange<N> {
    pub fn with_shape<const M : usize>(self,shape : &[usize;M]) -> ProtoLinearRange<M> {
        ProtoLinearRange{
            shape : *shape,
            lower : self.lower,
            upper : self.upper,
            sparsity : self.sparsity,
            is_integer : self.is_integer
        }
    }
    pub fn with_sparsity(self, sparsity : &[[usize;N]]) -> ProtoLinearRange<N> {
        let st = self.shape.to_strides();
        let sparsity = sparsity.iter().map(|index| st.to_linear(index)).collect();
        ProtoLinearRange{
            sparsity : Some(sparsity), 
            ..self
        }
    }
    pub fn integer(self) -> Self {
        ProtoLinearRange{
            is_integer : true,
            ..self
        }
    }
    pub fn continuous(self) -> Self {
        ProtoLinearRange{
            is_integer : false,
            ..self
        }
    }
}

pub trait IntoProtoRangeBound {
    type Result;
    fn make(self, other : Self) -> Self::Result;
}

impl IntoProtoRangeBound for f64 {
    type Result = ScalableLinearRange;
    fn make(self, other : Self) -> Self::Result {
        ScalableLinearRange{
            lower : self,
            upper : other,
            is_integer : false
        }
    }
}

impl IntoProtoRangeBound for Vec<f64> {
    type Result = ProtoLinearRange<1>;
    fn make(self, other : Self) -> Self::Result {
        ProtoLinearRange{
            shape      : [self.len()],
            lower      : Either::Right(self),
            upper      : Either::Right(other),
            sparsity   : None,
            is_integer : false,
        }
    }
}

impl IntoProtoRangeBound for &[f64] {
    type Result = ProtoLinearRange<1>;
    fn make(self, other : Self) -> Self::Result { self.to_vec().make(other.to_vec()) }
}

pub fn range<T>(lower : T, upper : T) -> T::Result where T : IntoProtoRangeBound {
    lower.make(upper)
}

pub trait IntoLinearRange {
    type Result;
    fn into_range(self) -> Result<Self::Result,String>;
}

pub trait IntoShapedLinearRange<const N : usize> {
    fn into_range(self, shape : [usize;N]) -> Result<LinearRangeDomain<N>,String>;
}

impl IntoLinearRange for ScalableLinearRange {
    type Result = LinearRangeDomain<0>;
    fn into_range(self) -> Result<Self::Result,String> {
        Ok(LinearRangeDomain{
            shape : [],
            lower : vec![self.lower],
            upper : vec![self.upper], 
            sparsity : None,
            is_integer : self.is_integer,
        })
    }
}

impl<const N : usize> IntoShapedLinearRange<N> for ScalableLinearRange {
    fn into_range(self,shape: [usize;N]) -> Result<LinearRangeDomain<N>,String> {
        let n = shape.iter().product();
        Ok(LinearRangeDomain{
            shape,
            lower : vec![self.lower; n],
            upper : vec![self.upper; n], 
            sparsity : None,
            is_integer : self.is_integer,
        })
    }
}

impl<const N : usize> IntoShapedLinearRange<N> for ProtoLinearRange<N> {
    fn into_range(self, shape : [usize;N]) -> Result<LinearRangeDomain<N>,String> {
        if shape != self.shape {
            return Err("Domain shape does not match the expected shape.".to_string());
        }

        IntoLinearRange::into_range(self)
    }
}

impl<const N : usize> IntoLinearRange for ProtoLinearRange<N> {
    type Result = LinearRangeDomain<N>;
    fn into_range(self) -> Result<Self::Result,String> {
        let nelm = 
            if let Some(sp) = self.sparsity.as_ref() {
                if sp.iter().zip(sp[1..].iter()).any(|(a,b)| a >= b) {
                    return Err("Sparsity pattern is unsorted or contains duplicates.".to_string());
                }
                else if sp.iter().max().map(|&v| v >= self.shape.iter().product()).unwrap_or(false) {
                    return Err("Sparsity entry out of bounds".to_string());
                }
                sp.len()
            }
            else {
                self.shape.iter().product()
            };

        let lower =
            match self.lower {
                Either::Left(v) => { vec![v; nelm] },
                Either::Right(v) => { 
                    if v.len() != nelm {
                        return Err("Lower bound contains incorrect number of entries".to_string());
                    }
                    else {
                        v
                    }
                }
            };
        let upper = 
            match self.upper {
                Either::Left(v) => { vec![v; nelm] },
                Either::Right(v) => { 
                    if v.len() != nelm {
                        return Err("Upper bound contains incorrect number of entries".to_string());
                    }
                    else {
                        v
                    }
                }
            };

        Ok(LinearRangeDomain{
            shape : self.shape,
            lower,
            upper,
            sparsity : self.sparsity,
            is_integer : self.is_integer,
        })
    }
}

pub struct LinearRangeDomain<const N : usize> {
    pub shape      : [usize;N],
    pub lower      : Vec<f64>,
    pub upper      : Vec<f64>,
    pub sparsity   : Option<Vec<usize>>,
    pub is_integer : bool,
}

impl<const N : usize> LinearRangeDomain<N> {
    pub fn dissolve(self) -> ([usize;N],Vec<f64>,Vec<f64>,Option<Vec<usize>>,bool) {
        (self.shape,
         self.lower,
         self.upper,
         self.sparsity,
         self.is_integer)
    }
    pub fn dense(self) -> Self {
        if let Some(sp) = self.sparsity {
            let n = self.shape.iter().product();
            let mut lower = vec![0.0; n];
            let mut upper = vec![0.0; n];
            lower.permute_by_mut(sp.as_slice()).zip(self.lower.iter()).for_each(|(t,s)| *t = *s);
            upper.permute_by_mut(sp.as_slice()).zip(self.upper.iter()).for_each(|(t,s)| *t = *s);

            LinearRangeDomain{ 
                lower,
                upper,
                sparsity : None,
                ..self
            }
        }
        else { 
            self
        }
    }
}
//-----------------------------------------------------------------------------
// LinearDomain
//-----------------------------------------------------------------------------



/// A Linear domain defines bounds, shape and sparsity for a model item.
///
/// A set of member functions makes it possible to transform the domain by changing its shape
/// sparsity, offset etc. 
pub struct LinearDomain<const N : usize> {
    /// Bound type
    domain_type : LinearDomainType,
    /// Offset type, either a scalar or a vector that matches the shape/sparsity
    offset     : Vec<f64>,
    /// Shape of the domain, which will also define the shape of the model item
    shape : [usize; N],
    /// Sparsity - this is used to create sparsity for the model item
    sparsity    : Option<Vec<usize>>,
    /// Indicates if the domain in integer or continuous.
    is_integer : bool,
}

/// A Conic domain defines a conic domain, shape and cone dimension for a model item.
///
/// A set of member functions makes it possible to transform the domain by changing its shape
/// sparsity, offset etc. 
pub struct ConicDomain<const N : usize> {
    /// Cone type
    domain_type : ConicDomainType,
    /// Offset 
    /// nm
    offset  : Vec<f64>,
    /// Shape if the domain
    shape   : [usize; N],
    /// Dimension in which the cones are aligned
    conedim : usize,
    /// Indicates if the domain in integer or continuous.
    is_integer : bool
}

/// A semidefinite conic domain.
pub struct PSDDomain<const N : usize> {
    /// Shape of the domain - note that two dimensions must be the same to allow symmetry.
    shape    : [usize; N],
    /// The two cone dimensions where the cones are aligned.
    conedims : (usize,usize)
}


/////////////////////////////////////////////////////////////////////
// Domain implementations

impl<const N : usize> IntoConicDomain<N> for ConicDomain<N> {
    fn into_conic(self) -> ConicDomain<N> {
        self
    }
}

impl<const N : usize> IntoConicDomain<N> for LinearDomain<N> {
    fn into_conic(self) -> ConicDomain<N> {
        self.to_conic()
    }
}

impl<const N :usize> PSDDomain<N> {
    pub fn dissolve(self) -> ([usize;N],(usize,usize)) { (self.shape,self.conedims) }
}
impl<const N :usize> ConicDomain<N> {
    pub fn dissolve(self) -> (ConicDomainType,Vec<f64>,[usize;N],usize,bool) { (self.domain_type,self.offset,self.shape,self.conedim,self.is_integer) }
    pub fn get(&self) -> (&ConicDomainType,&[f64],&[usize;N],usize,bool) { (&self.domain_type,self.offset.as_slice(),&self.shape,self.conedim,self.is_integer) }
}
impl<const N : usize> LinearDomain<N> {
    pub fn dissolve(self) -> (LinearDomainType,Vec<f64>,Option<Vec<usize>>,[usize;N],bool) { (self.domain_type,self.offset,self.sparsity,self.shape,self.is_integer) }
    /// Create a [ConicDomain] equivalent to the linear domain.
    pub fn to_conic(self) -> ConicDomain<N> {
        let conedim = N.max(1) - 1;
        let domain_type = match self.domain_type {
            LinearDomainType::Zero => ConicDomainType::Zero,
            LinearDomainType::Free => ConicDomainType::Free,
            LinearDomainType::NonPositive => ConicDomainType::NonPositive,
            LinearDomainType::NonNegative => ConicDomainType::NonNegative
        };
        ConicDomain {
            domain_type,
            offset : self.offset,
            shape : self.shape,
            conedim,
            is_integer : self.is_integer
        }
    }

    /// Extract domain values.
    pub fn extract(self) -> (LinearDomainType,Vec<f64>,[usize;N],Option<Vec<usize>>,bool) {
        (self.domain_type,self.offset,self.shape,self.sparsity,self.is_integer)
    }
    /// Make a sparse domain into a dense, adding zeros as necessary.
    pub fn into_dense(self) -> Self {
        if let Some(sp) = &self.sparsity {
            let offset = {
                let mut offset = vec![0.0; self.shape.iter().product()];
                offset.permute_by_mut(sp.as_slice()).zip(self.offset.iter()).for_each(|(o,&v)| *o = v );
                offset
            };
            
            LinearDomain{
                offset,
                sparsity : None,
                ..self
            }
        }
        else {
            self
        }
    }
}

/// The OffsetTrait represents something that can act as an offset or bound value for a domain.
pub trait OffsetTrait {
    type Result;
    fn greater_than(self) -> Self::Result;
    fn less_than(self)    -> Self::Result;
    fn equal_to(self)     -> Self::Result;
}

/// Make `f64` work as a scalar offset value.
impl OffsetTrait for f64 {
    type Result = ScalableLinearDomain;
    fn greater_than(self) -> Self::Result { ScalableLinearDomain{ domain_type : LinearDomainType::NonNegative,  offset:self, is_integer : false } }
    fn less_than(self)    -> Self::Result { ScalableLinearDomain{ domain_type : LinearDomainType::NonPositive,  offset:self, is_integer : false } }
    fn equal_to(self)     -> Self::Result { ScalableLinearDomain{ domain_type : LinearDomainType::Zero,         offset:self, is_integer : false } }
}

/// Let `Vec<f64>` act as a vector offset value.
impl OffsetTrait for Vec<f64> {
    type Result = LinearProtoDomain<1>;
    fn greater_than(self) -> Self::Result { let n = self.len(); LinearProtoDomain{ domain_type : LinearDomainType::NonNegative, offset:Either::Right(self), shape:[n], sparsity : None, is_integer : false } }
    fn less_than(self)    -> Self::Result { let n = self.len(); LinearProtoDomain{ domain_type : LinearDomainType::NonPositive, offset:Either::Right(self), shape:[n], sparsity : None, is_integer : false } }
    fn equal_to(self)     -> Self::Result { let n = self.len(); LinearProtoDomain{ domain_type : LinearDomainType::Zero,        offset:Either::Right(self), shape:[n], sparsity : None, is_integer : false } }
}

/// Let `&[f64]` act as a vector offset value.
impl OffsetTrait for &[f64] {
    type Result = LinearProtoDomain<1>;

    fn greater_than(self) -> Self::Result { self.to_vec().greater_than() }
    fn less_than(self)    -> Self::Result { self.to_vec().less_than() } 
    fn equal_to(self)     -> Self::Result { self.to_vec().equal_to() }
}

impl<const N : usize> OffsetTrait for NDArray<N> {
    type Result = LinearProtoDomain<N>;
    fn greater_than(self) -> Self::Result { let (shape,sparsity,data) = self.dissolve(); LinearProtoDomain{ domain_type : LinearDomainType::NonNegative, offset:Either::Right(data), shape, sparsity, is_integer : false } }
    fn less_than(self)    -> Self::Result { let (shape,sparsity,data) = self.dissolve(); LinearProtoDomain{ domain_type : LinearDomainType::NonPositive, offset:Either::Right(data), shape, sparsity, is_integer : false } }
    fn equal_to(self)     -> Self::Result { let (shape,sparsity,data) = self.dissolve(); LinearProtoDomain{ domain_type : LinearDomainType::Zero,        offset:Either::Right(data), shape, sparsity, is_integer : false } }
}

////////////////////////////////////////////////////////////
// Domain constructors
////////////////////////////////////////////////////////////

/// Unbounded scalar domain.
pub fn unbounded() -> ScalableLinearDomain { ScalableLinearDomain{offset : 0.0, domain_type : LinearDomainType::Free, is_integer : false } }
/// Scalar domain of nonnegative values
pub fn nonnegative() -> ScalableLinearDomain { ScalableLinearDomain{ offset : 0.0, domain_type : LinearDomainType::NonNegative, is_integer : false } }
/// Scalar domain of nonpositive values
pub fn nonpositive() -> ScalableLinearDomain { ScalableLinearDomain{ offset : 0.0, domain_type : LinearDomainType::NonPositive, is_integer : false } }
/// Scalar domain of zeros
pub fn zero() -> ScalableLinearDomain { ScalableLinearDomain{ offset : 0.0, domain_type : LinearDomainType::Zero, is_integer : false } }


/// Domain of zeros of the given shape.
pub fn zeros<const N : usize>(shape : &[usize; N]) -> LinearProtoDomain<N> { zero().with_shape(shape) }
/// Domain of values greater than the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`
pub fn greater_than<T : OffsetTrait>(v : T) -> T::Result { v.greater_than() }
/// Domain of values less than the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`
pub fn less_than<T : OffsetTrait>(v : T) -> T::Result { v.less_than() }
/// Domain of values equal to the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`
pub fn equal_to<T : OffsetTrait>(v : T) -> T::Result { v.equal_to() }






/// Domain of a single quadratic cone ofsize `dim`. The result is a vector domain of size `dim`.
pub fn in_quadratic_cone()                     -> ScalableConicDomain { ScalableConicDomain{ domain_type: ConicDomainType::QuadraticCone,         is_integer : false, cone_dim : None} }
/// Domain of a single rotated quadratic cone ofsize `dim`. The result is a vector domain of size `dim`.
pub fn in_rotated_quadratic_cone()             -> ScalableConicDomain { ScalableConicDomain{ domain_type: ConicDomainType::RotatedQuadraticCone,  is_integer : false, cone_dim : None} }
/// Domain of a single scaled vectorized PSD cone of size `dim`, where `dim = n(n+1)/2` for some integer `n` The result is a vector domain of size `dim`.
pub fn in_svecpsd_cone()                       -> ScalableConicDomain { ScalableConicDomain{ domain_type: ConicDomainType::SVecPSDCone,           is_integer : false, cone_dim : None} }
/// Domain of a single geometric mean cone ofsize `dim`. The result is a vector domain of size `dim`.
pub fn in_geometric_mean_cone()           -> ScalableConicDomain { ScalableConicDomain{ domain_type: ConicDomainType::GeometricMeanCone,     is_integer : false, cone_dim : None} }
/// domain of a single dual geometric mean cone ofsize `dim`. the result is a vector domain of size `dim`.
pub fn in_dual_geometric_mean_cone()      -> ScalableConicDomain { ScalableConicDomain{ domain_type: ConicDomainType::DualGeometricMeanCone, is_integer : false, cone_dim : None} }
/// domain of a single exponential cone of size 3. the result is a vector domain of size 3.
pub fn in_exponential_cone()              -> ScalableConicDomain { ScalableConicDomain{ domain_type: ConicDomainType::ExponentialCone,       is_integer : false, cone_dim : None} }
/// Domain of a single dual exponential cone ofsize `dim`. The result is a vector domain of size `dim`.
pub fn in_dual_exponential_cone()         -> ScalableConicDomain { ScalableConicDomain{ domain_type: ConicDomainType::DualExponentialCone,   is_integer : false, cone_dim : None} }
/// Domain of a power cone of unknown size.
///
/// # Arguments
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_power_cone(alpha : &[f64]) -> ScalableConicDomain {
    let s : f64 = alpha.iter().sum(); 
    ScalableConicDomain{ 
        domain_type : ConicDomainType::PrimalPowerCone(alpha.iter().map(|a| a/s).collect()),
        is_integer  : false, 
        cone_dim    : None} }
/// Domain of a single power cone.
///
/// # Arguments
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_dual_power_cone(alpha : &[f64]) -> ScalableConicDomain { 
    let s : f64 = alpha.iter().sum();
    ScalableConicDomain{ 
        domain_type : ConicDomainType::DualPowerCone(alpha.iter().map(|a| a/s).collect()), 
        is_integer  : false,
        cone_dim    : None } 
}

fn in_cones<const N : usize>(shape : &[usize; N], cone_dim : usize,domain_type : ConicDomainType) -> ConicProtoDomain<N> {
    if cone_dim >= shape.len() {
        panic!("Invalid cone dimension");
    }
    ConicProtoDomain{domain_type,
                     offset : vec![0.0; shape.iter().product()],
                     shape:*shape,
                     cone_dim, 
                     is_integer : false}
}

/// Domain of a multiple quadratic cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_quadratic_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicProtoDomain<N> { in_cones(shape,conedim,ConicDomainType::QuadraticCone) }
/// domain of a multiple rotated quadratic cones.
/// 
/// # arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_rotated_quadratic_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicProtoDomain<N> { in_cones(shape,conedim,ConicDomainType::RotatedQuadraticCone) }
/// Domain of a multiple scaled vectorized PSD cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_svecpsd_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicProtoDomain<N> { 
    let dim = shape[conedim];
    let n = ((-1.0 + (1.0+8.0*dim as f64).sqrt())/2.0).floor() as usize;
    if n * (n+1)/2 != dim { panic!("Invalid dimension {} for svecpsd cone", dim) }

    in_cones(shape,conedim,ConicDomainType::SVecPSDCone) 
}
/// Domain of a multiple geometric mean cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_geometric_mean_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicProtoDomain<N> { in_cones(shape,conedim,ConicDomainType::GeometricMeanCone) }
/// Domain of a multiple dual geometric mean cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_dual_geometric_mean_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicProtoDomain<N> { in_cones(shape,conedim,ConicDomainType::DualGeometricMeanCone) }
/// domain of a multiple exponential cones.
/// 
/// # arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_exponential_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicProtoDomain<N> { 
    if let Some(&d) = shape.get(conedim) { if d != 3 { panic!("Invalid shape or exponential cone") } }
    in_cones(shape,conedim,ConicDomainType::GeometricMeanCone) 
}
/// Domain of a multiple dual exponential cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_dual_exponential_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicProtoDomain<N> { 
    if let Some(&d) = shape.get(conedim) { if d != 3 { panic!("Invalid shape or exponential cone") } }
    in_cones(shape,conedim,ConicDomainType::DualGeometricMeanCone) 
}

/// Domain of a number of power cones.
///
/// # Arguments
/// - `shape` Shape of the domain
/// - `conedim` Index of the dimension in which the individual cones are alighed.
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_power_cones<const N : usize>(shape : &[usize;N], cone_dim : usize, alpha : &[f64]) -> ConicProtoDomain<N> {
    let alphasum : f64 = alpha.iter().sum();
    ConicProtoDomain{
        domain_type:ConicDomainType::PrimalPowerCone(alpha.iter().map(|&a| a / alphasum ).collect()),
        shape : *shape,
        offset:vec![0.0; shape.iter().product()],
        cone_dim,
        is_integer : false}
}

/// Domain of a number of dual power cones.
///
/// # Arguments
/// - `shape` Shape of the domain
/// - `conedim` Index of the dimension in which the individual cones are alighed.
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_dual_power_cones<const N : usize>(shape : &[usize;N], cone_dim : usize, alpha : &[f64]) -> ConicProtoDomain<N> {
    let alphasum : f64 = alpha.iter().sum();
    ConicProtoDomain{
        domain_type:ConicDomainType::PrimalPowerCone(alpha.iter().map(|&a| a / alphasum ).collect()),
        shape : *shape,
        offset:vec![0.0; shape.iter().product()],
        cone_dim,
        is_integer : false}
}

/// Domain of a single symmetric positive semidefinite cones. For constraints this defines the constraint 
/// ```math 
/// 1/2 (E+E') ≽ 0
/// ```
/// If the expression is already symmetric, this simply means `E≽0`. For variables it is simply the
/// symmetric positive semidefinite cone.
///
///
/// # Arguments
/// - `dim` - Dimension of the PSD cone.
pub fn in_psd_cone() -> ScalablePSDDomain {
    ScalablePSDDomain{
        cone_dims : None
    }
}
/// Domain of a multiple symmetric positive semidefinite cones. The cones are aligned in the two
/// dimensions give by `conedim1` and `conedim2`. For constraints this means that each slice in `conedim1,
/// conedim2` defines the constraint
/// ```math 
/// 1/2 (E+E') ≽ 0
/// ```
/// If the expression is already symmetric, this simply means `E≽0`.
///
/// For variables is produces a stack of positive symmetric semidefinite cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone, where `shape[conedim1]==shape[conedim2]`.
/// - `conedim1` - first cone dimension
/// - `conedim2` - second cone dimension. `conedim2` must be different from `conedim1`.
pub fn in_psd_cones<const N : usize>(shape : &[usize; N]) -> PSDProtoDomain<N> {
    PSDProtoDomain{
        shape : *shape,
        cone_dims : None
    }
}
