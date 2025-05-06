//! 
//! This module and the submodules define domain functionality. 
//!
//! Domains are divided into different types, so a model can support some subset of domains. The
//! [super::ModelAPI] object requires support for linear and ranged constraints as well as integer
//! variables. Apart from that we currently define destinct types of domains that can be supported
//! individually:
//! - [VectorDomain], vector domains of the following types
//!     - [QuadraticCone]
//!     - [SVecPSDCone] Scaled vectorized positive semidefinite cone
//!     - [GeometricMeanCone]
//!     - [ExponentialCone]
//!     - [PowerCone]
//!     - [LinearCone] which is the same as linear constraints, but in a cone form.
//! - [PSDDomain] symmetric positive semidefinite domain
//!
//! Domains are created with a builder-like logic. An object is created, then modified with various
//! properties. The builder object will be passed to [super::ModelAPI::variable] or
//! [super::ModelAPI::constraint], which will turn it into a concrete domain. 
//! Creating a variable requires a domain that explicitly or implicitly defines a shape since the
//! shape is not otherwise given - something implementing the [IntoDomain] trait. For constraints,
//! the expression will specify the dimensionality, and the domain must implement the
//! [IntoShapedDomain] trait.
//!
#![doc = include_str!("../../js/mathjax.tag")]

use iter::PermuteByMutEx;
use itertools::Either;
use super::matrix::NDArray;
use crate::utils::*;

mod psd;
mod ranged;
mod linear;

pub use psd::*;
pub use ranged::*;
pub use linear::*;

#[derive(Clone,Copy)]
pub enum AsymmetricConeType {
    Primal,
    Dual
}

#[derive(Clone,Copy)]
pub enum QuadraticCone { Normal, Rotated }
#[derive(Clone,Copy)]
pub struct SVecPSDCone();
#[derive(Clone,Copy)]
pub struct GeometricMeanCone(pub AsymmetricConeType);
#[derive(Clone,Copy)]
pub struct ExponentialCone(pub AsymmetricConeType);
#[derive(Clone)]
pub struct PowerCone(pub Vec<f64>,pub AsymmetricConeType);
#[derive(Clone,Copy)]
pub struct LinearCone(LinearDomainType);

/// Trait for all vector domains. A vector domain is a domain that is a product of identical cones. Unlike linear  
pub trait VectorDomainTrait {
    /// Check if the cone type is compatible with a given cone size. 
    fn check_conesize(&self, d : usize) -> Result<(),String>;
    fn to_conic_domain_type(&self) -> VectorDomainType;
}

/// Marker trait for anything that is a dimensioned domain traint.
pub trait DomainTrait<const N : usize> { }

/// Trait for structs that can be turned into a domain without providing a shape.
///
/// When creating a variable, the domain is an [IntoDomain], and the variable calls the
/// [IntoDomain::try_into_domain] function to turn it into a concrete domain.
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

/// Trait for anything that can be turned into a [VectorDomain].
pub trait IntoVectorDomain<const N : usize,D> where D : VectorDomainTrait {
    fn into_conic(self) -> VectorDomain<N,D>;
}

impl VectorDomainTrait for QuadraticCone {
    fn check_conesize(&self, d : usize) -> Result<(),String> {
        match self {
            QuadraticCone::Normal => if d >= 1 { Ok(()) } else { Err("Invalid dimension for quadratic cone".to_string()) },
            QuadraticCone::Rotated => if d >= 2 { Ok(()) } else { Err("Invalid dimension for rotated quadratic cone".to_string()) },
        }
    }
    fn to_conic_domain_type(&self) -> VectorDomainType {
        match self {
            QuadraticCone::Normal => VectorDomainType::QuadraticCone,
            QuadraticCone::Rotated => VectorDomainType::RotatedQuadraticCone,
        }
    }
}

impl VectorDomainTrait for SVecPSDCone {
    fn check_conesize(&self, d : usize) -> Result<(),String> {
        if d < 1 {
            return Err(format!("Size of SVecPSDCone must be at least 1, got: {}", d)); 
        }
        let n = ((((1 + d*8) as f64).sqrt()-1.0)/2.0) as usize;
        if n * (n+1)/2 != d {
            return Err(format!("Size of SVecPSDCone must correspond to the lower triangular part of a square matrix, but got: {}", d)); 
        }
        Ok(())
    }
    fn to_conic_domain_type(&self) -> VectorDomainType {
        VectorDomainType::SVecPSDCone
    }
}
impl VectorDomainTrait for GeometricMeanCone {
    fn check_conesize(&self, d : usize) -> Result<(),String> { if d >= 1 { Ok(()) } else { Err("Invalid dimension for geometric mean code".to_string()) } }
    fn to_conic_domain_type(&self) -> VectorDomainType { 
        match self.0 {
            AsymmetricConeType::Primal => VectorDomainType::GeometricMeanCone,
            AsymmetricConeType::Dual => VectorDomainType::DualGeometricMeanCone
        }
    }
}
impl VectorDomainTrait for ExponentialCone {
    fn check_conesize(&self, d : usize) -> Result<(),String> { if d != 3 { Ok(()) } else { Err("Invalid dimension for exponential code".to_string()) } }
    fn to_conic_domain_type(&self) -> VectorDomainType {
        match self.0 {
            AsymmetricConeType::Primal => VectorDomainType::ExponentialCone,
            AsymmetricConeType::Dual => VectorDomainType::DualExponentialCone
        }
    }
}
impl VectorDomainTrait for PowerCone {
    fn check_conesize(&self, d : usize) -> Result<(),String> { 
        if d >= self.0.len() { Ok(()) } else { Err("Invalid dimension for power cone".to_string()) } 
    }
    fn to_conic_domain_type(&self) -> VectorDomainType {
        match self.1 {
            AsymmetricConeType::Primal => VectorDomainType::PrimalPowerCone(self.0.clone()),
            AsymmetricConeType::Dual => VectorDomainType::DualPowerCone(self.0.clone())
        }
    }

}
impl VectorDomainTrait for LinearCone {
    fn check_conesize(&self, d : usize) -> Result<(),String> { Ok(()) }
    fn to_conic_domain_type(&self) -> VectorDomainType {
        match self.0 {
            LinearDomainType::Zero => VectorDomainType::Zero,
            LinearDomainType::Free => VectorDomainType::Free,
            LinearDomainType::NonNegative => VectorDomainType::NonNegative,
            LinearDomainType::NonPositive => VectorDomainType::NonPositive
        }
    }
}

#[derive(Clone)]
pub enum VectorDomainType {
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

impl<const N : usize> DomainTrait<N>   for LinearDomain<N> {}
impl<const N : usize,D> DomainTrait<N> for VectorDomain<N,D> where D : VectorDomainTrait {}
impl<const N : usize> DomainTrait<N>   for PSDDomain<N> {}
impl<const N : usize> DomainTrait<N>   for LinearRangeDomain<N> {}



//impl<const N : usize,D> AnyVectorDomain for VectorDomain<N,D> where D : VectorDomainTrait {
//    fn extract(&self) -> (&D,&[f64],&[usize],usize,bool) { 
//        (&self.domain_type,self.offset.as_slice(),&self.shape,self.conedim,self.is_integer) 
//    }
//}



/// A conic domain with no defined shape.
pub struct ScalableVectorDomain<D> where D : VectorDomainTrait {
    domain_type : D,
    cone_dim   : Option<usize>,
    is_integer : bool
}

/// A struct that can be turned into [VectorDomain] via [IntoDomain] or [IntoShapedDomain].
///
/// The struct acts as a factory where the domain properties can be updated. Internally in the
/// [crate::Model] object it is turned into a [VectorDomain] and consistency is checked.
pub struct VectorProtoDomain<const N : usize,D> where D : VectorDomainTrait {
    shape       : [usize;N],
    domain_type : D,
    offset      : Vec<f64>,
    cone_dim    : usize,
    is_integer  : bool,
}

/// A Conic domain defines a conic domain, shape and cone dimension for a model item.
///
/// A set of member functions makes it possible to transform the domain by changing its shape
/// sparsity, offset etc.
pub struct VectorDomain<const N : usize, D> {
    /// Cone type, including any cone parameters (like the powers for a power cone)
    domain_type : D,
    /// Offset 
    offset  : Vec<f64>,
    /// Shape if the domain
    shape   : [usize; N],
    /// Dimension in which the cones are aligned
    conedim : usize,
    /// Indicates if the domain in integer or continuous.
    is_integer : bool
}

///////////////////////////////////////////////////////////////////////////////
// ScalableVectorDomain
///////////////////////////////////////////////////////////////////////////////

impl<D> ScalableVectorDomain<D> where D : VectorDomainTrait {
    pub fn with_shape<const N : usize>(self, shape : &[usize;N]) -> VectorProtoDomain<N,D> { 
        let cone_dim = if let Some(cd) = self.cone_dim { cd } else { N.max(1) - 1 };
        VectorProtoDomain{ 
            shape : *shape,             
            cone_dim,
            offset : vec![0.0; shape.iter().product()],
            domain_type : self.domain_type,
            is_integer : self.is_integer 
       } 
    }
    pub fn with_conedim(self,cone_dim : usize) -> Self { ScalableVectorDomain{ cone_dim : Some(cone_dim), ..self } }
    pub fn integer(self)    -> Self { ScalableVectorDomain{is_integer : true,  ..self} }
    pub fn continuous(self) -> Self { ScalableVectorDomain{is_integer : false, ..self} }
}

impl<D> IntoDomain for ScalableVectorDomain<D> where D : VectorDomainTrait {
    type Result = VectorDomain<0,D>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        Err(format!("Domain size or shape cannot be determined"))
    }
}

impl<const N : usize,D> IntoShapedDomain<N> for ScalableVectorDomain<D> where D : VectorDomainTrait {
    type Result = VectorDomain<N,D>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        let cd = 
            if shape.len() == 0 {
                1
            }
            else if let Some(d) = self.cone_dim {
                *shape.get(d).ok_or_else(|| ("Invalid cone dimension index for this shape".to_string()))?
            }
            else {
                shape[N-1]
            };

        self.domain_type.check_conesize(cd)?;

        Ok(VectorDomain{
           domain_type : self.domain_type,
           offset: vec![0.0; shape.iter().product()],
           shape,
           conedim : N-1,
           is_integer : self.is_integer
        })
    }
}

///////////////////////////////////////////////////////////////////////////////
// VectorProtoDomain
///////////////////////////////////////////////////////////////////////////////

impl<const N : usize,D> VectorProtoDomain<N,D> where D : VectorDomainTrait {
    pub fn with_shape<const M : usize>(self, shape : &[usize;M]) -> VectorProtoDomain<M,D> { 
        VectorProtoDomain{ 
            shape : *shape, 
            domain_type : self.domain_type,
            offset : self.offset,
            cone_dim : self.cone_dim, 
            is_integer : self.is_integer
        }
    }
    pub fn with_conedim(self,cone_dim : usize) -> Self { VectorProtoDomain{ cone_dim, ..self }}
    pub fn with_offset(self,offset : Vec<f64>) -> Self { VectorProtoDomain{ offset, ..self }}
    pub fn integer(self) -> Self { VectorProtoDomain{ is_integer : true, ..self } }
    pub fn continuous(self) -> Self { VectorProtoDomain{ is_integer : false, ..self } }
}

impl<const N : usize,D> IntoDomain for VectorProtoDomain<N,D> where D : VectorDomainTrait {
    type Result = VectorDomain<N,D>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        if self.offset.len() != self.shape.iter().product::<usize>() {
            return Err(format!("Domain offset length does not match shape"));
        }
        if self.cone_dim >= N {
            return Err(format!("Domain has invalid cone dimension, expected 0..{}, got {}",N-1,self.cone_dim));
        }
        let cd = self.shape[self.cone_dim];
        self.domain_type.check_conesize(cd)?;


        Ok(VectorDomain{
           domain_type : self.domain_type,
           offset: self.offset,
           shape : self.shape,
           conedim : self.cone_dim,
           is_integer : self.is_integer
        })
    }
}
impl<const N : usize,D> IntoShapedDomain<N> for VectorProtoDomain<N,D> where D : VectorDomainTrait {
    type Result = VectorDomain<N,D>;
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











//impl<const N : usize,D> IntoVectorDomain<N,D> for VectorDomain<N,D> where D : VectorDomainTrait {
//    fn into_conic(self) -> Self { self }
//}
//
//impl<const N : usize,D> IntoVectorDomain<N,D> for LinearDomain<N> where D : VectorDomainTrait {
//    fn into_conic(self) -> VectorDomain<N,D> { self.to_conic() }
//}

impl<const N :usize,D> VectorDomain<N,D> where D : VectorDomainTrait {
    pub fn dissolve(self) -> (D,Vec<f64>,[usize;N],usize,bool) { (self.domain_type,self.offset,self.shape,self.conedim,self.is_integer) }
    pub fn get(&self) -> (&D,&[f64],&[usize;N],usize,bool) { (&self.domain_type,self.offset.as_slice(),&self.shape,self.conedim,self.is_integer) }
}

/// The OffsetTrait represents something that can act as an offset or bound value for a domain.
pub trait OffsetTrait {
    type Result;
    fn greater_than(self) -> Self::Result;
    fn less_than(self)    -> Self::Result;
    fn equal_to(self)     -> Self::Result;
}



////////////////////////////////////////////////////////////
// Domain constructors
////////////////////////////////////////////////////////////


/// Domain of zeros of the given shape.
pub fn zeros<const N : usize>(shape : &[usize; N]) -> LinearProtoDomain<N> { zero().with_shape(shape) }
/// Domain of values greater than the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`. If `v` is a scalar, the
///   result is a scalable domain.
pub fn greater_than<T : OffsetTrait>(v : T) -> T::Result { v.greater_than() }

/// Domain of values less than the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`. If `v` is a scalar, the
///   result is a scalable domain.
pub fn less_than<T : OffsetTrait>(v : T) -> T::Result { v.less_than() }

/// Domain of values equal to the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`. If `v` is a scalar, the
///   result is a scalable domain.
pub fn equal_to<T : OffsetTrait>(v : T) -> T::Result { v.equal_to() }


/// Domain of a single quadratic cone of unknown size. The size can subsequently be defined, or it
/// can be deduced when used in a constraint. By default the cones are aligned in the inner-most
/// dimension.
///
/// The cone has the form
/// $$
/// \\left\\{ x \\in R^n | x_1^2 \\geq \\left\\Vert x_2^2 + \\cdots + x_n^2 \\right\\Vert^2, x₁ \\geq 0 \\right\\}
/// $$
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_quadratic_cone() -> ScalableVectorDomain<QuadraticCone> { 
        ScalableVectorDomain{ domain_type: QuadraticCone::Normal, is_integer : false, cone_dim : None} 
}

/// Domain of a single rotated quadratic cone of unknown size. The size can subsequently be defined, or it
/// can be deduced when used in a constraint. By default the cones are aligned in the inner-most
/// dimension.
///
/// The cone has the form
/// $$
/// \\left\\{ x \in R^n | \\frac{1}{2} x_1 x_2 \geq \\left\\Vert x_3^2 + \\cdots + x_n^2 \\right\\Vert^2, x_1, x_2 \\geq 0 \\right\\}
/// $$
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_rotated_quadratic_cone() -> ScalableVectorDomain<QuadraticCone> { 
    ScalableVectorDomain{ domain_type: QuadraticCone::Rotated, is_integer : false, cone_dim : None} 
}

/// Domain of a single scaled vectorized PSD cone of unknown size. The size can subsequently be defined, or it
/// can be deduced when used in a constraint. By default the cones are aligned in the inner-most
/// dimension.
///
/// For an `n` dimensional positive symmetric matrix this
/// is the scaled lower triangular part of the matrix in column-major format, i.e. 
/// $$
/// \\left\\{ x \\in R^{n(n+1)/2} | \\mathrm{sMat}(x) \\in S_+^n \\right\\}
/// $$
/// where
/// $$
/// \\mathrm{sMat}(x) = \\left[ \\begin{array}{cccc} 
///   x_1            & x_2/\\sqrt{2} & \\cdots & x_n/\\sqrt{2}      \\\\
///   x_2/\\sqrt{2}  & x_n+1         & \\cdots & x_{2n-1}/\\sqrt{2} \\\\
///                  &               & \\cdots &                    \\\\
///   x_n/\\sqrt{2}  & x_{2n-1}/\\sqrt{2} & \\cdots & x_{n(n+1_/2}^2
/// \\end{array} \\right]
/// $$
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_svecpsd_cone() -> ScalableVectorDomain<SVecPSDCone> { 
    ScalableVectorDomain{ domain_type: SVecPSDCone(), is_integer : false, cone_dim : None} 
}
/// Domain of a single geometric mean cone of unknown size. The size can subsequently be defined, or it
/// can be deduced when used in a constraint. By default the cones are aligned in the inner-most
/// dimension.
///
/// The cone is defined as
/// $$
/// \\left\\{ x \\in R^n| (x_1\\cdots x_{n-1})^{1/(n-1)} |x_n|, x_1,\\ldots,x_{n-1} \\geq 0\\right\\}
/// $$
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_geometric_mean_cone() -> ScalableVectorDomain<GeometricMeanCone> { 
    ScalableVectorDomain{ domain_type: GeometricMeanCone(AsymmetricConeType::Primal), is_integer : false, cone_dim : None} 
}


/// domain of a single dual geometric mean cone of unknown size. The size can subsequently be defined, or it
/// can be deduced when used in a constraint. By default the cones are aligned in the inner-most
/// dimension.
///
/// The cone is defined as
/// $$
/// \\left\\{ x \\in R^n | (n-1)(x_1 \\cdots x_{n-1})^{1/(n-1)} |x_n|, x_1,\\ldots,x_{n-1} \\geq 0\\right\\}
/// $$
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_dual_geometric_mean_cone() -> ScalableVectorDomain<GeometricMeanCone> { 
    ScalableVectorDomain{ domain_type: GeometricMeanCone(AsymmetricConeType::Dual), is_integer : false, cone_dim : None} 
}
/// domain of a single exponential cone of unknown size. By default the cones are aligned in the inner-most
/// dimension, which must be 3
///
/// The cone is defined as
/// $$
/// \\left\\{ x \\in R^3 | x_1 \\geq x_1 e^{x_3/x_2}, x_0, x_1 \geq 0 \\right\\}
/// $$
pub fn in_exponential_cone() -> ScalableVectorDomain<ExponentialCone> { 
    ScalableVectorDomain{ domain_type: ExponentialCone(AsymmetricConeType::Primal), is_integer : false, cone_dim : None} 
}

/// Domain of a single dual exponential cone of unknown size. The result is a vector domain of size `dim`. By default the cones are aligned in the inner-most
/// dimension, which must be 3.
///
/// The cone is defined as
/// $$
/// \\left\\{ x \\in R^3 | x_1 \\geq -x_3 e^{-1} e^{x_2/x_3}, x_3 \\geq 0, x_1 \\geq 0 \\right\\}
/// $$
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_dual_exponential_cone() -> ScalableVectorDomain<ExponentialCone> { 
    ScalableVectorDomain{ domain_type: ExponentialCone(AsymmetricConeType::Dual),   is_integer : false, cone_dim : None} 
}

/// Domain of a power cone of unknown size. By default the cones are aligned in the inner-most
/// dimension.
///
/// The cone is defined as 
/// $$
/// \\left\\{ x \\in R^n | x_2^{\\beta_1} \\cdots x_k^{\\beta_k} \\geq \\sqrt{x_{k+1}^2 \\cdots x_n^2}, x_0,\\ldots, x_k \geq 0 \\right\\}
/// $$
/// # Arguments
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_power_cone(alpha : &[f64]) -> ScalableVectorDomain<PowerCone> {
    let s : f64 = alpha.iter().sum(); 
    ScalableVectorDomain{ 
        domain_type : PowerCone(alpha.iter().map(|a| a/s).collect(),AsymmetricConeType::Primal),
        is_integer  : false, 
        cone_dim    : None} }
/// Domain of a single power cone. By default the cones are aligned in the inner-most
/// dimension.
///
/// The cone is defined as:
/// $$
/// \\left\\{ x \\in R^n | (x_1/\\beta_1)^{\\beta_1} \\cdots (x_k)^{\\beta_k} \geq \\sqrt{x_{k+1}^2 \\cdots x_n^2}, x_0,\\ldots, x_k \\geq 0 \\right\\}
/// $$
///
/// # Arguments
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_dual_power_cone(alpha : &[f64]) -> ScalableVectorDomain<PowerCone> { 
    let s : f64 = alpha.iter().sum();
    ScalableVectorDomain{ 
        domain_type : PowerCone(alpha.iter().map(|a| a/s).collect(),AsymmetricConeType::Dual), 
        is_integer  : false,
        cone_dim    : None } 
}

fn in_cones<const N : usize,D>(shape : &[usize; N], cone_dim : usize,domain_type : D) -> VectorProtoDomain<N,D> where D : VectorDomainTrait {
    if cone_dim >= shape.len() {
        panic!("Invalid cone dimension");
    }
    VectorProtoDomain{domain_type,
                      offset : vec![0.0; shape.iter().product()],
                      shape:*shape,
                      cone_dim, 
                      is_integer : false}
}

/// Domain of a multiple quadratic cones.
/// 
/// See [in_quadratic_cone].
///
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_quadratic_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> VectorProtoDomain<N,QuadraticCone> { 
    in_cones(shape,conedim,QuadraticCone::Normal) 
}
/// domain of a multiple rotated quadratic cones.
/// 
/// See [in_rotated_quadratic_cone].
///
/// # arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_rotated_quadratic_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> VectorProtoDomain<N,QuadraticCone> {
    in_cones(shape,conedim,QuadraticCone::Rotated) 
}
/// Domain of a multiple scaled vectorized PSD cones.
/// 
/// See [in_svecpsd_cone].
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_svecpsd_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> VectorProtoDomain<N,SVecPSDCone> { 
    let dim = shape[conedim];
    let n = ((-1.0 + (1.0+8.0*dim as f64).sqrt())/2.0).floor() as usize;
    if n * (n+1)/2 != dim { panic!("Invalid dimension {} for svecpsd cone", dim) }

    in_cones(shape,conedim,SVecPSDCone()) 
}
/// Domain of a multiple geometric mean cones.
/// 
/// See [in_geometric_mean_cone].
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_geometric_mean_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> VectorProtoDomain<N,GeometricMeanCone> {
    in_cones(shape,conedim,GeometricMeanCone(AsymmetricConeType::Primal))
}
/// Domain of a multiple dual geometric mean cones.
///
/// See [in_dual_geometric_mean_cone]
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_dual_geometric_mean_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> VectorProtoDomain<N,GeometricMeanCone> { 
    in_cones(shape,conedim,GeometricMeanCone(AsymmetricConeType::Dual))
}
/// domain of a multiple exponential cones.
/// 
/// See [in_exponential_cone].
/// # arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_exponential_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> VectorProtoDomain<N,ExponentialCone> { 
    if let Some(&d) = shape.get(conedim) { if d != 3 { panic!("Invalid shape or exponential cone") } }
    in_cones(shape,conedim,ExponentialCone(AsymmetricConeType::Primal)) 
}
/// Domain of a multiple dual exponential cones.
/// 
/// See [in_dual_exponential_cone].
///
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_dual_exponential_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> VectorProtoDomain<N,ExponentialCone> { 
    if let Some(&d) = shape.get(conedim) { if d != 3 { panic!("Invalid shape or exponential cone") } }
    in_cones(shape,conedim,ExponentialCone(AsymmetricConeType::Dual))
}

/// Domain of a number of power cones.
///
/// See [in_power_cone].
/// # Arguments
/// - `shape` Shape of the domain
/// - `conedim` Index of the dimension in which the individual cones are alighed.
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_power_cones<const N : usize>(shape : &[usize;N], cone_dim : usize, alpha : &[f64]) -> VectorProtoDomain<N,PowerCone> {
    let alphasum : f64 = alpha.iter().sum();
    VectorProtoDomain{
        domain_type:PowerCone(alpha.iter().map(|&a| a / alphasum ).collect(),AsymmetricConeType::Primal),
        shape : *shape,
        offset:vec![0.0; shape.iter().product()],
        cone_dim,
        is_integer : false}
}

/// Domain of a number of dual power cones.
///
/// See [in_dual_power_cone].
/// # Arguments
/// - `shape` Shape of the domain
/// - `conedim` Index of the dimension in which the individual cones are alighed.
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_dual_power_cones<const N : usize>(shape : &[usize;N], cone_dim : usize, alpha : &[f64]) -> VectorProtoDomain<N,PowerCone> {
    let alphasum : f64 = alpha.iter().sum();
    VectorProtoDomain{
        domain_type:PowerCone(alpha.iter().map(|&a| a / alphasum ).collect(),AsymmetricConeType::Dual),
        shape : *shape,
        offset:vec![0.0; shape.iter().product()],
        cone_dim,
        is_integer : false}
}

/// Define a range for use with [crate::Model::constraint] and
/// [crate::Model::variable] to create ranged variables and constraints.
/// 
/// The two bounds must have same type and can be either a scalar or a vector. Shape and sparsity
/// can be defined subsequently.
/// # Arguments
/// - `lower` Lower bound
/// - `upper` Upper bound
pub fn in_range<T>(lower : T, upper : T) -> T::Result where T : IntoProtoRangeBound {
    lower.make(upper)
}
