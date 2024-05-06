use itertools::izip;

use super::matrix::{DenseNDArray,Matrix};
use super::utils::perm_iter;

pub enum LinearDomainType {
    NonNegative,
    NonPositive,
    Zero,
    Free
}

pub enum ConicDomainType {
    QuadraticCone,
    RotatedQuadraticCone,
    GeometricMeanCone,
    DualGeometricMeanCone,
    ExponentialCone,
    DualExponentialCone,
    PrimalPowerCone(Vec<f64>),
    DualPowerCone(Vec<f64>)
}

pub enum ParamConicDomainType {
    PrimalPowerCone,
    DualPowerCone
}

pub enum LinearDomainOfsType {
    Scalar(f64),
    M(Vec<f64>)
}

/// A Linear domain defines bounds, shape and sparsity for a model item.
pub struct LinearDomain<const N : usize> {
    /// Bound type
    pub(super) dt    : LinearDomainType,
    /// Offset type, either a scalar or a vector that matches the shape/sparsity
    pub(super) ofs   : LinearDomainOfsType,
    /// Shape of the domain, which will also define the shape of the model item
    pub(super) shape : [usize; N],
    /// Sparsity - this is used to create sparsity for the model item
    pub(super) sp    : Option<Vec<usize>>,
    /// Indicates if the domain in integer or continuous.
    pub(super) is_integer : bool
}

/// A Conic domain defines a conic domain, shape and cone dimension for a model item.
pub struct ConicDomain<const N : usize> {
    /// Cone type
    pub(super) dt      : ConicDomainType,
    /// Offset 
    pub(super) ofs     : Vec<f64>,
    /// Shape if the domain
    pub(super) shape   : [usize; N],
    /// Dimension in which the cones are aligned
    pub(super) conedim : usize,
    /// Indicates if the domain in integer or continuous.
    pub(super) is_integer : bool
}

/// A semidefinite conic domain.
pub struct PSDDomain<const N : usize> {
    /// Shape of the domain - note that two dimensions must be the same to allow symmetry.
    pub(super) shape    : [usize; N],
    /// The two cone dimensions where the cones are aligned.
    pub(super) conedims : (usize,usize)
}


/////////////////////////////////////////////////////////////////////
// Domain implementations
impl<const N :usize> ConicDomain<N> {
    pub fn integer(mut self) -> ConicDomain<N> { 
        self.is_integer = true;
        self
    }
    pub fn axis(mut self, i : usize) -> ConicDomain<N> {
        self.conedim = i;
        self
    }
}
impl<const N : usize> LinearDomain<N> {
    /// Reshape the domain. The new shape must "match" the domain, meaning that if 
    /// - if `sparsity` is present, the shape must contain all sparsity elements, otherwise
    /// - if `ofs` is a scalar, any shape goes, otherwise
    /// - the shape must match the length of `ofs`
    pub fn with_shape<const M : usize>(self,shape : &[usize; M]) -> LinearDomain<M> {
        let (dt,ofs,_,sp,is_integer) = (self.dt,self.ofs,self.shape,self.sp,self.is_integer);

        let shapesize : usize = shape.iter().product();
        if let Some(ref sp) = sp {
            if let Some(&i) = sp.last() { 
                if i >= shapesize {
                    panic!("Shaped does not match sparsity");
                }
            }
        }
        else {
            match ofs {
                LinearDomainOfsType::Scalar(_) => { 
                    // ok, any shape goes
                },
                LinearDomainOfsType::M(ref ofs) => {
                    if ofs.len() != shapesize {
                        panic!("Shaped does not fit expression");
                    }
                }
            }
        } 

        LinearDomain{
            dt,
            ofs,
            shape:*shape,
            sp,
            is_integer,
        }
    }

    /// Set or update sparsity.
    ///
    /// The given sparsity pattern must "match" the domain. This means that
    /// - if `ofs` is a scalar, any sparsity goes that keeps within the shape, otherwise
    /// - the sparsity must match the length of `ofs`
    ///
    /// # Arguments
    ///
    /// - `sp` - The sparsity pattern. This must be sorted in ascending order.
    pub fn with_sparsity(self,sp : &[[usize;N]]) -> LinearDomain<N> {
        for idx in sp.iter() {
            if idx.iter().zip(self.shape.iter()).any(|(&i,&d)| i >= d) {
                panic!("Sparsity pattern entry out of bounds");
            }
        }

        let mut strides = [0usize; N]; 
        let _ = strides.iter_mut().zip(self.shape.iter()).rev().fold(1,|c,(s,&d)| { *s = c; c*d });

        let spx = sp.iter().map(|idx| idx.iter().zip(strides.iter()).map(|(&i,&s)| i * s).sum()).collect();

        self.with_sparsity_indexes(spx)
    }
    /// Set or update sparsity using linearized indexes rather than coordinate indexes.
    ///
    /// The sparsity pattern must match the domain: 
    /// - the maximum element of the sparsity pattern must be within the shape
    ///
    /// # Arguments
    ///
    /// - `sp` - The sparsity pattern. This must be sorted in ascending order.
    pub fn with_sparsity_indexes(self, sp : Vec<usize>) -> LinearDomain<N> {
        if let LinearDomainOfsType::M(ref ofs) = self.ofs {
            if ofs.len() != sp.len() {
                panic!("Sparsity pattern does not match domain");
            }
        }

        if sp.iter().zip(sp[1..].iter()).any(|(&i0,&i1)| i1 <= i0) {
            // unsorted
            let mut perm : Vec<usize> = (0..sp.len()).collect();
            perm.sort_by_key(|&i| unsafe{ *sp.get_unchecked(i) });
            if izip!(perm_iter(perm.as_slice(),sp.as_slice()),
                     perm_iter(&perm[1..],sp.as_slice())).any(|(&i0,&i1)| i1 <= i0) {
                panic!("Sparsity pattern contains duplicates");
            }
            
            let rsp = perm_iter(perm.as_slice(),sp.as_slice()).cloned().collect::<Vec<usize>>();
            let ofs = if let LinearDomainOfsType::M(ref ofs) = self.ofs {
                LinearDomainOfsType::M(perm_iter(perm.as_slice(),ofs.as_slice()).cloned().collect())
            } else {
                self.ofs
            };

            if *rsp.last().unwrap() >= self.shape.iter().product() {
                panic!("Sparsity pattern does not match domain");
            }

            LinearDomain{
                dt    : self.dt,
                ofs,
                shape : self.shape,
                sp    : Some(rsp), 
                is_integer : false
            }
        } else {
            LinearDomain::<N>{
                dt    : self.dt,
                ofs   : self.ofs,
                shape : self.shape,
                sp    : Some(sp), 
                is_integer : false
            }
        }
    }

    /// Set shape and sparsity at the same time. The input domain is consumed.
    ///
    /// # Arguments
    /// - `shape` - An `M`-dimensional array defining the new shape.
    /// - `sp` - An array defining the sparsity. The sparsity pattern must be valid in connection
    ///    with the give `shape` and must match the domains offset vector. If the offset is a
    ///    scalar, any number of sparsity elements are allowed, otherwise the number must match the
    ///    length of the offset. The sparsity pattern must be sorted.
    /// # Returns
    /// - A new linear domain with the given shape and sparsity
    pub fn with_shape_and_sparsity<const M : usize>(self,shape : &[usize; M], sp : &[[usize;M]]) -> LinearDomain<M> {
        LinearDomain{
            dt : self.dt,
            ofs : self.ofs,
            shape : *shape,
            sp : None, 
            is_integer : false}.with_sparsity(sp)
    }

    pub fn integer(mut self) -> Self { self.is_integer = true; self }
    pub fn extract(self) -> (LinearDomainType,Vec<f64>,[usize;N],Option<Vec<usize>>,bool) {
        match self.ofs {
            LinearDomainOfsType::M(v) => (self.dt,v,self.shape,self.sp,self.is_integer),
            LinearDomainOfsType::Scalar(s) => 
                if let Some(sp) = self.sp {
                    (self.dt,vec![s; sp.len()],self.shape,Some(sp),self.is_integer)
                } 
                else {
                    let totalsize = self.shape.iter().product();
                    (self.dt,vec![s; totalsize],self.shape,None,self.is_integer)
                }
        }
    }
}

/// The OffsetTrait represents something that can act as an offset or bound value for a domain.
pub trait OffsetTrait<const N : usize> {
    fn greater_than(self) -> LinearDomain<N>;
    fn less_than(self)    -> LinearDomain<N>;
    fn equal_to(self)     -> LinearDomain<N>;
}

/// Make `f64` work as a scalar offset value.
impl OffsetTrait<0> for f64 {
    fn greater_than(self) -> LinearDomain<0> { LinearDomain{ dt : LinearDomainType::NonNegative,  ofs:LinearDomainOfsType::Scalar(self), shape:[], sp : None, is_integer : false } }
    fn less_than(self)    -> LinearDomain<0> { LinearDomain{ dt : LinearDomainType::NonPositive,  ofs:LinearDomainOfsType::Scalar(self), shape:[], sp : None, is_integer : false } }
    fn equal_to(self)     -> LinearDomain<0> { LinearDomain{ dt : LinearDomainType::Zero,         ofs:LinearDomainOfsType::Scalar(self), shape:[], sp : None, is_integer : false } }
}

/// Let `Vec<f64>` act as a vector offset value.
impl OffsetTrait<1> for Vec<f64> {
    fn greater_than(self) -> LinearDomain<1> { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:LinearDomainOfsType::M(self), shape:[n], sp : None, is_integer : false } }
    fn less_than(self)    -> LinearDomain<1> { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:LinearDomainOfsType::M(self), shape:[n], sp : None, is_integer : false } }
    fn equal_to(self)     -> LinearDomain<1> { let n = self.len(); LinearDomain{ dt : LinearDomainType::Zero,        ofs:LinearDomainOfsType::M(self), shape:[n], sp : None, is_integer : false } }
}

/// Let `&[f64]` act as a vector offset value.
impl OffsetTrait<1> for &[f64] {
    fn greater_than(self) -> LinearDomain<1> { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:LinearDomainOfsType::M(self.to_vec()), shape:[n], sp : None, is_integer : false } }
    fn less_than(self)    -> LinearDomain<1> { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:LinearDomainOfsType::M(self.to_vec()), shape:[n], sp : None, is_integer : false } }
    fn equal_to(self)     -> LinearDomain<1> { let n = self.len(); LinearDomain{ dt : LinearDomainType::Zero,        ofs:LinearDomainOfsType::M(self.to_vec()), shape:[n], sp : None, is_integer : false } }
}

impl<M> OffsetTrait<2> for M where M : Matrix {
    fn greater_than(self) -> LinearDomain<2> { let (shape,data) = self.extract_full(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:LinearDomainOfsType::M(data), shape, sp:None, is_integer : false }
    }
    fn less_than(self)    -> LinearDomain<2> { let (shape,data) = self.extract_full(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:LinearDomainOfsType::M(data), shape, sp:None, is_integer : false } }
    fn equal_to(self)     -> LinearDomain<2> { let (shape,data) = self.extract_full(); LinearDomain{ dt : LinearDomainType::Zero,        ofs:LinearDomainOfsType::M(data), shape, sp:None, is_integer : false } }
}

impl<const N : usize> OffsetTrait<N> for DenseNDArray<N> {
    fn greater_than(self) -> LinearDomain<N> { let (shape,data) = self.extract(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:LinearDomainOfsType::M(data), shape, sp : None, is_integer : false } }
    fn less_than(self)    -> LinearDomain<N> { let (shape,data) = self.extract(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:LinearDomainOfsType::M(data), shape, sp : None, is_integer : false } }
    fn equal_to(self)     -> LinearDomain<N> { let (shape,data) = self.extract() ;LinearDomain{ dt : LinearDomainType::Zero,        ofs:LinearDomainOfsType::M(data), shape, sp : None, is_integer : false } }
}

////////////////////////////////////////////////////////////
// Domain constructors
////////////////////////////////////////////////////////////

/// Unbounded domain
pub fn unbounded() -> LinearDomain<0> { LinearDomain{ dt : LinearDomainType::Free, ofs : LinearDomainOfsType::Scalar(0.0), shape : [], sp : None, is_integer : false } }   
/// Scalar domain of nonnegative values
pub fn nonnegative() -> LinearDomain<0> { 0f64.greater_than() }
/// Scalar domain of nonpositive values
pub fn nonpositive() -> LinearDomain<0> { 0f64.less_than() }
/// Domain of values greater than the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`
pub fn greater_than<const N : usize, T : OffsetTrait<N>>(v : T) -> LinearDomain<N> { v.greater_than() }
/// Domain of values less than the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`
pub fn less_than<const N : usize, T : OffsetTrait<N>>(v : T) -> LinearDomain<N> { v.less_than() }
/// Domain of values equal to the offset `v`. 
/// 
/// # Arguments
/// - `v` - Offset, the shape of the domain is taken from the shape of `v`
pub fn equal_to<const N : usize, T : OffsetTrait<N>>(v : T) -> LinearDomain<N> { v.equal_to() }
/// Domain of a single quadratic cone if size `dim`. The result is a vector domain of size `dim`.
/// 
/// # Arguments
/// - `dim` - dimension of the cone.
pub fn in_quadratic_cone(dim : usize) -> ConicDomain<1> { ConicDomain{dt:ConicDomainType::QuadraticCone,ofs:vec![0.0; dim],shape:[dim],conedim:0, is_integer : false} }
/// Domain of a single rotated quadratic cone if size `dim`. The result is a vector domain of size `dim`.
/// 
/// # Arguments
/// - `dim` - dimension of the cone.
pub fn in_rotated_quadratic_cone(dim : usize) -> ConicDomain<1> { ConicDomain{dt:ConicDomainType::RotatedQuadraticCone,ofs:vec![0.0; dim],shape:[dim],conedim:0, is_integer : false} }
/// Domain of a single geometric mean cone if size `dim`. The result is a vector domain of size `dim`.
/// 
/// # Arguments
/// - `dim` - dimension of the cone.
pub fn in_geometric_mean_cone(dim : usize) -> ConicDomain<1> { ConicDomain{dt:ConicDomainType::GeometricMeanCone,ofs:vec![0.0; dim],shape:[dim],conedim:0, is_integer : false} }
/// domain of a single dual geometric mean cone if size `dim`. the result is a vector domain of size `dim`.
/// 
/// # arguments
/// - `dim` - dimension of the cone.
pub fn in_dual_geometric_mean_cone(dim : usize) -> ConicDomain<1> { ConicDomain{dt:ConicDomainType::DualGeometricMeanCone,ofs:vec![0.0; dim],shape:[dim],conedim:0, is_integer : false} }
/// domain of a single exponential mean cone if size `dim`. the result is a vector domain of size `dim`.
/// 
/// # arguments
/// - `dim` - dimension of the cone.
pub fn in_exponential_cone() -> ConicDomain<1> { ConicDomain{dt:ConicDomainType::ExponentialCone,ofs:vec![0.0; 3],shape:[3],conedim:0, is_integer : false} }
/// Domain of a single dual exponential cone if size `dim`. The result is a vector domain of size `dim`.
/// 
/// # Arguments
/// - `dim` - dimension of the cone.
pub fn in_dual_exponential_cone(dim : usize) -> ConicDomain<1> { ConicDomain{dt:ConicDomainType::DualExponentialCone,ofs:vec![0.0; dim],shape:[dim],conedim:0, is_integer : false} }

/// Domain of a single power cone.
///
/// # Arguments
/// - `dim` Dimension of the power cone
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_power_cone(dim : usize, alpha : &[f64]) -> ConicDomain<1> {
    if dim <= alpha.len() { panic!("Mismatching dimension and alpha"); }
    let alphasum : f64 = alpha.iter().sum();
    ConicDomain{
        dt:ConicDomainType::PrimalPowerCone(alpha.iter().map(|&a| a / alphasum ).collect()),
        shape:[alpha.len()+1],
        ofs:vec![0.0; dim],
        conedim:0, 
        is_integer : false}
}
/// Domain of a single power cone.
///
/// # Arguments
/// - `dim` Dimension of the power cone
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_dual_power_cone(dim : usize, alpha : &[f64]) -> ConicDomain<1> {
    if dim <= alpha.len() { panic!("Mismatching dimension and alpha"); }
    let alphasum : f64 = alpha.iter().sum();
    ConicDomain{
        dt:ConicDomainType::DualPowerCone(alpha.iter().map(|&a| a / alphasum ).collect()),
        shape:[alpha.len()+1],
        ofs:vec![0.0; dim],
        conedim:0, 
        is_integer : false}
}

fn in_cones<const N : usize>(shape : &[usize; N], conedim : usize,ct : ConicDomainType) -> ConicDomain<N> {
    if conedim >= shape.len() {
        panic!("Invalid cone dimension");
    }
    ConicDomain{dt:ct,
                ofs : vec![0.0; shape.iter().product()],
                shape:*shape,
                conedim, 
                is_integer : false}
}

/// Domain of a multiple quadratic cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_quadratic_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicDomain<N> { in_cones(shape,conedim,ConicDomainType::QuadraticCone) }
/// domain of a multiple rotated quadratic cones.
/// 
/// # arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_rotated_quadratic_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicDomain<N> { in_cones(shape,conedim,ConicDomainType::RotatedQuadraticCone) }
/// Domain of a multiple geometric mean cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_geometric_mean_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicDomain<N> { in_cones(shape,conedim,ConicDomainType::GeometricMeanCone) }
/// Domain of a multiple dual geometric mean cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_dual_geometric_mean_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicDomain<N> { in_cones(shape,conedim,ConicDomainType::DualGeometricMeanCone) }
/// domain of a multiple exponential cones.
/// 
/// # arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_exponential_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicDomain<N> { 
    if let Some(&d) = shape.get(conedim) { if d != 3 { panic!("Invalid shape or exponential cone") } }
    in_cones(shape,conedim,ConicDomainType::GeometricMeanCone) 
}
/// Domain of a multiple dual exponential cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone.
/// - `conedim` - index of the dimension in which the cones are aligned.
pub fn in_dual_exponential_cones<const N : usize>(shape : &[usize; N], conedim : usize) -> ConicDomain<N> { 
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
pub fn in_power_cones<const N : usize>(shape : &[usize;N], conedim : usize, alpha : &[f64]) -> ConicDomain<N> {
    if conedim >= shape.len() {
        panic!("Mismatching conedim and shape");
    }
    let dim = shape[conedim];

    if dim <= alpha.len() { panic!("Mismatching cone dimension size and alpha"); }
    let alphasum : f64 = alpha.iter().sum();
    ConicDomain{
        dt:ConicDomainType::PrimalPowerCone(alpha.iter().map(|&a| a / alphasum ).collect()),
        shape : *shape,
        ofs:vec![0.0; shape.iter().product()],
        conedim,
        is_integer : false}
}

/// Domain of a number of dual power cones.
///
/// # Arguments
/// - `shape` Shape of the domain
/// - `conedim` Index of the dimension in which the individual cones are alighed.
/// - `alpha` The powers of the power cone. This will be normalized, i.e. each element is divided
///   by `sum(alpha)`
pub fn in_dual_power_cones<const N : usize>(shape : &[usize;N], conedim : usize, alpha : &[f64]) -> ConicDomain<N> {
    if conedim >= shape.len() {
        panic!("Mismatching conedim and shape");
    }
    let dim = shape[conedim];

    if dim <= alpha.len() { panic!("Mismatching cone dimension size and alpha"); }
    let alphasum : f64 = alpha.iter().sum();
    ConicDomain{
        dt:ConicDomainType::PrimalPowerCone(alpha.iter().map(|&a| a / alphasum ).collect()),
        shape : *shape,
        ofs:vec![0.0; shape.iter().product()],
        conedim,
        is_integer : false}
}

/// Domain of a single symmetric positive semidefinite cones.
/// 
/// # Arguments
/// - `dim` - Dimension of the PSD cone.
pub fn in_psd_cone(dim : usize) -> PSDDomain<2> {
    PSDDomain{
        shape : [dim,dim],
        conedims : (0,1)
    }
}
/// Domain of a multiple symmetric positive semidefinite cones. The cones are aligned in the two
/// dimensions give by `conedim1` and `conedim2`. 
/// 
/// # Arguments
/// - `shape` - shape of the cone, where `shape[conedim1]==shape[conedim2]`.
/// - `conedim1` - first cone dimension
/// - `conedim2` - second cone dimension. `conedim2` must be different from `conedim1`.
pub fn in_psd_cones<const N : usize>(shape : &[usize; N], conedim1 : usize, conedim2 : usize) -> PSDDomain<N> {
    if conedim1 == conedim2 || conedim1 >= shape.len() || conedim2 >= shape.len() {
        panic!("Invalid shape or cone dimensions");
    }
    if shape[conedim1] != shape[conedim2] {
        panic!("Mismatching cone dimensions");
    }
    PSDDomain{
        shape : *shape,
        conedims : (conedim1,conedim2)
    }
}
