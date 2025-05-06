use super::*;

#[derive(Clone,Copy)]
pub enum LinearDomainType {
    NonNegative,
    NonPositive,
    Zero,
    Free
}

///////////////////////////////////////////////////////////////////////////////
// ScalableLinearDomain
///////////////////////////////////////////////////////////////////////////////

/// A Linear domain defines bounds, shape and sparsity for a model item.
///
/// A set of member functions makes it possible to transform the domain by changing its shape
/// sparsity, offset etc. 
#[derive(Clone)]
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
            offset      : vec![self.offset; shape.iter().product()],
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
            shape       : [],
            offset      : vec![self.offset],
            sparsity    : None,
            domain_type : self.domain_type,
            is_integer  : self.is_integer
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
        if n != shape.iter().product::<usize>() {
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

impl<const N : usize> LinearDomain<N> {
    pub fn dissolve(self) -> (LinearDomainType,Vec<f64>,Option<Vec<usize>>,[usize;N],bool) { (self.domain_type,self.offset,self.sparsity,self.shape,self.is_integer) }
    //    /// Create a [VectorDomain] equivalent to the linear domain.
    //    pub fn to_conic(self) -> VectorDomain<N,> {
    //        let conedim = N.max(1) - 1;
    //        let domain_type = match self.domain_type {
    //            LinearDomainType::Zero => VectorDomainType::Zero,
    //            LinearDomainType::Free => VectorDomainType::Free,
    //            LinearDomainType::NonPositive => VectorDomainType::NonPositive,
    //            LinearDomainType::NonNegative => VectorDomainType::NonNegative
    //        };
    //        VectorDomain {
    //            domain_type,
    //            offset : self.offset,
    //            shape : self.shape,
    //            conedim,
    //            is_integer : self.is_integer
    //        }
    //    }

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

/// Unbounded scalable domain.
pub fn unbounded() -> ScalableLinearDomain { ScalableLinearDomain{offset : 0.0, domain_type : LinearDomainType::Free, is_integer : false } }
/// Scalable domain of nonnegative values
pub fn nonnegative() -> ScalableLinearDomain { ScalableLinearDomain{ offset : 0.0, domain_type : LinearDomainType::NonNegative, is_integer : false } }
/// Scalable domain of nonpositive values
pub fn nonpositive() -> ScalableLinearDomain { ScalableLinearDomain{ offset : 0.0, domain_type : LinearDomainType::NonPositive, is_integer : false } }
/// Scalable domain of zeros
pub fn zero() -> ScalableLinearDomain { ScalableLinearDomain{ offset : 0.0, domain_type : LinearDomainType::Zero, is_integer : false } }
