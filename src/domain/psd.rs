use super::*;



/// A semidefinite conic domain.
pub struct PSDDomain<const N : usize> {
    /// Shape of the domain - note that two dimensions must be the same to allow symmetry.
    shape    : [usize; N],
    /// The two cone dimensions where the cones are aligned.
    conedims : (usize,usize)
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

impl<const N :usize> PSDDomain<N> {
    pub fn dissolve(self) -> ([usize;N],(usize,usize)) { (self.shape,self.conedims) }
}



/// Scalable domain of symmetric positive definite cones of unknown size. By default the cones will
/// be aligned in the two last dimensions, but this as well as shape can be changed subsequently. 
///
/// The exact meaning of the constraint is that each slice in the two cone dimensions define a
/// constraint of the form
/// $$ 
/// 1/2 (E+E^T) \\succ 0
/// $$
/// So _symmetry_ of E is not enforced.
///
/// If the expression is already symmetric, this simply means \\(E\\succ 0\\).
///
/// For variables, a shape must be defined.
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_psd_cone() -> ScalablePSDDomain {
    ScalablePSDDomain{
        cone_dims : None
    }
}

/// Domain of a multiple symmetric positive semidefinite cones. 
///
/// By default the cones are aligned in the two innermost dimensions, but this can be changed. See
/// [PSDProtoDomain]. The size of the two cone dimensions must be the same. 
///
/// The exact meaning of the constraint is that each slice in the two cone dimensions define a
/// constraint of the form
/// $$ 
/// 1/2 (E+E^T) \\succ 0
/// $$
/// So _symmetry_ of E is not enforced.
///
/// If the expression is already symmetric, this simply means \\(E\\succ 0\\).
///
/// For variables this produces a stack of positive symmetric semidefinite cones.
/// 
/// # Arguments
/// - `shape` - shape of the cone, where `shape[conedim1]==shape[conedim2]`.
///<script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
pub fn in_psd_cones<const N : usize>(shape : &[usize; N]) -> PSDProtoDomain<N> {
    PSDProtoDomain{
        shape : *shape,
        cone_dims : None
    }
}
