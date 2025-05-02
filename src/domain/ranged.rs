use super::*;
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


//pub trait IntoLinearRange {
//    type Result;
//    fn into_range(self) -> Result<Self::Result,String>;
//}
//
//pub trait IntoShapedLinearRange<const N : usize> {
//    fn into_range(self, shape : [usize;N]) -> Result<LinearRangeDomain<N>,String>;
//}

impl IntoDomain for ScalableLinearRange {
    type Result = LinearRangeDomain<0>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
        Ok(LinearRangeDomain{
            shape : [],
            lower : vec![self.lower],
            upper : vec![self.upper], 
            sparsity : None,
            is_integer : self.is_integer,
        })
    }
}

impl<const N : usize> IntoShapedDomain<N> for ScalableLinearRange {
    type Result = LinearRangeDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
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

impl<const N : usize> IntoShapedDomain<N> for ProtoLinearRange<N> {
    type Result = LinearRangeDomain<N>;
    fn try_into_domain(self,shape : [usize;N]) -> Result<Self::Result,String> {
        if shape != self.shape {
            return Err("Domain shape does not match the expected shape.".to_string());
        }

        IntoDomain::try_into_domain(self)
    }
}

impl<const N : usize> IntoDomain for ProtoLinearRange<N> {
    type Result = LinearRangeDomain<N>;
    fn try_into_domain(self) -> Result<Self::Result,String> {
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

