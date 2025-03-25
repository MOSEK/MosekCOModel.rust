
use std::iter::StepBy;
use std::ops::{Range,RangeTo,RangeFrom,RangeFull, RangeToInclusive,RangeInclusive};
use super::variable::*;
use super::ModelItemIndex;

pub enum IndexType {
    Index(usize),
    Range(usize,usize),
    RangeFrom(usize),
    RangeFull(),

}



trait ModelItemIndexElement {
    fn tt(self) -> IndexType;
}

impl ModelItemIndexElement for usize  { 
    fn tt(self) -> IndexType {
        IndexType::Index(self)
    }
}

impl ModelItemIndexElement for Range<usize> { 
    fn tt(self) -> IndexType {
        if self.start <= self.end {
            IndexType::Range(self.start,self.end-self.start,1)
        }
        else {
            IndexType::RevRange(self.start,0,1)
        }
    }
}

impl ModelItemIndexElement for RangeInclusive<usize> { 
    fn tt(self) -> IndexType {
        if self.start() <= self.end() {
            IndexType::Range(*self.start(), self.end()-self.start()+1,1)
        }
        else {
            IndexType::Range(self.start(),0,1)
        }
    }
}

impl ModelItemIndexElement for RangeFrom<usize> { 
    fn tt(self) -> IndexType {
        IndexType::RangeFrom(self.start,1)
    }
}

impl ModelItemIndexElement for RangeTo<usize> { 
    fn tt(self) -> IndexType {
        IndexType::Range(0,self.end,1)
    }
}

impl ModelItemIndexElement for RangeToInclusive<usize> { 
    fn tt(self) -> IndexType {
        IndexType::Range(0,self.end+1,1)
    }
}

impl ModelItemIndexElement for RangeFull { 
    fn tt(self) -> IndexType {
        IndexType::RangeFull(1)
    }
}

impl<I> ModelItemIndexElement for I where I : Iterator<Item = usize> {
    fn tt(self) -> IndexType {
        
    }
}









impl<const N : usize> ModelItemIndex<Variable<N>> for [IndexType; N] {
    type Output = Variable<N>;

    fn index(self,obj : &Variable<N>) -> Self::Output {
        unimplemented!("Index not implemented");
    }
}




impl<I1,I2> ModelItemIndex<Variable<2>> for (I1,I2) 
    where
        I1 : ModelItemIndexElement,
        I2 : ModelItemIndexElement
{
    type Output = Variable<2>;
    fn index(self, v : &Variable<1>) -> Variable<1> {
        let n = self.len();
        if let Some(ref sp) = v.sparsity {
            let first = match sp.binary_search(&self.start) {
                Ok(i)  => i,
                Err(i) => i
            };
            let last = match sp.binary_search(&self.start) {
                Ok(i) => i+1,
                Err(i) => i
            };

            Variable{
                idxs     : Rc::new(v.idxs[first..last].to_vec()), 
                sparsity : Some(Rc::new(sp[first..last].iter().map(|&i| i - self.start).collect())),
                shape    : [n]
            }
        }
        else {
            Variable{
                idxs     : Rc::new(v.idxs[self].to_vec()),
                sparsity : None,
                shape    : [n]
            }
        }+1
    }
}


