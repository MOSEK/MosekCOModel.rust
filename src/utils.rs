

pub struct Mutation {
    tgtsize : usize,
    idxs : Vec<usize>
}

pub struct MutationIter<'a,'b,T> {
    i : usize,
    n : usize,
    m : & 'a Mutation,
    t : & 'b [T]
}
// struct MutationIterMut<'a,'b,T> {
//     i : usize,
//     m : & 'a Mutation,
//     t : & 'b mut [T]
// }

impl Mutation {
    pub fn new(idxs : Vec<usize>) -> Mutation {
        let &n = idxs.iter().max().unwrap_or(&0);

        Mutation{
            tgtsize : n,
            idxs : idxs
        }
    }
    pub fn id(n : usize) -> Mutation {
        Mutation{
            tgtsize : n,
            idxs : (0..n).collect()
        }
    }
    pub fn range(first : usize, n : usize) -> Mutation {
        Mutation{
            tgtsize : n,
            idxs : (first..first+n).collect()
        }
    }

    pub fn sort_arg<T:Ord>(& mut self,target : &[T]) {
        if self.tgtsize > target.len() {
            panic!("Invalid sorting target");
        }
        self.idxs.sort_by(|&i0,&i1| unsafe{ &*target.get_unchecked(i0) }.cmp(unsafe{ &*target.get_unchecked(i1) }) );
    }

    pub fn sort_arg_slice<T:Ord>(& mut self,first : usize, num : usize, target : &[T]) {
        if self.tgtsize > target.len() {
            panic!("Invalid sorting target");
        }
        if first+num > self.idxs.len() {
            panic!("Slice out of bounds");
        }
        self.idxs[first..first+num].sort_by(|&i0,&i1| unsafe{ &*target.get_unchecked(i0) }.cmp(unsafe{ &*target.get_unchecked(i1) }) );
    }

    pub fn apply<'a,'b,T>(& 'a self, target : & 'b [T]) -> MutationIter<'a,'b,T> {
        if self.tgtsize > target.len() {
            panic!("Incompatible mutation");
        }
        MutationIter {
            i : 0,
            n : self.idxs.len(),
            m : self,
            t : target
        }
    }

    pub fn apply_slice<'a,'b,T>(& 'a self, first : usize, num : usize, target : & 'b [T]) -> MutationIter<'a,'b,T> {
        if self.tgtsize > target.len() {
            panic!("Incompatible mutation");
        }
        else if first+num > self.idxs.len() {
            panic!("Invalid slice range");
        }
        MutationIter {
            i : first,
            n : num,
            m : self,
            t : target
        }
    }
}

impl<'a,'b,T> std::iter::Iterator for MutationIter<'a,'b,T> {
    type Item = & 'b T;
    fn next(& mut self) -> Option<& 'b T> {
        if self.i < self.n {
            let idx = unsafe { * self.m.idxs.get_unchecked(self.i) };
            self.i += 1;
            Some(unsafe { & * self.t.get_unchecked(idx) })
        }
        else {
            None
        }
    }
}

// impl<'a,'b,T> std::iter::Iterator for MutationIterMut<'a,'b,T> {
//     type Item = & 'b mut T;
//     fn next(& mut self) -> Option<& 'b mut T> {
//         if self.i < self.m.idxs.len() {
//             let idx = unsafe { * self.m.idxs.get_unchecked(self.i) };
//             self.i += 1;
//             Some(unsafe { & mut (* self.t.get_unchecked_mut(idx)) })
//         }
//         else {
//             None
//         }
//     }
// }



/// An interator that produces an accumulation map, a bit like fold+map.
///
/// The iterator is initialized with a value, a mapping function and
/// an iterator. The iterator will produce
/// - i_0 -> v0
/// - i_{n+1} -> f(i_{n-1},it_n
///
/// Example:
///   [1,2,5,4,3].cummulate(0, |&v,&i| v+i)
///   -> [1,3,8,12,15]
struct FoldMapIter<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> {
    it : I,
    v : T,
    f : F
}


impl<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> Iterator for FoldMapIter<I,T,F> {
    type Item = T;
    fn next(& mut self) -> Option<Self::Item> {
        if let Some(v) = self.it.next() {
            let v = self.f(&self.v,v);
            self.v = v;
            Some(self.v)
        }
        else {
            None
        }
    }
    fn size_hint(&self) -> (usize,Option<usize>) {
        let (lb,ub) = self.it.size_hint();
        if let Some(ub) = ub { (lb,Some(ub+1)) }
        else { (lb,None) }
    }
}

trait FoldMapExt<T:Copy,F:FnMut(&T,Self::Item) -> T> : Iterator {
    /// Create a cummulating iterator
    fn fold_map(self, v0 : T, f : F) -> FoldMapIter<Self,T,F> where Self:Sized{
        FoldMapIter{it : self, v : v0, f : f}
    }
}
impl<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> FoldMapExt<T,F> for I {}


////////////////////////////////////////////////////////////
pub struct IJKLSliceIterator<'a,'b,'c,'d,'e> {
    subi : & 'a [i32],
    subj : & 'b [i32],
    subk : & 'c [i32],
    subl : & 'd [i32],
    cof  : & 'e [f64],

    pos  : usize
}

pub fn ijkl_slice_iterator<'a,'b,'c,'d,'e>(subi : & 'a [i32],
                                           subj : & 'b [i32],
                                           subk : & 'c [i32],
                                           subl : & 'd [i32],
                                           cof  : & 'e [f64]) -> IJKLSliceIterator<'a,'b,'c,'d,'e> {
    if subi.len() != subj.len()
        || subi.len() != subk.len()
        || subi.len() != subl.len()
        || subi.len() != cof.len() {
        panic!("Mismatching array length");
    }
    IJKLSliceIterator{subi,subj,subj,subj,cof,pos:0}
}

impl<'a,'b,'c,'d,'e> Iterator for IJKLSliceIterator<'a,'b,'c,'d,'e> {
    type Item = (i32,i32,&'c[i32],&'d[i32],&'e[f64]);
    fn next(& mut self) -> Option<Self::Item> {
        if self.pos < self.subi.len() {
            let p0 = self.pos;
            let i = unsafe{ *self.subi.get_unchecked(self.pos) };
            let j = unsafe{ *self.subj.get_unchecked(self.pos) };
            self.pos += 1;

            while unsafe{ *self.subi.get_unchecked(self.pos) == i } &&
                  unsafe{ *self.subj.get_unchecked(self.pos) == j } {
                self.pos += 1;
            }
            Some((i,j,&self.subk[p0..self.pos],&self.subl[p0..self.pos],&self.cof[p0..self.pos]))
        }
        else {
            None
        }
    }
}

////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        let a1 = &[1,4,6,3,2,5,7,9,8];
        let m1 = Mutation::id(9);
        let m2 = Mutation::range(3,7);
        let m3 = Mutation::new(&[8,7,6,5,4,3,2,1,0]);

        assert!(m1.apply(a1).zip(a1.iter()).all(|(&a,&b)| a == b));
        assert!(m2.apply(a1).zip([3,2,5,7].iter()).all(|(&a,&b)| a == b));
        assert!(m3.apply(a1).zip([8,9,7,5,2,3,6,4,1].iter()).all(|(&a,&b)| a == b));

        let mut m4 = Mutation::id(9); m4.sort_arg(a1);
        assert!(m4.apply(a1).zip([1,2,3,4,5,6,7,8,9].iter()).all(|(&a,&b)| a == b));
    }
}
