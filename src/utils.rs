//!
//! Utility module.
//!

// pub struct Mutation {
//     tgtsize : usize,
//     idxs : Vec<usize>
// }

// pub struct MutationIter<'a,'b,T> {
//     i : usize,
//     n : usize,
//     m : & 'a Mutation,
//     t : & 'b [T]
// }
// // struct MutationIterMut<'a,'b,T> {
// //     i : usize,
// //     m : & 'a Mutation,
// //     t : & 'b mut [T]
// // }

// impl Mutation {
//     pub fn new(idxs : Vec<usize>) -> Mutation {
//         let &n = idxs.iter().max().unwrap_or(&0);

//         Mutation{
//             tgtsize : n,
//             idxs : idxs
//         }
//     }
//     pub fn id(n : usize) -> Mutation {
//         Mutation{
//             tgtsize : n,
//             idxs : (0..n).collect()
//         }
//     }
//     pub fn range(first : usize, n : usize) -> Mutation {
//         Mutation{
//             tgtsize : n,
//             idxs : (first..first+n).collect()
//         }
//     }

//     pub fn sort_arg<T:Ord>(& mut self,target : &[T]) {
//         if self.tgtsize > target.len() {
//             panic!("Invalid sorting target");
//         }
//         self.idxs.sort_by(|&i0,&i1| unsafe{ &*target.get_unchecked(i0) }.cmp(unsafe{ &*target.get_unchecked(i1) }) );
//     }

//     pub fn sort_arg_slice<T:Ord>(& mut self,first : usize, num : usize, target : &[T]) {
//         if self.tgtsize > target.len() {
//             panic!("Invalid sorting target");
//         }
//         if first+num > self.idxs.len() {
//             panic!("Slice out of bounds");
//         }
//         self.idxs[first..first+num].sort_by(|&i0,&i1| unsafe{ &*target.get_unchecked(i0) }.cmp(unsafe{ &*target.get_unchecked(i1) }) );
//     }

//     pub fn apply<'a,'b,T>(& 'a self, target : & 'b [T]) -> MutationIter<'a,'b,T> {
//         if self.tgtsize > target.len() {
//             panic!("Incompatible mutation");
//         }
//         MutationIter {
//             i : 0,
//             n : self.idxs.len(),
//             m : self,
//             t : target
//         }
//     }

//     pub fn apply_slice<'a,'b,T>(& 'a self, first : usize, num : usize, target : & 'b [T]) -> MutationIter<'a,'b,T> {
//         if self.tgtsize > target.len() {
//             panic!("Incompatible mutation");
//         }
//         else if first+num > self.idxs.len() {
//             panic!("Invalid slice range");
//         }
//         MutationIter {
//             i : first,
//             n : num,
//             m : self,
//             t : target
//         }
//     }
// }

// impl<'a,'b,T> std::iter::Iterator for MutationIter<'a,'b,T> {
//     type Item = & 'b T;
//     fn next(& mut self) -> Option<& 'b T> {
//         if self.i < self.n {
//             let idx = unsafe { * self.m.idxs.get_unchecked(self.i) };
//             self.i += 1;
//             Some(unsafe { & * self.t.get_unchecked(idx) })
//         }
//         else {
//             None
//         }
//     }
// }



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
pub struct FoldMapIter<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> {
    it : I,
    v : T,
    f : F
}

impl<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> Iterator for FoldMapIter<I,T,F> {
    type Item = T;
    fn next(& mut self) -> Option<Self::Item> {
        if let Some(v) = self.it.next() {
            let v = (self.f)(&self.v,v);
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

pub trait FoldMapExt<T:Copy,F:FnMut(&T,Self::Item) -> T> : Iterator {
    /// Create a cummulating iterator
    fn fold_map(self, v0 : T, f : F) -> FoldMapIter<Self,T,F> where Self:Sized{
        FoldMapIter{it : self, v : v0, f : f}
    }
}
impl<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> FoldMapExt<T,F> for I {}


////////////////////////////////////////////////////////////
pub struct IJKLSliceIterator<'a,'b,'c,'d,'e> {
    subi : & 'a [i64],
    subj : & 'b [i32],
    subk : & 'c [i32],
    subl : & 'd [i32],
    cof  : & 'e [f64],

    pos  : usize
}

pub fn ijkl_slice_iterator<'a,'b,'c,'d,'e>(subi : & 'a [i64],
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
    IJKLSliceIterator{subi,subj,subk,subl,cof,pos:0}
}

impl<'a,'b,'c,'d,'e> Iterator for IJKLSliceIterator<'a,'b,'c,'d,'e> {
    type Item = (i64,i32,&'c[i32],&'d[i32],&'e[f64]);
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
// pub enum Either<T0:Copy,T1:Copy> {
//     Left(T0),
//     Right(T1)
// }

// pub struct PartialFoldMapIter<T,I:Iterator,F:FnMut(&T,&T) -> Option<T>> {
//     cum : Option<T>,
//     it  : I,
//     f   : F
// }

// impl<T,I:Iterator,F:FnMut(&T,&T) -> Option<T>> Iterator for PartialFoldMapIter<T,I,F> {
//     type Output = T;
//     fn next() -> Option<T> {
//         while true {
//             if let Some(v0) = self.cum {
//                 match self.it.next() {
//                     Some(v) => {
//                         match (self.f)(&v0,v) {
//                             Some(w) => { self.cum = Some(w) },
//                             None => { self.cum = Some(v); return Some(v0); }
//                         }
//                     },
//                     None => {
//                         self.cum = None;
//                         return Some(v0);
//                     }
//                 }
//             }
//             else if let Some(&v) = self.it.next() {
//                 self.cum = Some(v)
//             }
//             else {
//                 return None;
//             }
//         }
//     }
// }

// pub trait PartialFoldMapIterExt<T,I:Iterator,F:FnMut(Option<T>,T) -> Option(T)> {
//     /// A partial reduction of an iterator.
//     ///
//     /// The function f will take a state and a value from the underlying iterator and return an Either, where
//     /// - Left(v) means replace the state with v,
//     /// - Right(v) means let the iterator return the state and replace the state with v
//     ///
//     /// The initial state of the iterator is the first element of the underlying iterator.
//     fn partial_fold_map(self, f : F) -> PartialFoldMapIter<T,I,F> {
//         PartialFoldMapIter{
//             cum : None,
//             it : self,
//             f }
//     }
// }

// impl<T,I:Iterator,F:FnMut(Option<T>,T) -> Either<T,T>> PartialFoldMapIterExt<T,I,F> for I { }

////////////////////////////////////////////////////////////

/// Given a shape, and optionally a sparsity pattern, call a function
/// f with each index in the shape by lexicalic order.
///
/// Arguments:
/// - shape The shape
/// - sp Optional sparsity pattern
/// - f Function called for each index in the shape
pub fn for_each_index<F>(shape : &[usize], sp : Option<&[usize]>, mut f : F) where F : FnMut(usize,&[usize]) {
    let mut idx = vec![0;shape.len()];

    if let Some(sp) = sp {
        let mut stride : Vec<usize> = vec![0; shape.len()];
        let _ = shape.iter().rev().zip(stride.iter_mut().rev()).fold(1,|k,(&d,s)| { *s = k; k*d });
        for &i in sp {
            let _ = idx.iter_mut().zip(stride.iter()).fold(i,|k,(ix,&s)| { *ix = k / s; k % s } );
            f(i,idx.as_slice());
        }
    }
    else {
        for i in 0..shape.iter().product() {
            f(i,idx.as_slice());
            let _ = shape.iter().zip(idx.iter_mut()).rev()
                .fold(1,|carry,(&d,ix)| {
                    if carry == 0 { 0 }
                    else {
                        *ix += carry;
                        if *ix < d { 0 }
                        else { *ix = 0; 1 }
                    }
                });
        }
    }
}

////////////////////////////////////////////////////////////

pub struct ToDigit10Iter { d : usize, v : usize }
pub trait ToDigit10IterExt { fn digits_10(self) -> ToDigit10Iter; }
impl ToDigit10IterExt for usize {
    fn digits_10(self) -> ToDigit10Iter {
        let d10 = (self as f64).log10().floor() as usize;
        let d = (10.0f64).powi(d10 as i32) as usize;
        // println!("d10 = {}, d = {}, v = {}",d10,d,self);
        ToDigit10Iter{d:d.max(1),v:self}
    }
}

impl Iterator for ToDigit10Iter {
    type Item = char;
    fn next(& mut self) -> Option<char> {
        // println!("d = {}, v = {}",self.d,self.v);
        if self.d == 0 { None }
        else {
            let r = self.v / self.d;
            self.v %= self.d;
            self.d /= 10;
            Some((r as u8 + b'0') as char)
        }
    }
}

////////////////////////////////////////////////////////////

//fn chain<U>(self, other: U) -> Chain<Self, <U as IntoIterator>::IntoIter>ⓘ where
//    U: IntoIterator<Item = Self::Item>,

pub struct ChunksByIter<'a,'b,T,I>
where
    I:Iterator<Item = (&'b usize,&'b usize)>
{
    data : &'a [T],
    ptr  : I
}

// pub struct ChunksByIterMut<'a,'b,T,I>
// where
//     I:Iterator<Item = (&'b usize,&'b usize)>
// {
//     data : &'a mut [T],
//     ptr  : I
// }

impl<'a,'b,T,I> Iterator for ChunksByIter<'a,'b,T,I>
where
    I:Iterator<Item = (&'b usize,&'b usize)>
{
    type Item = &'a[T];
    fn next(& mut self) -> Option<Self::Item> {
        if let Some((&p0,&p1)) = self.ptr.next() {
            Some(&self.data[p0..p1])
        }
        else {
            None
        }
    }
}

// impl<'a,'b,T,I> Iterator for ChunksByIterMut<'a,'b,T,I>
// where
//     I:Iterator<Item = (&'b usize,&'b usize)>
// {
//     type Item = & 'a mut[T];
//     fn next(& mut self) -> Option<Self::Item> {
//         if let Some((&p0,&p1)) = self.ptr.next() {
//             Some(& mut self.data[p0..p1])
//         }
//         else {
//             None
//         }
//     }
// }

pub trait ChunksByIterExt<T> {
    fn chunks_by<'a,'b>(&'a self, ptr : &'b[usize]) -> ChunksByIter<'a,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>>;
}
// pub trait ChunksByIterMutExt<T> {
//     fn chunks_by_mut<'a,'b>(&'a mut self, ptr : &'b[usize]) -> ChunksByIterMut<'a,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>>;
// }

impl<T> ChunksByIterExt<T> for &[T] {
    fn chunks_by<'a,'b>(& 'a self, ptr : &'b[usize]) -> ChunksByIter<'a,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>> {
        if let Some(&p) = ptr.last() { if p > self.len() { panic!("Invalid ptr for chunks_by iterator") } }
        if ptr.iter().zip(ptr[1..].iter()).any(|(p0,p1)| p1 < p0) { panic!("Invalid ptr for chunks_by iterator") }

        ChunksByIter{ data : self, ptr:ptr.iter().zip(ptr[1..].iter()) }
    }
}
// impl<T> ChunksByIterMutExt<T> for &[T] {
//     fn chunks_by_mut<'a,'b>(& 'a mut self, ptr : &'b[usize]) -> ChunksByIterMut<'a,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>> {
//         if let Some(&p) = ptr.last() { if p > self.len() { panic!("Invalid ptr for chunks_by iterator") } }
//         if ptr.iter().zip(ptr[1..].iter()).any(|(p0,p1)| p1 < p0) { panic!("Invalid ptr for chunks_by iterator") }

//         ChunksByIterMut{ data : self, ptr:ptr.iter().zip(ptr[1..].iter()) }
//     }
// }




////////////////////////////////////////////////////////////

pub struct SelectFromSliceIter<'a,'b,T> {
    src : &'a[T],
    idx : &'b[usize],
    i   : usize
}

impl<'a,'b,T> Iterator for SelectFromSliceIter<'a,'b,T> {
    type Item = & 'a T;
    fn next(& mut self) -> Option<Self::Item> {
        if self.i >= self.idx.len() {
            None
        }
        else {
            self.i += 1;
            Some(unsafe{ &*self.src.get_unchecked(*self.idx.get_unchecked(self.i-1))})
        }
    }
}
trait SelectFromSliceExt<T>{
    fn select<'a,'b>(&'a self, idxs : &'b [usize]) -> SelectFromSliceIter<'a,'b,T>;
}

impl<T> SelectFromSliceExt<T> for &[T] {
    fn select<'a,'b>(&'a self, idxs : &'b [usize]) -> SelectFromSliceIter<'a,'b,T> {
        SelectFromSliceIter{
            src : self,
            idx : idxs,
            i : 0
        }
    }
}

// pub struct SelectFromSliceIterMut<'a,'b,T> {
//     src : & 'a mut [T],
//     idx : & 'b[usize],
//     i   : usize
// }

// impl<'a,'b,T> Iterator for SelectFromSliceIterMut<'a,'b,T> {
//     type Item = & 'a mut T;
//     fn next(& mut self) -> Option<Self::Item> {
//         if self.i >= self.idx.len() {
//             None
//         }
//         else {
//             self.i += 1;
//             let r : & 'a mut T = unsafe{ & mut *self.src.get_unchecked_mut(*self.idx.get_unchecked(self.i-1))};
//             Some(r)
//         }
//     }
// }
// trait SelectFromSliceMutExt<T>{
//     fn select<'a,'b>(&'a mut self, idxs : &'b [usize]) -> SelectFromSliceIterMut<'a,'b,T>;
// }

// impl<T> SelectFromSliceMutExt<T> for &[T] {
//     fn select<'a,'b>(&'a mut self, idxs : &'b [usize]) -> SelectFromSliceIterMut<'a,'b,T> {
//         SelectFromSliceIter{
//             src : self,
//             idx : idxs,
//             i : 0
//         }
//     }
// }


////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        // let a1 = &[1,4,6,3,2,5,7,9,8];
        // let m1 = Mutation::id(9);
        // let m2 = Mutation::range(3,7);
        // let m3 = Mutation::new(&[8,7,6,5,4,3,2,1,0]);

        // assert!(m1.apply(a1).zip(a1.iter()).all(|(&a,&b)| a == b));
        // assert!(m2.apply(a1).zip([3,2,5,7].iter()).all(|(&a,&b)| a == b));
        // assert!(m3.apply(a1).zip([8,9,7,5,2,3,6,4,1].iter()).all(|(&a,&b)| a == b));

        // let mut m4 = Mutation::id(9); m4.sort_arg(a1);
        // assert!(m4.apply(a1).zip([1,2,3,4,5,6,7,8,9].iter()).all(|(&a,&b)| a == b));


        assert!([1,5,4,7,3,2].iter().fold_map(0,|&a,b| a+b).zip([1,6,10,17,20,22].iter()).all(|(a,&b)| a == b));
    }
}
