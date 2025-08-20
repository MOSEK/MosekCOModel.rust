use std::{cmp::Ordering, iter::Peekable, marker::PhantomData, ptr::NonNull};

use itertools::Chunk;



/// Trait that provides a function that copies from an iterator into a slice.
///
/// # Notes
/// Implementations should not fail if the lengths do not match, but rather copy the maximum number
/// possible and return that number.
pub trait CopyFromIterExt<I> where I : Iterator, I::Item : Copy 
{
    /// Copy values from an iterator into an object
    fn copy_from_iter(&mut self, it : I) -> usize;
}


impl<I> CopyFromIterExt<I> for [I::Item]
    where 
        I : Iterator,
        I::Item : Copy
{
    fn copy_from_iter(&mut self, it : I) -> usize {
        self.iter_mut().zip(it).map(|(t,s)| *t = s).count()
    }
}

impl<I> CopyFromIterExt<I> for Vec<I::Item>
    where 
        I : Iterator,
        I::Item : Copy
{
    fn copy_from_iter(&mut self, it : I) -> usize {
        self.as_mut_slice().copy_from_iter(it)
    }
}



////////////////////////////////////////////////////////////

/// Permutation iterator. For an indexable object (for example a slice or a vector), this is an
/// iterator that iterates over the elements of that object in some mutated or permutated order. A
/// permutation may contain duplicate indexes and may not map to all elements in the object.
pub struct PermIter<'a,'b,T> {
    data : & 'b [T],
    perm : & 'a [usize],
    i : usize
}

impl<'a,'b,T> Iterator for PermIter<'a,'b,T> {
    type Item = & 'b T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&i) = self.perm.get(self.i) {
            self.i += 1;
            Some(unsafe{self.data.get_unchecked(i)})
        }
        else {
            None
        }
    }
}

/// Extends objects with `permute_by` function.
pub trait PermuteByEx<'a,'b,T> {
    /// Return an iterator that traverses `self` in some order given by `perm`.
    ///
    /// The function will panic if the permutation is not valid (if it contains indexes that are
    /// out of bound).
    fn permute_by(self,perm:&'a[usize]) -> PermIter<'a,'b,T>; 
}


impl<'a,'b,T> PermuteByEx<'a,'b,T> for &'b [T] {
    fn permute_by(self,perm:&'a [usize]) -> PermIter<'a,'b,T> {
        if let Some(&v) = perm.iter().max() { if v >= self.len() { panic!("Permutation index out of bounds")} }
        PermIter{ data: self,perm, i:0 }
    }
}

impl<'a,'b,T> PermuteByEx<'a,'b,T> for &'b Vec<T>  {
    fn permute_by(self,perm:&'a [usize]) -> PermIter<'a,'b,T> {
        if let Some(&v) = perm.iter().max() { if v >= self.len() { panic!("Permutation index out of bounds")} }
        PermIter{ data: self.as_slice(),perm, i:0 }
    }
}

////////////////////////////////////////////////////////////
// Mutable permutation iterator

/// Mutable permutation iterator.
///
/// NOTE: Mutable iterators cannot be implemented in purely safe Rust. Ideas stolen from implementation
/// of std::iter::IterMut
pub struct PermIterMut<'a,'b,T : 'b> {
    perm : & 'a [usize],
    ptr  : NonNull<T>,
    _marker : PhantomData<&'b T>,
    i : usize
}

impl<'a,'b,T> Iterator for PermIterMut<'a,'b,T> {
    type Item = & 'b mut T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&i) = self.perm.get(self.i) {
            self.i += 1;
            Some(unsafe{ &mut (*self.ptr.as_ptr().add(i).as_mut().unwrap()) })
        }
        else {
            None
        }
    }
}

impl<'a,'b,T> PermIterMut<'a,'b,T> {
    pub fn new(data : & 'b mut [T], perm : & 'a [usize]) -> Self {
        if let Some(&v) = perm.iter().max() { if v >= data.len() { panic!("Permutation index out of bounds")} }
        PermIterMut{ 
            perm, 
            ptr : NonNull::from(data).cast(),
            _marker : PhantomData,
            i:0 }
    }
}

impl<'a,'b,T> PermuteByMutEx<'a,'b,T> for &'b mut [T] {
    fn permute_by_mut(self,perm:&'a [usize]) -> PermIterMut<'a,'b,T> {
        PermIterMut::new(self,perm)
    }
}

pub trait PermuteByMutEx<'a,'b,T> {
    /// Return a mutable iterator that traverses the elements of `self` in some order given by
    /// `perm`.
    ///
    /// The function will panic if the permutation contains indexes that are out of bounds.
    fn permute_by_mut(self,perm:&'a[usize]) -> PermIterMut<'a,'b,T>; 
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

/// Given a shape, iterate over all indexes in that shape.
pub struct IndexIterator<const N : usize> {
    shape : [usize; N],
    cur   : [usize; N],
    done  : bool
}

impl<const N : usize> Iterator for IndexIterator<N> {
    type Item = [usize; N];
    fn next(& mut self) -> Option<Self::Item> {
        if self.done {
            None
        } 
        else {
            let res = self.cur;

            if 0 < self.shape.iter().zip(self.cur.iter_mut()).rev().fold(1, |v,(&d, i)| { *i += v; if *i < d { 0 } else { *i = 0; 1 } }) {
                self.done = true;
            }
            Some(res)
        }
    }
}

impl<const N : usize> IndexIterator<N> {
    pub fn new(shape : &[usize; N]) -> IndexIterator<N> {
        IndexIterator{
            shape : *shape,
            cur   : [0; N],
            done : shape.iter().any(|&v| v == 0)
        }
    }
}

pub trait IndexIteratorExt<const N : usize> {
    type R;
    #[allow(dead_code)]
    fn index_iterator(&self) -> Self::R;
}

impl<const N : usize> IndexIteratorExt<N> for [usize;N] {
    type R = IndexIterator<N>;
    fn index_iterator(&self) -> Self::R { IndexIterator::new(self) }
}



/// Given a shape and a list of linear indexes, iterate over the sparsity yielding the indexes
/// corresponding to the given shape.
pub struct SparseIndexIterator<'a, const N : usize> {
    stride : [usize; N],
    sp     : & 'a [usize],
    i      : usize
}

impl<'a, const N : usize> SparseIndexIterator<'a,N> {
    /// Create a new SparseIndexIterator
    ///
    /// The indexes are not verified.
    ///
    /// # Arguments
    /// - `shape` - shape to use
    /// - `sp` - list of sparsity indexes
    pub fn new(shape : &[usize; N], sp : &'a [usize]) -> SparseIndexIterator<'a,N> {
        let mut stride = [1usize; N];
        _ = stride.iter_mut().zip(shape.iter()).rev().fold(1, |v,(st,&d)| { *st = v; v*d });
        //println!("shape = {:?}, stride = {:?}",shape,stride);
        SparseIndexIterator{
            stride,
            sp,
            i : 0
        }
    }
}

impl<'a, const N : usize> Iterator for SparseIndexIterator<'a,N> {
    type Item = [usize; N];
    fn next(& mut self) -> Option<Self::Item> {
        if self.i < self.sp.len() {
            let mut res = [0usize; N];
            let v = unsafe{ *self.sp.get_unchecked(self.i) };
            self.i += 1;
            _ = res.iter_mut().zip(self.stride.iter()).fold(v,|v, (r,&s)| { *r = v / s; v % s });
            Some(res)
        }
        else {
            None
        }
    }
}


////////////////////////////////////////////////////////////

pub struct ChunksByIter<'a,'b1,'b2,T,I>
where
    I:Iterator<Item = (&'b1 usize,&'b2 usize)>
{
    data : &'a [T],
    ptr  : I
}

impl<'a,'b1,'b2,T,I> Iterator for ChunksByIter<'a,'b1,'b2,T,I>
where
    I:Iterator<Item = (&'b1 usize,&'b2 usize)>
{
    type Item = &'a[T];
    fn next(& mut self) -> Option<Self::Item> {
        if let Some((&p0,&p1)) = self.ptr.next() {
            Some(unsafe{ self.data.get_unchecked(p0..p1)})
        }
        else {
            None
        }
    }
}

pub trait ChunksByIterExt<T> {
    fn chunks_ptr<'a,'b>(&'a self, ptr : &'b[usize]) -> ChunksByIter<'a,'b,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>>;
    fn chunks_ptr2<'a,'b1,'b2>(&'a self, ptrb : &'b1[usize],ptre : &'b2[usize]) -> ChunksByIter<'a,'b1,'b2,T,std::iter::Zip<std::slice::Iter<'b1,usize>,std::slice::Iter<'b2,usize>>>;
}

impl<T> ChunksByIterExt<T> for [T] {
    fn chunks_ptr<'a,'b>(& 'a self, ptr : &'b[usize]) -> ChunksByIter<'a,'b,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>> {
        if let Some(&p) = ptr.last() { if p > self.len() { panic!("Invalid ptr for chunks_ptr iterator") } }
        if ptr.iter().zip(ptr[1..].iter()).any(|(p0,p1)| p1 < p0) { panic!("Invalid ptr for chunks_ptr iterator") }

        ChunksByIter{ data : self, ptr:ptr.iter().zip(ptr[1..].iter()) }
    }
    fn chunks_ptr2<'a,'b1,'b2>(& 'a self, ptrb : &'b1[usize], ptre : &'b2[usize]) -> ChunksByIter<'a,'b1,'b2,T,std::iter::Zip<std::slice::Iter<'b1,usize>,std::slice::Iter<'b2,usize>>> {
        if let Some(&p) = ptre.last() { if p > self.len() { panic!("Invalid ptr for chunks_ptr iterator") } }
        if ptrb.iter().zip(ptre.iter()).any(|(p0,p1)| p1 < p0) { panic!("Invalid ptrb/ptre for chunks_ptr iterator") }

        ChunksByIter{ data : self, ptr:ptrb.iter().zip(ptre.iter()) }
    }
}


////////////////////////////////////////////////////////////

/// Iterator that iterates over chunks of an array defined 
pub struct ChunksByIterMut<'a,'b,'c,T:'a>
{
    ptrb : & 'b [usize],
    ptre : & 'c [usize],
    ptr  : NonNull<T>,
    _marker : PhantomData<& 'a mut T>,
    i : usize
}

impl<'a,'b,'c,T:'a> ChunksByIterMut<'a,'b,'c,T> {
    fn new(data : &'a mut [T], ptrb : &'b[usize], ptre : &'c[usize]) -> Self {
        let n = ptrb.len().min(ptre.len());
        let ptrb = &ptrb[..n];
        let ptre = &ptre[..n];
        if let Some(&p) = ptrb.iter().max() { if p > data.len() { panic!("Ptrb element out of bounds for data: {} > {}",p,data.len()) } }
        if let Some(&p) = ptre.iter().max() { if p > data.len() { panic!("Invalid ptre for chunks_ptr iterator") } }
        if ptrb.iter().zip(ptre.iter()).any(|(&p0,&p1)| p0 > p1 ) {
            panic!("Invalid ptrb/ptre construction");
        }

        ChunksByIterMut{ 
            ptrb,
            ptre,
            ptr : NonNull::from(data).cast(), 
            _marker : PhantomData,
            i : 0 }
    }
}

impl<'a,'b,'c,T:'a> Iterator for ChunksByIterMut<'a,'b,'c,T> {
    type Item = &'a mut [T];
    fn next(& mut self) -> Option<Self::Item> {
        if self.i < self.ptrb.len() && self.i < self.ptre.len() {
            let (b,e) = unsafe {
                (*self.ptrb.get_unchecked(self.i),
                 *self.ptre.get_unchecked(self.i))
            };
            self.i += 1;

            unsafe {
                Some(std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(b).as_mut().unwrap(),e-b))
            }
        }
        else {
            None
        }
    }
}

pub trait ChunksByIterMutExt<T> {
    fn chunks_ptr_mut<'a,'b,'c>(&'a mut self, ptrb : &'b[usize],ptre : &'c[usize]) -> ChunksByIterMut<'a,'b,'c,T> where T:'a;
}

impl<T> ChunksByIterMutExt<T> for [T] {
    fn chunks_ptr_mut<'a,'b,'c>(& 'a mut self, ptrb : &'b[usize], ptre : & 'c[usize]) -> ChunksByIterMut<'a,'b,'c,T> where T:'a {
        ChunksByIterMut::new(self,ptrb,ptre)
    }
}

////////////////////////////////////////////////////////////

pub struct InnerJoinBy<I1,I2,F> 
    where 
        I1:Iterator, 
        I2:Iterator, 
        F:FnMut(&I1::Item,&I2::Item) -> std::cmp::Ordering 
{
    i1 : std::iter::Peekable<I1>,
    i2 : std::iter::Peekable<I2>,
    f : F
}

pub trait InnerJoinByExt<I2,F> 
    where 
        Self:Iterator+Sized, 
        I2:Iterator, 
        F:FnMut(&Self::Item,&I2::Item) -> std::cmp::Ordering 
{
    fn inner_join_by(self, f : F, other : I2) -> InnerJoinBy<Self,I2,F> {
        InnerJoinBy{
            i1 : self.peekable(),
            i2 : other.peekable(),
            f
        }
    }
}

impl<I1,I2,F> InnerJoinByExt<I2,F> for I1
    where 
        I1:Iterator, 
        I2:Iterator, 
        F:FnMut(&I1::Item,&I2::Item) -> std::cmp::Ordering
{
}

impl<T1,T2,I1,I2,F> Iterator for InnerJoinBy<I1,I2,F> 
    where 
        T1:Copy,
        T2:Copy,
        I1:Iterator<Item=T1>, 
        I2:Iterator<Item=T2>, 
        F:FnMut(&I1::Item,&I2::Item) -> std::cmp::Ordering 
{
    type Item = (I1::Item,I2::Item);
    fn next(&mut self) -> Option<Self::Item> {
        while let (Some(a),Some(b)) = (self.i1.peek(),self.i2.peek()) {
            match (self.f)(a,b) {
                std::cmp::Ordering::Less    => { _ = self.i1.next(); }
                std::cmp::Ordering::Greater => { _ = self.i2.next(); }
                std::cmp::Ordering::Equal   => { break }
            }
        }
        if let (Some(a),Some(b)) = (self.i1.next(),self.i2.next()) {
            Some((a,b))
        }
        else {
            None
        }
    }
}
////////////////////////////////////////////////////////////
pub struct OuterMergeBy<I1,I2,C> 
    where 
        I1 : Iterator,
        I2 : Iterator,
        C : FnMut(&I1::Item,&I2::Item) -> Ordering
{
    i1 : Peekable<I1>,
    i2 : Peekable<I2>,
    c  : C
}

pub trait OuterMergeByEx where Self : Iterator+Sized {
    /// Given two iterators, that are assumed to be sorted, perform an "outer merge", i.e. given a
    /// comparison function, zip the two lists by order.
    fn outer_merge_by<I2,C>(self, cmp : C, other : I2) -> OuterMergeBy<Self,I2,C>
        where 
            I2 : Iterator, 
            C : FnMut(&Self::Item,&I2::Item) -> Ordering 
    {
        OuterMergeBy{
            i1 : self.peekable(),
            i2 : other.peekable(),
            c : cmp
        }
    }
}

impl<I1,I2,C> Iterator for OuterMergeBy<I1,I2,C>
    where 
        I1 : Iterator,
        I2 : Iterator,
        C : FnMut(&I1::Item,&I2::Item) -> Ordering
{
    type Item = (Option<I1::Item>,Option<I2::Item>);
    fn next(&mut self) -> Option<Self::Item> {
        match (self.i1.peek(),self.i2.peek()) {
            (None,None) => None,
            (Some(_),None) => { let r = self.i1.next(); Some((r,None)) },
            (None,Some(_)) => { let r = self.i2.next(); Some((None,r)) },
            (Some(a),Some(b)) => match (self.c)(a,b) {
                Ordering::Less => { let r = self.i1.next(); Some((r,None)) },
                Ordering::Greater => { let r = self.i2.next(); Some((None,r)) },
                Ordering::Equal => { 
                    let r1 = self.i1.next();
                    let r2 = self.i2.next();
                    Some((r1,r2))
                }

            }
        }
    }
}





////////////////////////////////////////////////////////////

pub struct Interleave<I1,I2> 
where 
    I1 : Iterator,
    I2 : Iterator<Item=I1::Item>
{
    i1 : I1,
    i2 : I2,
    which : bool
}

pub trait InterleaveEx where Self : Iterator+Sized {
    /// Interleave two iterators, picking alternatingly an element from one and from the other.
    fn interleave<I2>(self, other : I2) -> Interleave<Self,I2> where I2 : Iterator<Item=Self::Item> {
        Interleave{
            i1 : self,
            i2 : other,
            which : false
        }
    }
}

impl<I1,I2> Iterator for Interleave<I1,I2> where I1:Iterator, I2:Iterator<Item=I1::Item> {
    type Item = I1::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.which {
            self.which = ! self.which;
            self.i2.next()
        }
        else {
            self.which = ! self.which;
            self.i1.next()
        }
    }
}

pub struct InterleaveN<'a,I> where I : Iterator {
    ii : &'a mut [I],
    which : usize
}

pub fn interleave<'a,I>(ii : &'a mut [I]) -> InterleaveN<'a,I> where I : Iterator {
    InterleaveN{
        ii,
        which : 0
    }
}

impl<'a, I> Iterator for InterleaveN<'a,I> where I : Iterator {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        {
            let i = self.which;
            self.which = (self.which+1)%self.ii.len();
            unsafe{self.ii.get_unchecked_mut(i)}.next()
        }
    }
}

////////////////////////////////////////////////////////////
//
//trait CopyFromIterEx<I,T> where T:Copy, I : Iterator<Item=T> {
//    fn copy_from_iter(&mut self, it : I);
//}
//trait CloneFromIterEx<I,T> where T:Clone, I : Iterator<Item=T> {
//    fn clone_from_iter(&mut self, it : I);
//}
//
//impl<I,T> CopyFromIterEx<I,T> for [T] where T:Copy, I:Iterator<Item=T> {
//    fn copy_from_iter(&mut self, it : I) {
//        for (t,s)
//    }
//}
//
////////////////////////////////////////////////////////////



/// Permutation of an array. Really, it is a mutation, since it may contain duplicate indexes.
pub struct Permutation<'a> {
    perm : &'a[usize],
    min  : usize,
    max  : usize
}

impl<'a> Permutation<'a> {
    pub fn new(perm : &'a[usize]) -> Permutation<'a> { 
        let (min,max) = if perm.is_empty() { (0,0) } else { perm.iter().fold((usize::MAX,0),|(min,max),&v| (v.min(min),v.max(max))) };
        Permutation{ perm,min,max }
    } 

    pub fn permute<'b,T>(&self,data : &'b [T]) -> Result<PermIter<'a,'b,T>,()> {
        if data.len() <= self.max { Err(()) }
        else { Ok(PermIter{data,perm:self.perm,i:0}) }
    }
    pub fn permute_mut<'b,T>(&self,data : &'b mut[T]) -> Result<PermIterMut<'a,'b,T>,()> {
        if data.len() <= self.max { Err(()) }
        else { Ok(PermIterMut::new(data,self.perm)) }
    }
}

pub struct ChunksByIter2<'a,'b,T>
{
    data : &'a [T],
    ptr : &'b [usize],
    index : usize,
}

pub struct ChunksByIter3<'a,'b,T> 
{
    data : &'a[T],
    len  : &'b[usize],
    index : usize,
}

impl<'a,'b,T> Iterator for ChunksByIter2<'a,'b,T>
{
    type Item = &'a[T];
    fn next(& mut self) -> Option<Self::Item> {
        if self.index+1 < self.ptr.len() {
            let i = self.index;
            self.index += 1;
            Some(unsafe{ self.data.get_unchecked(*self.ptr.get_unchecked(i)..*self.ptr.get_unchecked(i+1))})
        }
        else {
            None
        }
    }
}








pub struct ChunkationIter<'a,'b,'c,T> {
    c    : &'c Chunkation<'a>,
    data : &'b[T],
    i    : usize
}

impl<'a,'b,'c,T> Iterator for ChunkationIter<'a,'b,'c,T> {
    type Item = &'b[T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.i+1 < self.c.ptr.len() {
            let i = self.i; 
            self.i += 1;
            println!(" i = {}, len = {}",i,self.c.ptr.len());
            Some(unsafe{self.data.get_unchecked(*self.c.ptr.get_unchecked(i)..*self.c.ptr.get_unchecked(i+1))})
        }
        else {
            None
        }
    }
}

pub struct Chunkation<'a> {
    ptr : & 'a[usize],
    max : usize
}

impl<'a> Chunkation<'a> {
    pub fn new(ptr : &'a[usize]) -> Option<Self> {
        if let Some(max) = ptr.last() {
            if ptr.iter().zip(ptr[1..].iter()).any(|item| *item.0 > *item.1) { None }
            else {
                Some(Chunkation { ptr, max:*max })
            }
        } 
        else {
            None
        }
    }
    pub fn chunks<'b,'c,T>(&'a self, data : &'b [T]) -> Option<ChunkationIter<'a,'b,'c,T>> {
        if self.max > data.len() { None } 
        else { Some(ChunkationIter{ c : self, data, i : 0 }) }
    }
}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_iter_mut() {
        let vals = &mut [11,10,9,8,7,6,5,4,3,2];
        let perm : &[usize] = &[9,8,7,6,5,4,3,2,1,0];

        let ptr : &[usize] = &[0,3,5,7,9,10];

        
        assert_eq!(vals.permute_by(perm).cloned().collect::<Vec<usize>>().as_slice(),&[2,3,4,5,6,7,8,9,10,11]);

        for (i,v) in vals.permute_by_mut(perm).enumerate() { 
            //intln!("{} : perm = {}, v = {}",i,*p,*v);
            *v = i*2;
        }
        println!("vals = {:?}",vals);
        assert_eq!(vals,&[18,16,14,12,10,8,6,4,2,0]);
    }
}
