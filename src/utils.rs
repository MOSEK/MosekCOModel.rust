//!
//! Utility module.
//!

use std::{cmp::Ordering, marker::PhantomData, ptr::NonNull};

use itertools::izip;
use utils::iter::*;

//->utils::iter -- 
//->utils::iter -- /// Trait that provides a function that copies from an iterator into a slice.
//->utils::iter -- ///
//->utils::iter -- /// # Notes
//->utils::iter -- /// Implementations should not fail if the lengths do not match, but rather copy the maximum number
//->utils::iter -- /// possible and return that number.
//->utils::iter -- pub trait AssignFromIterExt<I> where I : Iterator, I::Item : Copy 
//->utils::iter -- {
//->utils::iter --     #[allow(dead_code)]
//->utils::iter --     fn copy_from_iter(&mut self, it : I) -> usize;
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- impl<I> AssignFromIterExt<I> for [I::Item]
//->utils::iter --     where 
//->utils::iter --         I : Iterator,
//->utils::iter --         I::Item : Copy
//->utils::iter -- {
//->utils::iter --     fn copy_from_iter(&mut self, it : I) -> usize {
//->utils::iter --         self.iter_mut().zip(it).map(|(t,s)| *t = s).count()
//->utils::iter --     }
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- impl<I> AssignFromIterExt<I> for Vec<I::Item>
//->utils::iter --     where 
//->utils::iter --         I : Iterator,
//->utils::iter --         I::Item : Copy
//->utils::iter -- {
//->utils::iter --     fn copy_from_iter(&mut self, it : I) -> usize {
//->utils::iter --         self.iter_mut().zip(it).map(|(t,s)| *t = s).count()
//->utils::iter --     }
//->utils::iter -- }

/// An interator that produces an accumulation map, a bit like fold+map.
///
/// The iterator is initialized with a value, a mapping function and
/// an iterator. The iterator will produce
/// - i_0 -> v0
/// - i_{n+1} -> f(i_{n-1},it_n
///
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
        FoldMapIter{it : self, v : v0, f}
    }
}
impl<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> FoldMapExt<T,F> for I {}



////////////////////////////////////////////////////////////
pub struct FoldMap0Iter<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> {
    it : I,
    v : T,
    f : F
}

impl<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> Iterator for FoldMap0Iter<I,T,F> {
    type Item = T;
    fn next(& mut self) -> Option<Self::Item> {
        if let Some(v) = self.it.next() {
            let v = (self.f)(&self.v,v);
            let res = self.v;
            self.v = v;
            Some(res)
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

pub trait FoldMap0Ext<T:Copy,F:FnMut(&T,Self::Item) -> T> : Iterator {
    /// Create a cummulating iterator
    fn fold_map0(self, v0 : T, f : F) -> FoldMap0Iter<Self,T,F> where Self:Sized{
        FoldMap0Iter{it : self, v : v0, f}
    }
}
impl<I:Iterator,T:Copy,F:FnMut(&T,I::Item) -> T> FoldMap0Ext<T,F> for I {}



//->utils::iter -- ////////////////////////////////////////////////////////////
//->utils::iter -- 
//->utils::iter -- pub struct PermIter<'a,'b,T> {
//->utils::iter --     data : & 'b [T],
//->utils::iter --     perm : & 'a [usize],
//->utils::iter --     i : usize
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- pub fn perm_iter<'a,'b,T>(perm : &'a [usize], data : &'b[T]) -> PermIter<'a,'b,T> {
//->utils::iter --     if let Some(&v) = perm.iter().max() { if v >= data.len() { panic!("Permutation index out of bounds")} }
//->utils::iter --     PermIter{ data,perm,i:0 }
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- impl<'a,'b,T> Iterator for PermIter<'a,'b,T> {
//->utils::iter --     type Item = & 'b T;
//->utils::iter --     fn next(&mut self) -> Option<Self::Item> {
//->utils::iter --         if let Some(&i) = self.perm.get(self.i) {
//->utils::iter --             self.i += 1;
//->utils::iter --             Some(unsafe{self.data.get_unchecked(i)})
//->utils::iter --         }
//->utils::iter --         else {
//->utils::iter --             None
//->utils::iter --         }
//->utils::iter --     }
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- 
//->utils::iter -- pub trait PermuteByEx<'a,'b,T> {
//->utils::iter --     fn permute_by(self,perm:&'a[usize]) -> PermIter<'a,'b,T>; 
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- pub trait PermuteByMutEx<'a,'b,T> {
//->utils::iter --     fn permute_by_mut(self,perm:&'a[usize]) -> PermIterMut<'a,'b,T>; 
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- impl<'a,'b,T> PermuteByEx<'a,'b,T> for &'b [T] {
//->utils::iter --     fn permute_by(self,perm:&'a [usize]) -> PermIter<'a,'b,T> {
//->utils::iter --         if let Some(&v) = perm.iter().max() { if v >= self.len() { panic!("Permutation index out of bounds")} }
//->utils::iter --         PermIter{ data: self,perm, i:0 }
//->utils::iter --     }
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- ////////////////////////////////////////////////////////////
//->utils::iter -- // Mutable permutation iterator
//->utils::iter -- 
//->utils::iter -- // Ideas stolen from implementation of std::iter::IterMut
//->utils::iter -- pub struct PermIterMut<'a,'b,T : 'b> {
//->utils::iter --     perm : & 'a [usize],
//->utils::iter --     ptr  : NonNull<T>,
//->utils::iter --     _marker : PhantomData<&'b T>,
//->utils::iter --     i : usize
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- impl<'a,'b,T> Iterator for PermIterMut<'a,'b,T> {
//->utils::iter --     type Item = & 'b mut T;
//->utils::iter --     fn next(&mut self) -> Option<Self::Item> {
//->utils::iter --         if let Some(&i) = self.perm.get(self.i) {
//->utils::iter --             self.i += 1;
//->utils::iter --             Some(unsafe{ & mut ( *self.ptr.add(i).as_mut()) })
//->utils::iter --         }
//->utils::iter --         else {
//->utils::iter --             None
//->utils::iter --         }
//->utils::iter --     }
//->utils::iter -- }
//->utils::iter -- 
//->utils::iter -- impl<'a,'b,T> PermuteByMutEx<'a,'b,T> for &'b mut [T] {
//->utils::iter --     fn permute_by_mut(self,perm:&'a [usize]) -> PermIterMut<'a,'b,T> {
//->utils::iter --         if let Some(&v) = perm.iter().max() { if v >= self.len() { panic!("Permutation index out of bounds")} }
//->utils::iter --         PermIterMut{ 
//->utils::iter --             perm, 
//->utils::iter --             ptr : NonNull::from(self).cast(),
//->utils::iter --             _marker : PhantomData,
//->utils::iter --             i:0 }
//->utils::iter --     }
//->utils::iter -- }




////////////////////////////////////////////////////////////

pub fn append_name_index(buf : & mut String, index : &[usize]) {
    buf.push('[');
    if let Some(i) = index.first() {
        for c in i.digits_10() { buf.push(c); }
        for i in &index[1..] {
            buf.push(',');
            for c in i.digits_10() { buf.push(c); }
        }
    }
    buf.push(']');
}

////////////////////////////////////////////////////////////

pub struct ToDigit10Iter { d : usize, v : usize }
pub trait ToDigit10IterExt { fn digits_10(self) -> ToDigit10Iter; }
impl ToDigit10IterExt for usize {
    fn digits_10(self) -> ToDigit10Iter {
        let d10 = (self as f64).log10().floor() as usize;
        let d = (10.0f64).powi(d10 as i32) as usize;
        ToDigit10Iter{d:d.max(1),v:self}
    }
}

impl Iterator for ToDigit10Iter {
    type Item = char;
    fn next(& mut self) -> Option<char> {
        if self.d == 0 { None }
        else {
            let r = self.v / self.d;
            self.v %= self.d;
            self.d /= 10;
            Some((r as u8 + b'0') as char)
        }
    }
}



//pub fn chunks_by_iterator<'a,'b,T,I>(items : &'a [T], i : I) 
//where 
//    I:Clone+Iterator<Item=(&'b usize,&'b usize)>
//{
//    if i.clone().max_by_key(|(&p0,&p1)| p1).copied().unwrap_or(0) >= items.len()
//        || i.clone().any(|(&p0,&p1)| p1 > p0) {
//        panic!("Invalid index iterator");
//    }
//}
//






#[allow(dead_code)]
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

#[allow(dead_code)]
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

////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct IndexHashMap<'a,'b,'c,'d,T : Copy> {
    data   : & 'a mut[T],
    index  : & 'b mut[usize],
    next   : & 'c mut[usize],
    bucket : & 'd mut[usize],
    dflt   : T,
    n      : usize
}

fn hash(i : usize) -> usize { i }

impl<'a,'b,'c,'d,T : Copy> IndexHashMap<'a,'b,'c,'d,T> {
    pub fn new(data   : & 'a mut[T],
               index  : & 'b mut[usize],
               next   : & 'c mut[usize],
               bucket : & 'd mut[usize],
               dflt   : T) -> IndexHashMap<'a,'b,'c,'d,T> {
        bucket.iter_mut().for_each(|h| *h = usize::MAX);
        if next.len() != data.len() || next.len() != index.len() {
            panic!("Mismatching array sizes");
        }

        IndexHashMap{
            data,
            index,
            next,
            bucket,
            dflt,
            n : 0
        }
    }

    pub fn with_data(data   : & 'a mut[T],
                     index  : & 'b mut[usize],
                     next   : & 'c mut[usize],
                     bucket : & 'd mut[usize],
                     dflt : T) -> IndexHashMap<'a,'b,'c,'d,T> {
        bucket.iter_mut().for_each(|h| *h = usize::MAX);
        let n = data.len();
        let m = bucket.len();

        if next.len() != data.len() || next.len() != index.len() {
            panic!("Mismatching array sizes");
        }

        // Assume that index and data contains data to be put in the map
        for (i,&k,next) in izip!(0..n,index.iter(), next.iter_mut()) {
            let b = unsafe { &mut *bucket.get_unchecked_mut(hash(k) % m) };
            *next = *b;
            *b = i;
        }

        IndexHashMap{
            data,
            index,
            next,
            bucket,
            dflt,
            n}
    }

    pub fn at(&self,i : usize) -> Option<&T> {
        let mut index = unsafe { * self.bucket.get_unchecked(hash(i) % self.bucket.len()) };

        while index < usize::MAX && i != unsafe { * self.index.get_unchecked(index) }  {
            index = unsafe{ * self.next.get_unchecked(index) };
        }

        if index < usize::MAX {
            Some(unsafe { self.data.get_unchecked(index) })
        }
        else {
            None
        }
    }

    pub fn at_mut(&mut self, i : usize) -> &mut T {
        let key = hash(i) % self.bucket.len();
        let head = unsafe { self.bucket.get_unchecked_mut(key) };
        let mut index = *head;

        //println!("IndexHashMap, lookup {}\n\thead = {}",i,index);
        while index < usize::MAX && i != unsafe { * self.index.get_unchecked(index) } {
            //println!("\tindex = {}",index);
            index = unsafe{ * self.next.get_unchecked(index) };
        }

        if index < usize::MAX {
            unsafe { &mut * self.data.get_unchecked_mut(index) }
        }
        else if self.n < self.next.len() {
            index = self.n; self.n += 1;
            unsafe { *self.next.get_unchecked_mut(index) = *head; }
            unsafe { *self.index.get_unchecked_mut(index) = i; }
            *head = index;

            unsafe { *self.data.get_unchecked_mut(index) = self.dflt; }
            unsafe { & mut *self.data.get_unchecked_mut(index) }
        }
        else {
            panic!("Hashmap is full");
        }
    }

    pub fn len(&self) -> usize { self.n }
}


////////////////////////////////////////////////////////////

pub fn shape_eq(s0 : &[usize], s1 : &[usize]) -> bool { s0.iter().zip(s1.iter()).all(|(&a,&b)| a == b) }

/// Compare two shapes, disregarding one given dimension. This will return true if
/// 1. the lenths are the same and all entries are equal except entry [d], or
/// 2. the lengths are not the same, but all entries except [d] are
///    equal, where the remaining entries in the shorter shape are
///    considered to be 1.
pub fn shape_eq_except(s0 : &[usize], s1 : &[usize], d : usize) -> bool{
    match s0.cmp(s1) {
        Ordering::Greater => shape_eq_except(s1,s0,d),
        Ordering::Less => {
            if d >= s0.len() {
                // println!("{}:{}: shape_eq_except({:?},{:?},{})",file!(),line!(),s0,s1,d);
                shape_eq(s0,&s1[..s0.len()])
                    && s1[s1.len()..].iter().all(|&d| d == 1)
            }
            else {
                // println!("{}:{}: shape_eq_except({:?},{:?},{})",file!(),line!(),s0,s1,d);
                shape_eq(&s0[..d],&s1[..d])
                    && ( d >= s0.len()-1 || shape_eq(&s0[d+1..],&s1[d+1..s0.len()]))
                    && ( d >= s1.len()-1 || s1[d+1..].iter().all(|&d| d == 1))
            }
        },
        Ordering::Equal => {
            d >= s0.len() ||
                ( ( d   == 0        || shape_eq(&s0[..d],&s1[..d]) )
                    && ( d+1 == s0.len() || shape_eq(&s0[d+1..],&s1[d+1..]) ) )
        }
    }
}

#[allow(unused)]
pub trait NBoundGtOne<const N : usize> { }
impl<T> NBoundGtOne<2>  for T {}
impl<T> NBoundGtOne<3>  for T {}
impl<T> NBoundGtOne<4>  for T {}
impl<T> NBoundGtOne<5>  for T {}
impl<T> NBoundGtOne<6>  for T {}
impl<T> NBoundGtOne<7>  for T {}
impl<T> NBoundGtOne<8>  for T {}
impl<T> NBoundGtOne<9>  for T {}
impl<T> NBoundGtOne<10> for T {}
impl<T> NBoundGtOne<11> for T {}
impl<T> NBoundGtOne<12> for T {}
impl<T> NBoundGtOne<13> for T {}
impl<T> NBoundGtOne<14> for T {}
impl<T> NBoundGtOne<15> for T {}
impl<T> NBoundGtOne<16> for T {}
impl<T> NBoundGtOne<17> for T {}
impl<T> NBoundGtOne<18> for T {}
impl<T> NBoundGtOne<19> for T {}
impl<T> NBoundGtOne<20> for T {}



#[allow(unused)]
pub fn shape_to_strides<const N : usize>(shape : &[usize;N]) -> [usize;N] {
    let mut res = [0usize; N];
    _ = res.iter_mut().zip(shape.iter()).rev().fold(1,|c,(r,&d)| { *r = c; c * d });
    res
}
#[allow(unused)]
pub fn key_from_strides_index<const N : usize>(strides : [usize;N], index : usize) -> [usize;N] {
    let mut res = [0usize; N];
    _ = res.iter_mut().zip(strides.iter()).fold(index,|i,(r,&s)| { *r = i / s; i % s });
    res
}



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

    #[test]
    fn perm_iter_mut() {
        let vals = &mut [0,1,2,3,4,5,6,7,8,9];
        let ptr : &[usize] = &[0,3,5,7,9,10];
        let perm : &[usize] = &[9,4,2,8,3,0,5,1,5,7];

        for (i,v) in vals.permute_by_mut(perm).enumerate() { *v = i*2; }
        assert_eq!(vals, &[0,2,4,6,8,10,12,14,16,18]);

        
        for (i,c) in vals.chunks_by_mut(ptr, &ptr[1..]).enumerate() {
            c.iter_mut().for_each(|v| *v = i);
        }
    }
}
