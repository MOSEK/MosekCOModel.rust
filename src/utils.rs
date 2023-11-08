//!
//! Utility module.
//!

use std::slice::SliceIndex;

use itertools::izip;




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



////////////////////////////////////////////////////////////

pub struct PermIter<'a,'b,T> {
    data : & 'b [T],
    perm : & 'a [usize],
    i : usize
}

pub fn perm_iter<'a,'b,T>(perm : &'a [usize], data : &'b[T]) -> PermIter<'a,'b,T> {
    if let Some(&v) = perm.iter().max() { if v >= data.len() { panic!("Invalid permutation")} }
    PermIter{ data,perm,i:0 }
}

impl<'a,'b,T> Iterator for PermIter<'a,'b,T> {
    type Item = & 'b T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&i) = self.perm.get(self.i) {
            self.i += 1;
            Some(unsafe{&*self.data.get_unchecked(i)})
        }
        else {
            None
        }
    }
}

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

////////////////////////////////////////////////////////////


pub struct ChunksByIter<'a,'b,T,I>
where
    I:Iterator<Item = (&'b usize,&'b usize)>
{
    data : &'a [T],
    ptr  : I
}

impl<'a,'b,T,I> Iterator for ChunksByIter<'a,'b,T,I>
where
    I:Iterator<Item = (&'b usize,&'b usize)>
{
    type Item = &'a[T];
    fn next(& mut self) -> Option<Self::Item> {
        if let Some((&p0,&p1)) = self.ptr.next() {
            // Note: The constructor of the ChunksByIter object MUST ensure that all slices are
            // valid!
            Some(unsafe{ self.data.get_unchecked(p0..p1)})
            //Some(&self.data[p0..p1])
        }
        else {
            None
        }
    }
}

pub trait ChunksByIterExt<T> {
    fn chunks_by<'a,'b>(&'a self, ptr : &'b[usize]) -> ChunksByIter<'a,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>>;
}

impl<T> ChunksByIterExt<T> for &[T] {
    fn chunks_by<'a,'b>(& 'a self, ptr : &'b[usize]) -> ChunksByIter<'a,'b,T,std::iter::Zip<std::slice::Iter<'b,usize>,std::slice::Iter<'b,usize>>> {
        if let Some(&p) = ptr.last() { if p > self.len() { panic!("Invalid ptr for chunks_by iterator") } }
        if ptr.iter().zip(ptr[1..].iter()).any(|(p0,p1)| p1 < p0) { panic!("Invalid ptr for chunks_by iterator") }

        ChunksByIter{ data : self, ptr:ptr.iter().zip(ptr[1..].iter()) }
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

////////////////////////////////////////////////////////////


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
            dflt : dflt,
            n}
    }

    #[allow(dead_code)]
    pub fn at(&self,i : usize) -> Option<&T> {
        let mut index = unsafe { * self.bucket.get_unchecked(hash(i) % self.bucket.len()) };

        while index < usize::MAX || i == unsafe { * self.index.get_unchecked(index) }  {
            index = unsafe{ * self.next.get_unchecked(index) };
        }

        if index < usize::MAX {
            Some(unsafe { &* self.data.get_unchecked(index) })
        }
        else {
            None
        }
    }

    pub fn at_mut(&mut self, i : usize) -> &mut T {
        let head = unsafe { * self.bucket.get_unchecked(hash(i) % self.bucket.len()) };
        let mut index = head;

        while index < usize::MAX || i == unsafe { * self.index.get_unchecked(index) }  {
            index = unsafe{ * self.next.get_unchecked(index) };
        }

        if index < usize::MAX {
            unsafe { &mut * self.data.get_unchecked_mut(index) }
        }
        else {
            if self.n < self.next.len() {
                index = self.n; self.n += 1;
                unsafe { *self.next.get_unchecked_mut(index) = head; }
                unsafe { *self.index.get_unchecked_mut(index) = i; }
                unsafe { *self.bucket.get_unchecked_mut(head) = index; }

                unsafe { *self.data.get_unchecked_mut(index) = self.dflt; }
                unsafe { & mut *self.data.get_unchecked_mut(index) }
            }
            else {
                panic!("Hashmap is full");
            }
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
    if s1.len() < s0.len() {
        shape_eq_except(s1,s0,d)
    }
    else if s0.len() < s1.len() {
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
    }
    else {
        // println!("{}:{}: shape_eq_except({:?},{:?},{})",file!(),line!(),s0,s1,d);
        d >= s0.len() ||
            ( ( d   == 0        || shape_eq(&s0[..d],&s1[..d]) )
                && ( d+1 == s0.len() || shape_eq(&s0[d+1..],&s1[d+1..]) ) )
    }
}

////////////////////////////////////////////////////////////

// Convert between linear index and coordinate index

/// Convert linear index to coordinate index for a given shape.
///
pub struct ToCoord<const N : usize> {    
    strides : [usize; N],
}
/// Convert koordinate index to linear index for a given shape.
pub struct FromCoord<const N : usize> {
    strides : [usize; N],
}
impl<const N : usize> ToCoord<N> {
    pub fn new(shape : &[usize; N]) -> ToCoord<N> {
        let mut strides = [0usize; N];
        let _ = strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v * d});
        ToCoord{ strides }
    }
}
/// Create a function that converts from linear index to coordinate index.
///
/// # Examples
/// 
/// ```
/// use mosekmodel::utils::*;
/// let shape = [2,4,3];
/// let lindexes = &[1,5,10,0,22,18,6];
/// let cindexes : Vec<[usize; 3]> = lindexes.iter().map(to_coord(&shape)).collect();
/// ```
pub fn to_coord<const N : usize,F>(shape:&[usize; N]) -> F where F : Fn(usize) -> [usize; N] { 
    let mut strides = [0usize; N];
    let _ = strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v * d});

    | i | {
        let mut r = [0; N];
        let _ = r.iter_mut().zip(strides.iter()).fold(i,|i,(r,&s)| unsafe{ *r = i/s; i % s } );
        r
    }
}


/// Create a function that converts from coordinate index to linear index.
///
/// # Examples
/// 
/// ```
/// use mosekmodel::utils::*;
/// let shape = [2,4,3];
/// let cindexes = &[ [0,0,0],
///                   [1,3,2],
///                   [1,2,1],
///                   [0,2,0] ];
/// let lindexes : Vec<usize> = cindexes.iter().map(from_coord(shape)).collect();
/// ```
pub fn from_coord<const N : usize,F>(shape:&[usize; N]) -> F where F : Fn(&[usize; N]) -> usize {
    let mut strides = [0usize; N];
    let _ = strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v * d});

    | i | strides.iter().zip(shape.iter()).fold(0,|v,(&st,&sh)| v + st*sh)
}


/// Create a function that perform a permutation on a fixed size array.
/// 
/// # Examples
///
/// ```
/// use mosekmodel::utils::perm_coord;
/// let perm : [usize;4] = [3,1,0,2];
/// let index : &[[usize; 4]] = &[ [ 0,0,0,0 ],
///                                [ 0,0,0,1 ],
///                                [ 0,0,2,0 ],
///                                [ 0,3,0,0 ],
///                                [ 4,0,0,0 ] ];
/// let pindex : Vec<[usize; 4]> = index.iter().map(perm_coord(&perm)).collect();
/// ```
pub fn perm_coord<const N : usize,T,F>(perm : &[usize; N]) -> Result<F,String> 
    where 
        F : Fn(&[T; N]) -> [T; N],
        T : Default
{
    // check that it is a permutation
    let mut check = [false; N];
    for (i,&p) in perm.iter().enumerate() {
        if p > N { return Err("Invalid permuration".to_string()); }
        if unsafe { *check.get_unchecked(p) } {
            return Err("Invalid permuration".to_string());
        }
        else {
            unsafe{ *check.get_unchecked_mut(p) = true };
        }
    }

    Ok(| i | { 
        let mut r = [T::default(); N]; 
        for (r,&p) in izip!(r.iter_mut(),i.iter(),perm.iter()) {
            *r = unsafe{ i.get_unchecked(p) };
        }

        r
    })
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
}
}
