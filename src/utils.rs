//!
//! Utility module.
//!

use std::cmp::Ordering;

use itertools::izip;


/// Trait that provides a function that copies from an iterator into a slice.
///
/// # Notes
/// Implementations should not fail if the lengths do not match, but rather copy the maximum number
/// possible and return that number.
pub trait AssignFromIterExt<I> where I : Iterator, I::Item : Copy 
{
    #[allow(dead_code)]
    fn copy_from_iter(&mut self, it : I) -> usize;
}

impl<I> AssignFromIterExt<I> for [I::Item]
    where 
        I : Iterator,
        I::Item : Copy
{
    fn copy_from_iter(&mut self, it : I) -> usize {
        self.iter_mut().zip(it).map(|(t,s)| *t = s).count()
    }
}

impl<I> AssignFromIterExt<I> for Vec<I::Item>
    where 
        I : Iterator,
        I::Item : Copy
{
    fn copy_from_iter(&mut self, it : I) -> usize {
        self.iter_mut().zip(it).map(|(t,s)| *t = s).count()
    }
}

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
            Some(unsafe{self.data.get_unchecked(i)})
        }
        else {
            None
        }
    }
}

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


// /// Given a shape, and optionally a sparsity pattern, call a function
// /// f with each index in the shape by lexicalic order.
// ///
// /// Arguments:
// /// - shape The shape
// /// - sp Optional sparsity pattern
// /// - f Function called for each index in the shape
// pub fn for_each_index<F>(shape : &[usize], sp : Option<&[usize]>, mut f : F) where F : FnMut(usize,&[usize]) {
//     let mut idx = vec![0;shape.len()];
// 
//     if let Some(sp) = sp {
//         let mut stride : Vec<usize> = vec![0; shape.len()];
//         let _ = shape.iter().rev().zip(stride.iter_mut().rev()).fold(1,|k,(&d,s)| { *s = k; k*d });
//         for &i in sp {
//             let _ = idx.iter_mut().zip(stride.iter()).fold(i,|k,(ix,&s)| { *ix = k / s; k % s } );
//             f(i,idx.as_slice());
//         }
//     }
//     else {
//         for i in 0..shape.iter().product() {
//             f(i,idx.as_slice());
//             let _ = shape.iter().zip(idx.iter_mut()).rev()
//                 .fold(1,|carry,(&d,ix)| {
//                     if carry == 0 { 0 }
//                     else {
//                         *ix += carry;
//                         if *ix < d { 0 }
//                         else { *ix = 0; 1 }
//                     }
//                 });
//         }
//     }
// }

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



pub fn shape_to_strides<const N : usize>(shape : &[usize;N]) -> [usize;N] {
    let mut res = [0usize; N];
    _ = res.iter_mut().zip(shape.iter()).rev().fold(1,|c,(r,&d)| { *r = c; c * d });
    res
}
pub fn key_from_strides_index<const N : usize>(strides : [usize;N], index : usize) -> [usize;N] {
    let mut res = [0usize; N];
    _ = res.iter_mut().zip(strides.iter()).fold(index,|i,(r,&s)| { *r = i / s; i % s });
    res
}

//pub struct MergeIter<I0,I1,J> where 
//    I0 : Iterator<Item = J>,
//    I1 : Iterator<Item = J>,
//    J : PartialOrd
//{
//    i0 : Peekable<I0>,
//    i1 : Peekable<I1>
//}
//
//impl<I0,I1,J> MergeIter<I0,I1,J> where 
//    I0 : Iterator<Item = J>,
//    I1 : Iterator<Item = J>,
//    J : PartialOrd
//{
//    pub fn new(i0 : I0, i1 : I1) -> MergeIter<I0,I1,J> { MergeIter{ i0: i0.peekable(), i1 : i1.peekable() } } 
//}
//
//
//impl<I0,I1,J> Iterator for MergeIter<I0,I1,J> where 
//    I0 : Iterator<Item = J>,
//    I1 : Iterator<Item = J>,
//    J : PartialOrd
//{
//    type Item = J;
//    fn next(& mut self) -> Option<J> {
//        match (self.i0.peek(),self.i1.peek()) {
//            (Some(v0),Some(v1)) => if v0 <= v1 { self.i0.next() } else { self.i1.next() },
//            (_,None) => self.i0.next(),
//            (None,_) => self.i1.next()
//        }
//    }
//}


////////////////////////////////////////////////////////////

// Convert between linear index and coordinate index
// /// Create a function that converts from linear index to coordinate index.
// ///
// /// # Examples
// /// 
// /// ```
// /// use mosekmodel::utils::*;
// /// let shape = [2,4,3];
// /// let lindexes = &[1,5,10,0,22,18,6];
// /// let cindexes : Vec<[usize; 3]> = lindexes.iter().map(to_coord(&shape)).collect();
// /// ```
// pub fn to_coord<const N : usize>(shape:&[usize; N]) -> impl Fn(usize) -> [usize; N] { 
//     let mut strides = [0usize; N];
//     let _ = strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v * d});
//     
//     | i | -> [usize;N] {
//         let mut r = [0; N];
//         let _ = r.iter_mut().zip(strides.iter()).fold(i,|i,(r,&s)| unsafe{ *r = i/s; i % s } );
//         r;
//     }
// }
// 
// 
// /// Create a function that converts from coordinate index to linear index.
// ///
// /// # Examples
// /// 
// /// ```
// /// use mosekmodel::utils::*;
// /// let shape = [2,4,3];
// /// let cindexes = &[ [0,0,0],
// ///                   [1,3,2],
// ///                   [1,2,1],
// ///                   [0,2,0] ];
// /// let lindexes : Vec<usize> = cindexes.iter().map(from_coord(shape)).collect();
// /// ```
// pub fn from_coord<const N : usize,F>(shape:&[usize; N]) -> F where F : Fn(&[usize; N]) -> usize {
//     let mut strides = [0usize; N];
//     let _ = strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v * d});
// 
//     | i | strides.iter().zip(shape.iter()).fold(0,|v,(&st,&sh)| v + st*sh)
// }
// 
// 
// /// Create a function that perform a permutation on a fixed size array.
// /// 
// /// # Examples
// ///
// /// ```
// /// use mosekmodel::utils::perm_coord;
// /// let perm : [usize;4] = [3,1,0,2];
// /// let index : &[[usize; 4]] = &[ [ 0,0,0,0 ],
// ///                                [ 0,0,0,1 ],
// ///                                [ 0,0,2,0 ],
// ///                                [ 0,3,0,0 ],
// ///                                [ 4,0,0,0 ] ];
// /// let pindex : Vec<[usize; 4]> = index.iter().map(perm_coord(&perm)).collect();
// /// ```
// pub fn perm_coord<const N : usize,T,F>(perm : &[usize; N]) -> Result<F,String> 
//     where 
//         F : Fn(&[T; N]) -> [T; N],
//         T : Default
// {
//     // check that it is a permutation
//     let mut check = [false; N];
//     for (i,&p) in perm.iter().enumerate() {
//         if p > N { return Err("Invalid permuration".to_string()); }
//         if unsafe { *check.get_unchecked(p) } {
//             return Err("Invalid permuration".to_string());
//         }
//         else {
//             unsafe{ *check.get_unchecked_mut(p) = true };
//         }
//     }
// 
//     Ok(| i | { 
//         let mut r = [T::default(); N]; 
//         for (r,&p) in izip!(r.iter_mut(),i.iter(),perm.iter()) {
//             *r = unsafe{ i.get_unchecked(p) };
//         }
// 
//         r
//     })
// }
// 

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
