#[allow(dead_code)]
pub mod iter;

use itertools::izip;


#[derive(Debug,Clone,Copy)]
pub struct Strides<const N : usize> {
    strides : [usize;N]
}

impl<const N : usize> Strides<N> {
    pub fn from_shape(shape : &[usize;N]) -> Strides<N> {
        let mut strides = [0usize; N]; 
        strides.iter_mut().zip(shape.iter()).rev().fold(1usize,|c,(t,&s)| { *t = c; c*s });
        Strides{ strides }
    }
    pub fn to_array(&self) -> [usize;N] { self.strides }
    pub fn to_linear(&self, index : &[usize;N]) -> usize {
        index.iter().zip(self.strides.iter()).map(|(a,b)| a*b).sum()
    }
    pub fn to_index(&self, i : usize) -> [usize;N] {
        let mut r = [0usize;N];
        r.iter_mut().zip(self.strides.iter()).fold(i,|i,(r,&s)| { *r = i/s; i%s} );
        r
    }
    pub fn iter(&self) -> std::slice::Iter<usize> { self.strides.iter() }
}


pub trait ShapeToStridesEx<const N : usize> {
    fn to_strides(&self) -> Strides<N>;
}

impl<const N : usize> ShapeToStridesEx<N> for [usize;N] {
    fn to_strides(&self) -> Strides<N> {
        Strides::from_shape(self)
    }
}





pub trait Cummulate {
    fn cummulate(& mut self);
}

impl<T> Cummulate for [T] where
    T : Copy+std::ops::AddAssign
{
    fn cummulate(& mut self) {
        if ! self.is_empty() {
            let v0 = self[0];
            self[1..].iter_mut().fold(v0,|c,v| { *v += c; *v });
        }
    }
}

impl<T> Cummulate for Vec<T> where
    T : Copy+std::ops::AddAssign
{
    fn cummulate(& mut self) {
        if ! self.is_empty() {
            let v0 = self[0];
            self[1..].iter_mut().fold(v0,|c,v| { *v += c; *v });
        }
    }
}







/// A trait that supplies functionality for appending self to a string.
pub trait NameAppender {
    /// Append self to a string
    fn append_to_string(&self, s : & mut String);
}

impl<T> NameAppender for [T] where T : NameAppender {
    fn append_to_string(&self, s : & mut String) {
        s.push('[');
        if self.len() > 0 {
            self[0].append_to_string(s);
            for i in self[1..].iter() { s.push(','); i.append_to_string(s) }
        }
        s.push(']');
    }
}

impl NameAppender for usize {
    fn append_to_string(&self, s : & mut String) {
        let mut v = *self;
        let w = *self/10;
        let mut n = 1usize; while n < w { n *= 10; }
        while n > 0 {
            s.push(((v / n) + '0' as usize) as u8 as char);
            v %= n;
            n /= 10;
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

