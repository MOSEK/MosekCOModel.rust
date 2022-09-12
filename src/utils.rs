

struct Mutation {
    tgtsize : usize,
    idxs : Vec<usize>
}

struct MutationIter<'a,'b,T> {
    i : usize,
    m : & 'a Mutation,
    t : & 'b [T]
}
// struct MutationIterMut<'a,'b,T> {
//     i : usize,
//     m : & 'a Mutation,
//     t : & 'b mut [T]
// }


impl Mutation {
    pub fn new(idxs : &[usize]) -> Mutation {
        let &n = idxs.iter().max().unwrap_or(&0);

        Mutation{
            tgtsize : n,
            idxs : idxs.to_vec()
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

    // pub fn sort_by_key<F,T:Ord>(& mut self,f : F) where
    //     F : FnMut(&usize) -> &T {
    //     self.idxs.sort_by_key(f);
    // }

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
            m : self,
            t : target
        }
    }
    // pub fn apply_mut<'a,'b,T>(& 'a self, target : & mut 'b [T]) -> AppliedMutationMut {
    //     AppliedMutationMut {
    //         m : self,
    //         t : target
    //     }
    // }
}

impl<'a,'b,T> std::iter::Iterator for MutationIter<'a,'b,T> {
    type Item = & 'b T;
    fn next(& mut self) -> Option<& 'b T> {
        if self.i < self.m.idxs.len() {
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
