#![allow(unused)]

use itertools::izip;
use mosekcomodel::utils::iter::ChunksByIterExt;
use mosekcomodel::utils::*;


#[derive(Default)]
pub struct MatrixStore {
    ptr  : Vec<usize>,
    len  : Vec<usize>,
    subj : Vec<usize>,
    cof  : Vec<f64>,
    b    : Vec<f64>,

    map  : Vec<usize>
}

impl MatrixStore {
    pub fn new() -> MatrixStore { Default::default() }
    pub fn append_row(&mut self, subj : &[usize], cof : &[f64], b : f64) -> usize {        
        assert_eq!(subj.len(),cof.len());
        self.len.push(subj.len());

        let res = self.map.len();
        self.map.push(self.ptr.len());
        self.ptr.push(self.subj.len());

        self.subj.extend_from_slice(subj);
        self.cof.extend_from_slice(cof);
        self.b.push(b);
        
        res
    }

    pub fn num_row(&self) -> usize { self.map.len() }

    pub fn get<'a>(&'a self, i : usize) -> Option<(&'a[usize],&'a[f64],f64)> {
        self.map.get(i)
            .map(|&i| {
                let p = unsafe{*self.ptr.get_unchecked(i)};
                let l = unsafe{*self.len.get_unchecked(i)};
                
                (unsafe{self.subj.get_unchecked(p..p+l)},
                 unsafe{self.cof.get_unchecked(p..p+l)},
                 unsafe{*self.b.get_unchecked(i)})
            })
    }
    pub fn get_mut<'a>(&'a mut self, i : usize) -> Option<(&'a mut[usize],&'a mut [f64],&'a mut f64)> {
        self.map.get(i)
            .map(|&i| {
                let p = unsafe{*self.ptr.get_unchecked(i)};
                let l = unsafe{*self.len.get_unchecked(i)};
                
                (unsafe{self.subj.get_unchecked_mut(p..p+l)},
                 unsafe{self.cof.get_unchecked_mut(p..p+l)},
                 unsafe{self.b.get_unchecked_mut(i)})
            })
    }


    pub fn num_nonzeros(&self) -> usize {
        self.len.permute_by(self.map.as_slice()).sum()
    }

    
    pub fn replace_row(&mut self, i : usize, subj: &[usize], cof : &[f64],b : f64) {
        assert_eq!(subj.len(),cof.len());
        assert!(i < self.map.len());

        let rowi = unsafe{self.map.get_unchecked_mut(i)};
        let leni = unsafe{self.len.get_unchecked_mut(*rowi)};
        let ptri = unsafe{self.ptr.get_unchecked(*rowi)};
        if subj.len() <= *leni {
            *leni = subj.len();
            self.subj[*ptri..*ptri+*leni].copy_from_slice(subj);
            self.cof[*ptri..*ptri+*leni].copy_from_slice(cof);
        }
        else {
            *rowi = self.len.len();
            self.len.push(subj.len());
            self.ptr.push(self.subj.len());
            self.subj.extend_from_slice(subj);
            self.cof.extend_from_slice(cof);
        }
        self.b[*rowi] = b;
    }
    pub fn replace_rows(&mut self, rows : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64], b : &[f64]) {
        if !rows.is_empty() {
            assert_eq!(subj.len(),cof.len());
            assert_eq!(ptr.len(),rows.len()+1);
            assert_eq!(b.len(),rows.len());
            assert!(ptr.iter().zip(ptr[1..].iter()).all(|(a,b)| *a <= *b));
            assert_eq!(*ptr.last().unwrap(),subj.len());
            assert!(*rows.iter().max().unwrap() < self.map.len());

            for (rowi,subj,cof,b) in izip!(self.map.permute_by_mut(rows),subj.chunks_ptr(ptr),cof.chunks_ptr(ptr),b.iter()) {
                let leni = unsafe{self.len.get_unchecked_mut(*rowi)};
                let ptri = unsafe{self.ptr.get_unchecked(*rowi)};
                if subj.len() <= *leni {
                    *leni = subj.len();
                    self.subj[*ptri..*ptri+*leni].copy_from_slice(subj);
                    self.cof[*ptri..*ptri+*leni].copy_from_slice(cof);
                }
                else {
                    *rowi = self.len.len();
                    self.len.push(subj.len());
                    self.ptr.push(self.subj.len());
                    self.subj.extend_from_slice(subj);
                    self.cof.extend_from_slice(cof);
                }
                self.b[*rowi] = *b;
            }
        }
    }

    pub fn row_iter<'a>(&'a self) -> impl Iterator<Item=(&'a [usize],&'a[f64],f64)> {
        let perm = Permutation::from(&self.map);


        izip!(self.ptr.permute_by(perm),
              self.len.permute_by(perm),
              self.b.permute_by(perm))
            .map(|(&p,&l,&b)| {
                (unsafe{self.subj.get_unchecked(p..p+l)},
                 unsafe{self.cof.get_unchecked(p..p+l)},
                 b)
            })
    }

    pub fn eval_into(&self, x : &[f64], res : &mut Vec<f64>) -> Result<(),()> {        
        let perm = Permutation::from(&self.map);

        for (&p,&l,&b) in izip!(self.ptr.permute_by(perm),
                                self.len.permute_by(perm),
                                self.b.permute_by(perm))
        {
            let subj = unsafe{ self.subj.get_unchecked(p..p+l) };
            let cof  = unsafe{ self.cof.get_unchecked(p..p+l) };
            if subj.iter().max().map(|&v| v >= x.len()).unwrap_or(false) { return Err(()); }
            res.push(x.permute_by(subj).zip(cof.iter()).map(|(&a,&b)| a*b).sum::<f64>()+b);
        }
        Ok(())
    }
    
}
