







struct SquareMatrix<const N : usize> {
    data : [[f64;N];N]
}


impl<const N : usize> SquareMatrix<N> {
    pub fn zero() -> Self { SquareMatrix{ data : [[0.0;N];N] } }
    pub fn eye() -> Self { let mut R = Self::zero(); for (i,row) in R.data.iter_mut().enumerate() { row[i] = 1.0; } }
    pub fn new(data : &[[ f64; N]; N]) -> Self { SquareMatrix{ data : *data } }

    pub fn get(&self) -> [[f64;N];N] { return self.data } 
    pub fn inplace_swap_rows(& mut self, i : usize, j : usize) {
        self.data.swap(i,j);
    }
    pub fn inplace_swap_cols(&mut self, i : usize, j : usize) {
        self.data.iter_mut(|row| row.swap(i,j) );
    }

    #[allow(non_snake_case)]
    pub fn transpose(&self) -> Self {
        let mut R = Self::zero();
        for (i,trow) in R.data.iter_mut().enumerate() {
            for (j,t) in trow.iter_mut().enumerate() {
                *t = self.data[j][i];
            }
        }
        R
    }
    pub fn try_left_divide_matrix(&self, rhs : & mut Self) -> Result<(),String> {
        try_left_divide(& self.data,& mut rhs.data)
    }

    #[allow(non_snake_case)]
    pub fn try_invert(&self) -> Result<Self,String> {
        let mut R = Self::eye();
        self.try_left_divide_matrix(& mut R)?;
        Ok(R)
    }
}

#[allow(non_snake_case)]
fn swap_rows<const N : usize>
(   i   : usize,
    j   : usize,
    P   : &mut [[f64;N];N], 
    res : &mut [[f64;N];N]) {
   
    P.swap(i,j);
    res.swap(i,j);
}

fn inplace_mul_row<const N : usize>
(   i : usize,
    c : f64,
    P : &mut [[f64;N];N], 
    R : &mut [[f64;N];N]) {
   
    (&mut P[i]).iter_mut().for_each(|t| *t *= c);
    (&mut R[i]).iter_mut().for_each(|t| *t *= c);
}

#[allow(non_snake_case)]
fn inplace_add_rows<const N : usize>
(   i : usize,
    j : usize,
    c : f64,
    P : &mut [[f64;N];N], 
    R : &mut [[f64;N];N]) {
  
    vec_inplace_add(& mut P[i], c, &P[j]);
    vec_inplace_add(& mut R[i], c, &R[j]);
}

fn vec_inplace_add<const N : usize>(rhs : & mut [f64;N], c : f64, lhs : &[f64;N]) {
    rhs.iter_mut().zip(lhs.iter()).for_each(|(t,&s)| *t += c * s);
}

#[allow(non_snake_case)]
fn try_left_divide<const N : usize>(P : &[[f64;N];N], res : & mut [[f64;N];N]) -> Result<(),String> {
    let mut P = *P;

    for i in 0..N {
        // find first row with non-zero in column i
        let (k,_) = (i..).zip(P[i..].iter()).find(|item| item.1[i] < 0 || 0 < item.1[i]).ok_or(Err("Not invertible".to_string()))?;
        if k != i { swap_rows(i,k,& mut P, & mut res) }
        
        let c = 1.0/P[i][i];
        inplace_mul_row(i,c,& mut P, & mut R);

        for j in i+1.. {
            let c = P[j][i];
            if c < 0.0 || c > 0.0 {
                inplace_add_rows(j,i,-1.0/c,& mut P, & mut R);
            }
        }
    }
    // P should now be triu
    for i in (1..N).rev() {
        for j in 0..i {
            let c = P[j][i];
            if c < 0.0 || c > 0.0 {
                inplace_add_rows(j,i,-1.0/c,& mut P, & mut R);
            }
        }
    }
    // P should now be I and res is the inverse of the original P
    Ok(())
}

#[allow(non_snake_case)]
fn try_invert<const N : usize>( P : [[f64;N];N]) -> [[f64;N];N] {
    let R = [[0.0;N];N];
    for (i,r) in R.iter_mut().enumerate() { r[i] = 1.0 }

    try_left_divide()
}

pub fn transpose<const N : usize>(P : & mut [[f64;N];N]) {
    let Px = *P;
    for (i,trow) in P.iter_mut().enumerate() {
        for (j,t) in trow.iter_mut().enumerate() {
            *t = Px[j][i];
        }
    }
}



