// This module defines the generic expression evaluation, i.e. the
// evaluation part that is not concerned with expression sub-part
// types.

use std::{iter::once};
use super::*;
use super::workstack::WorkStack;
use utils::{*,iter::*};

use itertools::{izip,iproduct};

pub fn diag(anti : bool, index : i64, rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> { 
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let nd = shape.len();

    if nd != 2 || shape[0] != shape[1] {
        return Err(ExprEvalError::new(file!(), line!(), "Diagonals can only be taken from square matrixes"));
    }
    let d = shape[0];
    if index.unsigned_abs() as usize >= d {
        return Err(ExprEvalError::new(file!(),line!(),"Diagonal index out of bounds"));
    }

    let absidx = index.unsigned_abs() as usize;
    if let Some(sp) = sp {
        let (first,num) = match (anti,index >= 0) {
            (false,true)  => (index as usize,       d - absidx),
            (false,false) => (d*(-index) as usize,  d - absidx),
            (true,true)   => (d-index as usize,     d - absidx),
            (true,false)  => (d*(-index) as usize-1,d - absidx)
        };
        let last = num*d;
        // Count elements and nonzeros
        let (rnnz,rnelm) = izip!(sp.iter(),
                               ptr.iter(),
                               ptr[1..].iter())
            .filter(|(&i,_,_)| (i < last && 
                                (((!anti) && index >= 0 && i%d == i/d + absidx) ||
                                 ((!anti) && index <  0 && i%d - absidx == i/d) || 
                                 ( anti && index >= 0 && d-i%d - absidx == i/d) || 
                                 ( anti && index <  0 && d-i%d + absidx == i/d))))
            .fold((0,0),|(nzi,elmi),(_,&p0,&p1)| (nzi+p1-p0,elmi+1));

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&[d],rnnz,rnelm);

        let mut nzi = 0;
        rptr[0] = 0;
        if let Some(rsp) = rsp {
            izip!(sp.iter(),ptr.iter(),ptr[1..].iter())
                .filter(|(&i,_,_)| (i < last && 
                                    ((!anti && index >= 0 && i%d == i/d + absidx) ||
                                     (!anti && index <  0 && i%d - absidx == i/d) || 
                                     ( anti && index >= 0 && d-i%d - absidx == i/d) || 
                                     ( anti && index <  0 && d-i%d + absidx == i/d))))
                .zip(rptr[1..].iter_mut().zip(rsp.iter_mut()))
                .for_each(|((&i,&p0,&p1),(rp,ri))| {
                    *rp = p1-p0;
                    *ri = (i-first)/d;
                    rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                    nzi += p1-p0;
                })
        }
        else {
            izip!(sp.iter(),ptr.iter(),ptr[1..].iter())
                .filter(|(&i,_,_)| (i < last && 
                                    ((!anti && index >= 0 && i%d == i/d + absidx) ||
                                     (!anti && index <  0 && i%d - absidx == i/d) || 
                                     ( anti && index >= 0 && d-i%d - absidx == i/d) || 
                                     ( anti && index <  0 && d-i%d + absidx == i/d))))
                .zip(rptr[1..].iter_mut())
                .for_each(|((_,&p0,&p1),rp)| {
                    *rp = p1-p0;
                    rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                    nzi += p1-p0;
                })
        }   
    } 
    else {
        let (first,num,step) = match (anti,index >= 0) {
            (false,true)  => (absidx,    d-absidx, d+1),
            (false,false) => (d*absidx,  d-absidx, d+1),
            (true,true)   => (d-absidx,  d-absidx, d-1),
            (true,false)  => (d*absidx-1,d-absidx, d-1)
        };
        
        let rnnz = izip!(0..num,
                         ptr[first..].iter().step_by(step),
                         ptr[first+1..].iter().step_by(step))
                       .map(|(_,&p0,&p1)| p1-p0).sum();
        let rnelm = num;
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[num],rnnz,rnelm);
        rptr[0] = 0;
        let mut nzi = 0;
        izip!(rptr[1..].iter_mut(),
             ptr[first..].iter().step_by(step),
             ptr[first+1..].iter().step_by(step))
            .for_each(|(rp,&p0,&p1)| {
                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                *rp = p1-p0;
                nzi += p1-p0;
            });
        let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p } );

    }
    rs.check();
    Ok(())
}

pub fn triangular_part(upper : bool, with_diag : bool, rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> { 
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let nd = shape.len();

    if nd != 2 || shape[0] != shape[1] {
        return Err(ExprEvalError::new(file!(), line!(), "Triangular parts can only be taken from square matrixes"));
    }
    let d = shape[0];

    if let Some(sp) = sp {
        let rnelm = sp.iter()
            .filter(|&spi| { let (i,j) = (spi / d, spi % d); (upper && i < j) || (! upper && i > j) || (with_diag && i == j) })
            .count(); 
        let rnnz = izip!(sp.iter().map(|i| (i/d,i%d)),ptr.iter(),ptr[1..].iter())
            .filter(|((i,j),_,_)| (upper && i < j) || (! upper && i > j) || (with_diag && i == j) )
            .map(|(_,pb,pe)| pe-pb)
            .sum();
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape,rnnz,rnelm);

        izip!(subj.chunks_ptr(ptr),
              cof.chunks_ptr(ptr),
              sp.iter().map(|i| (i/d,i%d)))
            .filter(|(_,_,(i,j))| (upper && i < j) || (! upper && i > j) || (with_diag && i == j))
            .flat_map(|(subj,cof,_)| subj.iter().zip(cof.iter()))
            .zip(rsubj.iter_mut().zip(rcof.iter_mut()))
            .for_each(|((&sj,&sc),(tj,tc))| { *tj = sj; *tc = sc; });

        rptr[0] = 0;
        let rsp = rsp.unwrap();
        izip!(ptr.iter(),ptr[1..].iter(),sp.iter().map(|i| (i/d,i%d)))
            .filter(|(_,_,(i,j))| (upper && i < j) || (! upper && i > j) || (with_diag && i == j))
            .zip(rptr[1..].iter_mut().zip(rsp.iter_mut()))
            .for_each(|((&pb,&pe,(i,j)),(rp,spi))|  { *rp = pe-pb; *spi = i * d + j } );
        rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
    }
    else {  
        //println!("  case 2: Dense");
        let rnelm = if with_diag { d * (d+1)/2 } else { d * (d-1) / 2 };
        let rnnz : usize = match (upper,with_diag) {
            (true,true)   => ptr.iter().step_by(d+1).zip(ptr[d..].iter().step_by(d)).map(|(&p0,&p1)| p1-p0).sum::<usize>(),
            (true,false)  => ptr[1..].iter().step_by(d+1).zip(ptr[d..].iter().step_by(d)).map(|(&p0,&p1)| p1-p0).sum::<usize>(),
            (false,true)  => ptr.iter().step_by(d).zip(ptr[1..].iter().step_by(d+1)).map(|(&p0,&p1)| p1-p0).sum::<usize>(),
            (false,false) => ptr[d..].iter().step_by(d).zip(ptr[d+1..].iter().step_by(d+1)).map(|(&p0,&p1)| p1-p0).sum::<usize>()
        };

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape,rnnz,rnelm);

        izip!(subj.chunks_ptr(ptr),
              cof.chunks_ptr(ptr),
              iproduct!(0..d,0..d))
            .filter(|(_,_,(i,j))| 
                (upper     && i  < j) ||
                (! upper   && i  > j) ||
                (with_diag && i == j))
            .flat_map(|(subj,cof,_)| subj.iter().zip(cof.iter()))
            .zip(rsubj.iter_mut().zip(rcof.iter_mut()))
            .for_each(|((&sj,&sc),(tj,tc))| { *tj = sj; *tc = sc; });

        rptr[0] = 0; rptr.iter_mut().for_each(|v| *v = 0);
        let rsp = rsp.unwrap();
        izip!(ptr.iter(),ptr[1..].iter(),iproduct!(0..d,0..d))
            .filter(|(_,_,(i,j))| 
                (upper     && i  < j) ||
                (! upper   && i  > j) ||
                (with_diag && i == j))
            .zip(rptr[1..].iter_mut().zip(rsp.iter_mut()))
            .for_each(|((&pb,&pe,(i,j)),(rp,spi))| { 
                *rp = pe-pb; 
                *spi = i * d + j;
            });
        rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
    }
    rs.check();
    Ok(())
}

pub fn sum(rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> { 
    let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();
    let (rptr,_rsp,rsubj,rcof)    = rs.alloc_expr(&[],*ptr.last().unwrap(),1);
    rptr[0] = 0;
    rptr[1] = *ptr.last().unwrap();
    rsubj.clone_from_slice(subj);
    rcof.clone_from_slice(cof);
    rs.check();
    Ok(())
}


pub fn slice(begin : &[usize], end : &[usize], rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> { 
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nnz = *ptr.last().unwrap();
    let nelem = ptr.len()-1;
    let n = shape.len();

    // check indexes
    assert!(n == begin.len() && n == end.len());
    if shape.iter().zip(end.iter()).any(|(a,b)| a < b) { 
        return Err(ExprEvalError::new(file!(),line!(),"Index out of bounds"));
    }

    let (urest,fpart) = xs.alloc(n*5+nelem*2+1+nnz,nnz);
    let (strides,urest) = urest.split_at_mut(n); 
    let (rstrides,urest) = urest.split_at_mut(n); 
    let (rshape,urest) = urest.split_at_mut(n); 
    let (ix,urest) = urest.split_at_mut(n); 
    let (ix2,upart) = urest.split_at_mut(n); 

    let _ = strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v*d } );
    let _ = izip!(rstrides.iter_mut(),begin.iter(),end.iter()).rev().fold(1,|v,(s,&b,&e)| { *s = v; v*(e-b) } );

    izip!(rshape.iter_mut(),begin.iter(),end.iter()).for_each(|(r,&b,&e)| *r = e-b );

    if let Some(sp) = sp {
        let xcof = fpart;
        let (xptr,upart) = upart.split_at_mut(nelem+1);
        let (xsp,xsubj) = upart.split_at_mut(nelem);
        xptr[0] = 0;

        let mut rnnz = 0usize;
        let mut rnelem = 0usize;
        for ((&spi,&p0,&p1),(xp,xs)) in 
            izip!(sp.iter(), ptr.iter(), ptr[1..].iter())
                .filter(|(&spi,_,_)| {
                    ix.iter_mut().for_each(|v| *v = 0);
                    let _ = ix.iter_mut().zip(strides.iter()).fold(spi,|v,(i,&s)| { *i = v / s; v % s});
                   izip!(ix.iter(),begin.iter(),end.iter()).all(|(&i,&b,&e)| b <= i && i < e ) })
                .zip(xptr[1..].iter_mut().zip(xsp.iter_mut())) {

            ix2.iter_mut().for_each(|v| *v = 0);
            let _ = ix2.iter_mut().zip(strides.iter()).fold(spi,|v,(i,&s)| { *i = v / s; v % s});
           
            (*xs,_) = izip!(strides.iter(),rstrides.iter(),begin.iter()).fold((spi,0),|(i,r),(&s,&rs,&b)| (i%s, r + (i/s-b) * rs) );
            xsubj.clone_from_slice(&subj[p0..p1]);
            xcof.clone_from_slice(&cof[p0..p1]);
            rnnz += p1-p0;                    
            *xp = rnnz;
            rnelem += 1;
        }

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rshape, rnnz, rnelem);
        rptr[0] = 0;
        rptr.clone_from_slice(&xptr[..rnelem+1]);
        rsubj.clone_from_slice(&xsubj[..rnnz]);
        rcof.clone_from_slice(&xcof[..rnnz]);
        if let Some(rsp) = rsp {
            rsp.clone_from_slice(&xsp[..rnelem]);
        }
    } 
    else {
        let rnelem : usize = begin.iter().zip(end.iter()).map(|(&a,&b)| b-a).product(); 
        let xcof = fpart;
        let (xptr,xsubj) = upart.split_at_mut(rnelem+1);
        let mut rnnz = 0usize;
        {
            xptr[0] = 0;
            ix.iter_mut().for_each(|v| *v = 0);

            for xp in xptr[1..].iter_mut() {
                let sofs = izip!(ix.iter(), strides.iter(), begin.iter())
                    .fold(0,|v,(&i,&s,&b)| v+(i+b)*s);

                let (&p0,&p1) = unsafe{ (ptr.get_unchecked(sofs),ptr.get_unchecked(sofs+1)) };
                
                xsubj[rnnz..rnnz+p1-p0].copy_from_slice(&subj[p0..p1]);
                xcof[rnnz..rnnz+p1-p0].copy_from_slice(&cof[p0..p1]);
                rnnz += p1-p0;
                *xp = rnnz;

                ix.iter_mut().zip(rshape.iter()).rev().fold(1,|carry,(i,&d)| { *i += carry; if *i >= d { *i = 0; 1 } else { 0 } });
            }
        }
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(rshape, rnnz, rnelem);
        rptr.clone_from_slice(xptr);
        rsubj.clone_from_slice(&xsubj[..rnnz]);
        rcof.clone_from_slice(&xcof[..rnnz]);
    }
    rs.check();
    Ok(())
}

pub fn repeat(dim : usize, num : usize, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    if dim >= shape.len() {
        return Err(ExprEvalError::new(file!(),line!(),"Invalid stacking dimension"));
    }
    let mut rshape = shape.to_vec();
    rshape[dim] *= num;
    let nelm = ptr.len()-1;
    let rnnz = ptr.last().unwrap()*num;
    let rnelm = (ptr.len()-1)*num;


    let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(), rnnz, rnelm);

    if let (Some(sp),Some(rsp)) = (sp,rsp) {
        let _d0 : usize = shape[..dim].iter().product();
        let d1 : usize = shape[dim];
        let d2 : usize = shape[dim+1..].iter().product();
        let rd1 = rshape[dim];
        let (uslice,_) = xs.alloc(rnelm*3,0);
        let (xsp,uslice) = uslice.split_at_mut(rnelm);
        let (xidx,perm)  = uslice.split_at_mut(rnelm);

        for (xspi,xi,(k,spi,i)) in izip!(xsp.iter_mut(),
                                         xidx.iter_mut(),
                                         (0..num).flat_map(|i| izip!(0..nelm,sp.iter(),std::iter::repeat(i)))) {
            let (i0,i1,i2) = (spi / (d1*d2), (spi / d2) % d1, spi % d2);
            *xspi = i0 * rd1 * d2 + (i1 + i * d1) * d2 + i2;
            *xi = k;
        }

        perm.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        perm.sort_by_key(|&i| xsp[i]);

        rptr.iter_mut().for_each(|p| *p = 0);
        rsp.iter_mut().zip(xsp.permute_by(perm)).for_each(|(t,&s)| *t = s);

        let mut p = 0usize;
        for (rptr,&i) in izip!(rptr[1..].iter_mut(), xidx.permute_by(perm)) {
            let ptrb = ptr[i];
            let ptre = ptr[i+1];
            let n = ptre-ptrb;
            *rptr = n;
            rsubj[p..p+n].copy_from_slice(&subj[ptrb..ptre]);
            rcof[p..p+n].copy_from_slice(&cof[ptrb..ptre]);
            p += n;
        }
        _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
    } 
    else { // dense
        let _d0 : usize = num * shape[..dim].iter().product::<usize>();
        let d1 : usize = shape[dim..].iter().product();
        rptr[0] = 0;
        let mut rptr_pos = 0usize;
        for ((ptrb,ptre),rptr) in izip!(ptr.chunks(d1),ptr[1..].chunks(d1)).map(|v| std::iter::repeat(v).take(num)).flatten().zip(rptr[1..].chunks_mut(d1)) {
            izip!(rptr.iter_mut(),ptrb.iter(),ptre.iter()).for_each(|(r,&pb,&pe)| *r = pe-pb);
            let pb = ptrb[0];
            let pe = *ptre.last().unwrap();
            let n = pe-pb;
            rsubj[rptr_pos..rptr_pos+n].copy_from_slice(&subj[pb..pe]);
            rcof[rptr_pos..rptr_pos+n].copy_from_slice(&cof[pb..pe]);
            rptr_pos += n;
        }

        _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
    }
    rs.check();
    Ok(())
}


pub fn permute_axes<const N : usize>(
    perm : &[usize;N],
    rs : & mut WorkStack,
    ws : & mut WorkStack,
    xs : & mut WorkStack) -> Result<(),ExprEvalError> 
{
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nelem = ptr.len()-1;

    if shape.len() != N || *perm.iter().max().unwrap() >= N {
        return Err(ExprEvalError::new(file!(),line!(),"Mismatching permutation and shape"));
    }
    let shape = { let mut r = [0usize; N]; r.copy_from_slice(shape); r };
    let mut rshape = [usize::MAX; N];
    rshape.iter_mut().zip(shape.permute_by(perm))
        .for_each(|(d,&s)| {
            if *d < usize::MAX {
                panic!("Invalid permutation {:?}",perm);
            } 
            else {
                *d = s;
            }
        });
    let nnz = subj.len();

    let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape,nnz,nelem);
    rptr[0] = 0;
        
    let (uslice,_)   = xs.alloc(nelem*2,0);
    let (spx,uslice) = uslice.split_at_mut(nelem);
    let (elmperm,_)  = uslice.split_at_mut(nelem);

    let strides = shape.to_strides();
    let rstrides = rshape.to_strides();

    let mut prstrides = [0usize;N];
    prstrides.permute_by_mut(perm).zip(rstrides.iter()).for_each(|(d,&s)| *d = s);

    if let Some(sp) = sp {
        spx.iter_mut().zip(sp.iter()).for_each(|(ix,&i)| {
            let (_,ri) = strides.iter().zip(prstrides.iter()).fold((i,0),|(v,r),(&s,&rs)| (v%s,r+(v/s)*rs));
            *ix = ri;
        });

        elmperm.iter_mut().enumerate().for_each(|(i,t)| *t = i);
        elmperm.sort_by_key(|&i| unsafe{* spx.get_unchecked(i) });
       
        // apply permutation
        if let Some(rsp) = rsp { 
            for (t,&s) in rsp.iter_mut().zip(sp.permute_by(elmperm)) {
                (_,*t) = izip!(strides.iter(),prstrides.iter()).fold((s,0),|(sv,r),(&s,&ps)| (sv % s,r + (sv/s)*ps)  );
            }
        };
        rptr.iter_mut().for_each(|p| *p = 0);

        { 
            let mut nzi = 0;
    
            for (&p0,&p1,p) in izip!(ptr[0..nelem].permute_by(elmperm),ptr[1..nelem+1].permute_by(elmperm),rptr[1..].iter_mut()) {
                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                *p = p1-p0;
                nzi += p1-p0;
            }
            _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p } );
        }
    }
    else {
        rptr.iter_mut().for_each(|v| *v = 0);
        for (si,n) in ptr.iter().zip(ptr[1..].iter()).map(|(&p0,&p1)| p1-p0).enumerate() {
            let (_,ti) = strides.iter().zip(prstrides.iter()).fold((si,0),|(v,r),(&s,&rs)| (v%s,r+(v/s)*rs));
            rptr[ti+1] = n
        }
        let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
        
        for (si,(ssubj,scof)) in izip!(subj.chunks_ptr(ptr),cof.chunks_ptr(ptr)).enumerate() {
            let (_,ti) = strides.iter().zip(prstrides.iter()).fold((si,0),|(v,r),(&s,&rs)| (v%s,r+(v/s)*rs));
            let n = ssubj.len();
            let nzi = rptr[ti];

            rsubj[nzi..nzi+n].clone_from_slice(ssubj);
            rcof[nzi..nzi+n].clone_from_slice(scof);
        }
    }
    rs.check();
    Ok(())
} // fn permute_axes


/// Add `n` expression residing on `ws`. Result pushed to `rs`.
pub fn add(n  : usize,
                  rs : & mut WorkStack,
                  ws : & mut WorkStack,
                  xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    // check that shapes match
    let exprs = ws.pop_exprs(n);

    exprs.iter().map(|(shape,_,_,_,_)| shape)
        .zip(exprs[1..].iter().map(|(shape,_,_,_,_)| shape))
        .find(|(shape1,shape2)| shape1 != shape2)
        .iter().for_each(|(shape1,shape2)|
            panic!("Mismatching operand shapes: {:?} and {:?}",shape1,shape2)
        );

    let (shape,_,_,_,_) = exprs.first().unwrap();

    // count result nonzeros
    let rnnz : usize = exprs.iter().map(|(_,_,_,subj,_)| subj.len()).sum();

    // check sparsity
    let has_dense = exprs.iter().any(|(_,_,sp,_,_)| sp.is_none() );

    if has_dense {
        let rnelm = shape.iter().product();
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(shape,rnnz,rnelm);
        // build rptr
        rptr.fill(0);
        for (_,ptr,sp,_,_) in exprs.iter() {
            if let Some(sp) = sp {
                izip!(ptr.iter(),ptr[1..].iter(),sp.iter())
                    .for_each(|(&p0,&p1,&i)| unsafe{ *rptr.get_unchecked_mut(i+1) += p1-p0 });
            }
            else {
                izip!(ptr.iter(),ptr[1..].iter(),rptr[1..].iter_mut())
                    .for_each(|(&p0,&p1,rp)| *rp += p1-p0);
            }
        }
        let _ = rptr.iter_mut().fold(0,|a,p| { *p += a; *p });

        for (_,ptr,sp,subj,cof) in exprs.iter() {
            if let Some(sp) = sp {
                izip!(subj.chunks_ptr(ptr),
                      cof.chunks_ptr(ptr),
                      sp.iter())
                    .for_each(|(js,cs,&i)| {
                        let nnz = js.len();
                        let p0 = unsafe{ *rptr.get_unchecked(i) };
                        rsubj[p0..p0+nnz].clone_from_slice(js);
                        rcof[p0..p0+nnz].clone_from_slice(cs);

                        unsafe{ *rptr.get_unchecked_mut(i) += nnz };
                    });
            }
            else {
                izip!(subj.chunks_ptr(ptr),
                      cof.chunks_ptr(ptr),
                      rptr.iter())
                    .for_each(|(js,cs,&p0)| {
                        let nnz = js.len();
                        rsubj[p0..p0+nnz].clone_from_slice(js);
                        rcof[p0..p0+nnz].clone_from_slice(cs);
                    });
                rptr.iter_mut().zip(ptr.iter().zip(ptr[1..].iter()).map(|(&p0,&p1)| p1-p0))
                    .for_each(|(rp,n)| *rp += n);
            }
        }
        // revert rptr
        let _ = rptr.iter_mut().fold(0,|prevp,p| { let nextp = *p; *p = prevp; nextp } );
    }
    else {
        let nelm_bound = if has_dense { shape.iter().product() } else { shape.iter().product::<usize>().max(exprs.iter().map(|(_,ptr,_,_,_)| ptr.len()-1).sum()) };
        let (uslice,_) = xs.alloc(nelm_bound*5,0);
        let (hdata,uslice)  = uslice.split_at_mut(nelm_bound);
        let (hindex,uslice) = uslice.split_at_mut(nelm_bound);
        let (hnext,uslice)  = uslice.split_at_mut(nelm_bound);
        let (hbucket,hperm) = uslice.split_at_mut(nelm_bound);
        hdata.fill(0);


        let (rptr,rsp,rsubj,rcof) = {
            // We use a hash map to count number of elements and nonzeros
            let mut h = IndexHashMap::new(hdata,hindex,hnext,hbucket,0);

            // count row nonzeros
            for (_,ptr,sp,_subj,_cof) in exprs.iter() {
                if let Some(sp) = sp {
                    izip!(sp.iter(),ptr.iter(),ptr[1..].iter()).for_each(|(&i,&p0,&p1)| *h.at_mut(i) += p1-p0);
                }
            }
            rs.alloc_expr(shape,rnnz,h.len())
        };

        let rnelm = rptr.len()-1;

        rptr[0] = 0;
        // Compute sorting permutation
        let perm = & mut hperm[..rnelm];
        perm.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        perm.sort_by_key(|&i| unsafe{* hindex.get_unchecked(i) });
        // copy data to solution
        if let Some(rsp) = rsp {
            //println!("len rptr: {}, rsp: {}, perm: {}",rptr.len(),rsp.len(),perm.len());
            for (rp,ri,&si,&sd) in izip!(rptr[1..].iter_mut(),
                                         rsp.iter_mut(),
                                         hindex.permute_by(perm),
                                         hdata.permute_by(perm)) {
                *ri = si;
                *rp = sd;
            }
            hindex[..rnelm].clone_from_slice(rsp);
        }
        else {
            for (rp,&sd) in izip!(rptr[1..].iter_mut(), hdata.permute_by(perm)) {
                *rp = sd;
            }
            hindex[..rnelm].iter_mut().enumerate().for_each(|(i,x)| *x = i);
        }
        // cummulate ptr
        _ = rptr.iter_mut().fold(0,|c,p| { *p += c; *p });
                 
        // Create a mapping from linear sparsity index into sp/ptr index
        hdata.iter_mut().enumerate().for_each(|(i,d)| *d = i);
        let h = IndexHashMap::with_data(&mut hdata[..rnelm],
                                               &mut hindex[..rnelm],
                                               &mut hnext[..rnelm],
                                               &mut hbucket[..rnelm],
                                               0);

        for (_,ptr,sp,subj,cof) in exprs.iter() {
            if let Some(sp) = sp {
                for (&i,sj,sc) in izip!(sp.iter(),
                                        subj.chunks_ptr(ptr),
                                        cof.chunks_ptr(ptr)) {
                    let n = sj.len();
                    if let Some(index) = h.at(i) {
                        let rp = { let p = &mut rptr[*index]; *p += n; *p - n };
                        rsubj[rp..rp+n].clone_from_slice(sj);
                        rcof[rp..rp+n].clone_from_slice(sc);
                    }
               }
            }
        }
        // Recompute ptr
        _ = rptr.iter_mut().fold(0,|v,p| { let tmp = *p; *p = v; tmp } );
    }
    rs.check();
    Ok(())
} // add

pub fn mul_matrix_expr_transpose(mshape : (usize,usize),
                                 msp   : Option<&[usize]>,
                                 mdata : &[f64],
                                 rs    : & mut WorkStack,
                                 ws    : & mut WorkStack,
                                 _xs    : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nd = shape.len();
    let nnz = *ptr.last().unwrap();
    let nelem = ptr.len()-1;
    if nd != 2 || shape[1] != mshape.1 {
        panic!("Mismatching operand shapes");
    }
    let shape = [shape[0],shape[1]];
    let rshape = [mshape.0,shape[0]];

    // compute size
    match (msp,sp) {
        (None,None) => {
            let (rnelm, rnnz) = (rshape[0]*rshape[1],nnz * mshape.0);

            let (rptr,_,rsubj,rcof) = rs.alloc_expr(&rshape, rnnz, rnelm);

            // build rptr
            rptr[0] = 0;
            ptr.iter().step_by(mshape.1).zip(ptr[mshape.1..].iter().step_by(mshape.1))
                .cycle()
                .zip(rptr[1..].iter_mut())
                .fold(0,|v,((p0,p1),rp)| { *rp = v+p1-p0; *rp });
    
            // build rsubj
            for jj in rsubj.chunks_mut(nnz) { jj.copy_from_slice(subj); }
            // build cof
            izip!(mdata.chunks(mshape.1).flat_map(|mrow| std::iter::repeat(mrow).take(shape[0])),
                  ptr.chunks(mshape.1).zip(ptr[1..].chunks(mshape.1)).cycle())
                .flat_map(|(mrow,(ptrbs,ptres))| izip!(mrow.iter(), ptrbs.iter(),ptres.iter()))
                .flat_map(|(mc,&p0,&p1)| izip!(std::iter::repeat(mc).take(p1-p0)))
                .zip(rcof.iter_mut().zip(cof.iter().cycle()))
                .for_each(|(mc,(rc,&c))| *rc = c*mc);
        },
        (Some(msp),None) => {
            if msp.is_empty() {
                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, 0,0);
                rptr[0] = 0;
            }
            else {
                assert!(msp.len() == mdata.len());

                // count nonzero data rows
                let num_nonempty_rows = msp.chunk_by(|a,b| *a / mshape.1 == *b / mshape.1).count();
                let rnelem = num_nonempty_rows * shape[0];

                // count nonzeros and elements
                let rnnz = msp.chunk_by(|a,b| *a / mshape.1 == *b / mshape.1)
                    //.scan(0,|p,msprow| { let (p0,p1) = (*p,*p+msprow.len()); Some((msprow, unsafe{ mdata.get_unchecked(p0..p1)})) })
                    .flat_map(|item| std::iter::repeat(item).take(shape[0]))
                    .zip(ptr.chunks(shape[1]).zip(ptr[1..].chunks(shape[1])).cycle())
                    .fold(0,|rnnz,(msprow,(ptrbs,ptres))| {
                        let elmnz = msprow.iter().map(|i| { let i = i % shape[1]; unsafe{ *ptres.get_unchecked(i) - *ptrbs.get_unchecked(i) } }).sum::<usize>();
                        rnnz + elmnz
                    });

                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, rnnz, rnelem);
                rptr[0] = 0;

                izip!(
                    msp.chunk_by(|a,b| *a / mshape.1 == *b / mshape.1)
                        .scan(0,|p,msprow| { let (p0,p1) = (*p,*p+msprow.len()); Some((msprow, unsafe{ mdata.get_unchecked(p0..p1)})) })
                        .flat_map(|item| std::iter::repeat(item).take(shape[0])),
                        ptr.chunks(shape[1]).zip(ptr[1..].chunks(shape[1])).cycle(),
                        rptr[1..].iter_mut())
                    .flat_map(|((msprow,mcofrow),(ptrbs,ptres),rp)| {
                        *rp = msprow.iter().map(|i| unsafe{ *ptres.get_unchecked(i % shape[1]) - *ptrbs.get_unchecked(i % shape[1]) }).sum();

                        msprow.iter().zip(mcofrow.iter())
                            .map(|(i,mc)| (mc, unsafe{*ptrbs.get_unchecked(i % shape[1])}, unsafe{*ptres.get_unchecked(i % shape[1])}))
                    })
                    .flat_map(|(mc,p0,p1)| izip!(std::iter::repeat(mc),unsafe{ subj.get_unchecked(p0..p1)}, unsafe{ cof.get_unchecked(p0..p1)}))
                    .zip(rsubj.iter_mut().zip(rcof.iter_mut()))
                    .for_each(|((mc,&j,&c),(rj,rc))| {
                        *rj = j;
                        *rc = c*mc;
                    });
                rptr.iter_mut().fold(0usize, |p,rp| { *rp += p; *rp });


                if let Some(rsp) = rsp {
                    msp.chunk_by(|a,b| *a / mshape.1 == *b / mshape.1)
                        .map(|row| unsafe{*row.get_unchecked(0)/mshape.1})
                        .flat_map(|i| std::iter::repeat(i).zip(0..shape[0]))
                        .zip(rsp.iter_mut())
                        .for_each(|((i,j),rsp)| {
                            *rsp = i*shape[0]+j;
                        }) ;
                }
            }
        },
        (None,Some(sp)) => {
            if sp.is_empty() {
                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, 0,0);
                rptr[0] = 0;
            }
            else {
                let nonempty_rows = sp.chunk_by(|&i0,&i1| i0 / shape[1] == i1 / shape[1]).count();
                let rnelem = mshape.0*nonempty_rows;
                let rnnz = mshape.0*nnz;
                
                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, rnnz, rnelem);
        
                // build rptr
                rptr[0] = 0;
                (0..mshape.0)
                    .flat_map(|_| {
                        sp.chunk_by(|i0,i1| i0/shape[1] == i1/shape[1])
                            .scan(0,|p, row| { 
                                let (p0,p1) = (*p,*p+row.len()); 
                                *p = p1;
                                Some(unsafe{ (ptr.get_unchecked(p0),ptr.get_unchecked(p1)) })
                            })})
                    .zip(rptr[1..].iter_mut())
                    .fold(0usize,|p,((p0,p1),rp)| {
                        *rp = p+(p1-p0); 
                        *rp 
                    });

                if let Some(rsp) = rsp {
                    (0..mshape.0)
                        .flat_map(|i| std::iter::repeat(i).zip(sp.chunk_by(|i0,i1| i0/shape[1] == i1/shape[1]).map(|row| (unsafe{*row.get_unchecked(0)/shape[1]},row))))
                        .map(|(i,(j,_))| i*shape[1]+j )
                        .zip(rsp.iter_mut())
                        .for_each(|(idx,rsp)| *rsp = idx)
                        ;
                }

                // build subj
                rsubj.chunks_mut(nnz).for_each(|jj| jj.copy_from_slice(subj));

                // build subj, cof
                mdata.chunks(mshape.1)
                    .flat_map(|mrow| std::iter::repeat(mrow).take(shape[0])) // repeat each matrix
                                                                             // row shape[0] times.
                    .zip(
                        (0..shape[0]).flat_map(|_| {
                            sp.chunk_by(|i0,i1| i0/shape[1] == i1/shape[1])
                                .scan(0,|p,esprow| {
                                    let (p0,p1) = (*p,*p+esprow.len());
                                    *p = p1;
                                    println!("ptr = {:?}, p0 = {}, p1 = {}",ptr,p0,p1);
                                    Some((esprow, unsafe{ ptr.get_unchecked(p0..=p1) }))
                                })}))
                    .flat_map(|(mrow, (esprow,eptr))| {
                        izip!(esprow.iter().map(|i| unsafe{ *mrow.get_unchecked(i % mshape.1) }),
                              eptr.iter(),
                              eptr[1..].iter())
                    })
                    .flat_map(|(mc,&p0,&p1)| izip!(std::iter::repeat(mc),unsafe{cof.get_unchecked(p0..p1)}.iter()))
                    .zip(rcof.iter_mut())
                    .for_each(|((mc,c),rc)| *rc = mc*c)
                    ;
            }
        },
        (Some(msp),Some(sp)) => {
            if msp.is_empty() || sp.is_empty() {
                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, 0,0);
                rptr[0] = 0;
            }
            else {
                let mut rnelem = 0usize;
                let mut rnnz = 0usize;
                // For each matrix row
                let (rnelem,rnnz) = 
                    izip!(
                        msp.chunk_by(|&a,&b| a/mshape.1 == b/mshape.1)
                            .scan(0,|p,row| { let (p0,p1) = (*p,*p+row.len()); *p = p1; Some((row,unsafe{mdata.get_unchecked(p0..p1)})) } )
                            .flat_map(|item| std::iter::repeat(item).take(shape[0])),
                        (0..mshape.0)
                            .flat_map(|_| sp.chunk_by(|&a,&b| a/shape[1] == b/shape[1])
                                            .scan(0,|p,row| { let (p0,p1) = (*p,*p+row.len()); *p = p1; Some((row,unsafe{ptr.get_unchecked(p0..p1+1)})) })))
                        .map(|((mrowi,mrowc),(erowi,erowptrs))| {
                            let i = unsafe{ *mrowi.get_unchecked(0) } / mshape.1;
                            let j = unsafe{ *erowi.get_unchecked(0) } / shape[1];
                            ( i * shape[0] + j,
                              mrowi.iter().inner_join_by(|&a,b| (a%mshape.1).cmp(&(b.0%shape[1])), izip!(erowi.iter(),erowptrs.iter(),erowptrs[1..].iter())).map(|(_,(_,epb,epe))| epe-epb ).sum::<usize>()) 
                        })
                        .fold((0,0),|(rnelm,rnnz),(rspi, nnz)| if nnz > 0 { (rnelm+1,rnnz+nnz) } else { (rnelm,rnnz) });

                println!("rnelem = {}, rnnz = {}",rnelem,rnnz);
                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, rnnz, rnelem);
                rptr[0] = 0;
                // compute subj/cof
                izip!(
                    msp.chunk_by(|&a,&b| a/mshape.1 == b/mshape.1)
                        .scan(0,|p,row| { let (p0,p1) = (*p,*p+row.len()); *p = p1; Some((row,unsafe{mdata.get_unchecked(p0..p1)})) } )
                        .flat_map(|item| std::iter::repeat(item).take(shape[0])),
                    (0..mshape.0)
                        .flat_map(|_| sp.chunk_by(|&a,&b| a/shape[1] == b/shape[1])
                                        .scan(0,|p,row| { let (p0,p1) = (*p,*p+row.len()); *p = p1; Some((row,unsafe{ptr.get_unchecked(p0..p1+1)})) })))
                    // for each conbination matrix row/expr row
                    .flat_map(|((mrowi,mrowc),(erowi,erowptrs))| {
                        mrowi.iter().zip(mrowc.iter()).inner_join_by(|a,b| (a.0%mshape.1).cmp(&(b.0%shape[1])), izip!(erowi.iter(),erowptrs.iter(),erowptrs[1..].iter()))
                            .flat_map(|((_,&mc),(_,&pb,&pe))| {
                                let mc : f64 = mc;
                                unsafe{subj.get_unchecked(pb..pe)}.iter().zip(unsafe{cof.get_unchecked(pb..pe)}.iter().map(move |&c| c * mc))
                            })
                    })
                    .zip(rsubj.iter_mut().zip(rcof.iter_mut()))
                    .for_each(|((&j,c),(rj,rc))| { *rj = j; *rc = c; });
                // compute ptr
                let mut it = 
                    izip!(
                        msp.chunk_by(|&a,&b| a/mshape.1 == b/mshape.1)
                            .scan(0,|p,row| { let (p0,p1) = (*p,*p+row.len()); *p = p1; Some((row,unsafe{mdata.get_unchecked(p0..p1)})) } )
                            .flat_map(|item| std::iter::repeat(item).take(shape[0])),
                        (0..mshape.0)
                            .flat_map(|_| sp.chunk_by(|&a,&b| a/shape[1] == b/shape[1])
                                            .scan(0,|p,row| { let (p0,p1) = (*p,*p+row.len()); *p = p1; Some((row,unsafe{ptr.get_unchecked(p0..p1+1)})) })))
                        // for each conbination matrix row/expr row
                        .map(|((mrowi,mrowc),(erowi,erowptrs))| {
                            let i = unsafe{ *mrowi.get_unchecked(0) } / mshape.1;
                            let j = unsafe{ *erowi.get_unchecked(0) } / shape[1];
                            ( i * shape[0] + j, 
                              mrowi.iter().zip(mrowc.iter()).inner_join_by(|a,b| (a.0%mshape.1).cmp(&(b.0%shape[1])), izip!(erowi.iter(),erowptrs.iter(),erowptrs[1..].iter())).map(|(a,b)| b.2-b.1).sum::<usize>() ) })
                        .filter(|(_,nnz)| *nnz > 0);
                // compute sp
                if let Some(rsp) = &rsp {
                    izip!(it,rptr[1..].iter_mut(),rsp.iter_mut())
                        .fold(0,|p,((spi,nnz),rptri,rspi)| { *rptri = p+nnz; *rspi = spi; p+nnz });
                }
                else {
                    izip!(it,rptr[1..].iter_mut())
                        .fold(0,|p,((_,nnz),rptri)| { *rptri = p+nnz; p+nnz });
                }
                println!("rptr  = {:?}",rptr);
                if let Some(rsp) { println!("rsp   = {:?}",rsp); }
                println!("rsubj = {:?}",rsubj);
                println!("rcof  = {:?}",rcof);
            }
        }
    };

    Ok(())
}

/// Evaluates `lhs` * expr.
pub fn mul_left_dense(mdata : &[f64],
                             mdimi : usize,
                             mdimj : usize,
                             rs    : & mut WorkStack,
                             ws    : & mut WorkStack,
                             _xs    : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let nd   = shape.len();
    let nnz  = subj.len();

    if nd != 2 && nd != 1{ return Err(ExprEvalError::new(file!(),line!(),"Invalid shape for multiplication")) }
    if mdimj != shape[0] { return Err(ExprEvalError::new(file!(),line!(),"Mismatching shapes for multiplication")) }

    let rdimi = mdimi;
    let rdimj = if nd == 1 { 1 } else { shape[1] };
    //let edimi = shape[0];
    let edimj = shape.get(1).copied().unwrap_or(1);
    let rnnz = nnz * mdimi;
    let rnelm = mdimi * rdimj;

    let (rptr,_rsp,rsubj,rcof) = if nd == 2 {
        rs.alloc_expr(&[rdimi,rdimj],rnnz,rnelm)
    }
    else {
        rs.alloc_expr(&[rdimi],rnnz,rnelm)
    };

    // sparse expr
    if let Some(sp) = sp {
        rptr.fill(0);
        for (i,p0,p1) in izip!(sp.iter(),ptr.iter(),ptr[1..].iter()) {
            let (_ii,jj) = (i/edimj, i%edimj);
            for n in rptr[jj..].iter_mut().step_by(edimj) { *n += p1-p0 }
        }

        // build ptr
        let _ = rptr.iter_mut().fold(0,|v,p| { let prev = *p; *p = v; v+prev });

        for (i,&p0,&p1) in izip!(sp.iter(),ptr.iter(),ptr[1..].iter()) {
            let (ii,jj) = (i/edimj, i%edimj);

            let rownnz = p1-p0;

            for (p,&v) in izip!(rptr[jj..].iter_mut().step_by(edimj),
                                mdata[ii..].iter().step_by(mdimj)) {
                rsubj[*p..*p+rownnz].clone_from_slice(&subj[p0..p1]);
                rcof[*p..*p+rownnz].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c * v);
                *p += rownnz;
            }
        }

        let _ = rptr.iter_mut().fold(0,|v,p| { let prev = *p; *p = v; prev });
    }
    // dense expr
    else {        
        rptr[0] = 0;
        let mut nzi = 0;
        for (mrow,rptrrow) in mdata.chunks(mdimj).zip(rptr[1..].chunks_mut(edimj)) {
            for (j,rp) in rptrrow.iter_mut().enumerate() {
                for (&v,&p0,&p1) in izip!(mrow.iter(),ptr[j..].iter().step_by(edimj),ptr[j+1..].iter().step_by(edimj)) {
                    rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c * v);
                    nzi += p1-p0;
                }
                *rp = nzi;
            }
        }
    }
    Ok(())
} // mul_left_dense

pub fn mul_right_dense(mdata : &[f64],
                              mdimi : usize,
                              mdimj : usize,
                              rs    : & mut WorkStack,
                              ws    : & mut WorkStack,
                              _xs    : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let nd   = shape.len();
    let nnz  = subj.len();
    //let nelm = ptr.len()-1;
    if nd != 2 && nd != 1{ return Err(ExprEvalError::new(file!(),line!(),"Invalid shape for multiplication")) }
    let (edimi,edimj) = if let Some(d2) = shape.get(1) {
        (shape[0],*d2)
    }
    else {
        (1,shape[0])
    };

    if mdimi != edimj { return Err(ExprEvalError::new(file!(),line!(),"Mismatching shapes for multiplication")) }

    let rdimi = edimi;
    let rdimj = mdimj;
    let rnnz = nnz * mdimj;
    let rnelm = rdimi * rdimj;

    let (rptr,_rsp,rsubj,rcof) = if nd == 2 {
        rs.alloc_expr(&[rdimi,rdimj],rnnz,rnelm)
    }
    else {
        rs.alloc_expr(&[rdimj],rnnz,rnelm)
    };

    if let Some(sp) = sp {
        rptr.fill(0);

        for (k,p0,p1) in izip!(sp.iter(),ptr.iter(),ptr[1..].iter()) {
            let (ii,_jj) = (k/edimj,k%edimj);
            rptr[ii*rdimj..(ii+1)*rdimj].iter_mut().for_each(|p| *p += p1-p0);
        }
        let _ = rptr.iter_mut().fold(0,|v,p| { let prev = *p; *p = v; v+prev });



        for (k,&p0,&p1) in izip!(sp.iter(),ptr.iter(),ptr[1..].iter()) {
            let (ii,jj) = (k/edimj,k%edimj);

            for (rp,v) in izip!(rptr[ii*rdimj..(ii+1)*rdimj].iter_mut(),
                                mdata[jj*mdimj..(jj+1)*mdimj].iter()) {
                rsubj[*rp..*rp+p1-p0].clone_from_slice(&subj[p0..p1]);
                rcof[*rp..*rp+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c * v);
                *rp += p1-p0;
            }
        }
        let _ = rptr.iter_mut().fold(0,|v,p| { let prev = *p; *p = v; prev });
    }
    // dense expr
    else {
        rptr[0] = 0;
        let mut nzi = 0;
        for (rp,((ptrb,ptre),i)) in rptr[1..].iter_mut().zip(iproduct!(ptr.chunks(edimj).zip(ptr[1..].chunks(edimj)), 0..mdimj)) {
            for (&p0,&p1,&v) in izip!(ptrb.iter(),ptre.iter(),mdata[i..].iter().step_by(mdimj)) {
                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                izip!(rcof[nzi..nzi+p1-p0].iter_mut(),cof[p0..p1].iter())
                    .for_each(|(rc,&c)| *rc = c * v);
                nzi += p1-p0;
            }
            *rp = nzi;
        }
    }
    Ok(())
} // mul_right_dense


pub fn mul_left_sparse(mheight : usize,
                              mwidth : usize,
                              msparsity : &[usize],
                              mdata : &[f64],
                              rs : & mut WorkStack,
                              ws : & mut WorkStack,
                              xs : & mut WorkStack) -> Result<(),ExprEvalError> {

    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nd = shape.len();
    if nd != 1 && nd != 2 { panic!("Expression is incorrect shape for multiplication: {:?}",shape); }
    if shape[0] != mwidth { return Err(ExprEvalError::new(file!(),line!(),"Mismatching operand shapes for multiplication")); }
    let _eheight = shape[0];
    let ewidth = shape.get(1).copied().unwrap_or(1);

    if let Some(sp) = sp {
        let (us,_) = xs.alloc(mheight // msubi
                              + mheight+1 // mrowptr
                              + sp.len() // eperm
                              + ewidth // esubj
                              + ewidth+1, // wcolptr
                              0);

        let (msubi,us)   = us.split_at_mut(mheight);
        let (mrowptr,us) = us.split_at_mut(mheight+1);
        let (perm,us)    = us.split_at_mut(sp.len());
        let (esubj,ecolptr) = us.split_at_mut(ewidth);

        // build row ptr structure for matrix data
        let nummrows = {
            let mut rowidx = 0; let mut rowi = usize::MAX;
            for (k,&ix) in msparsity.iter().enumerate() {
                let i = ix / mwidth;
                if i != rowi {
                    mrowptr[rowidx] = k;
                    msubi[rowidx] = i;
                    rowidx += 1;
                    rowi = i;
                }
            }
            mrowptr[rowidx] = msparsity.len();
            rowidx
        };
        let msubi   = & msubi[..nummrows];
        let mrowptr = & mrowptr[..nummrows+1];

        // build column-major ordering permutation and ptr for expression
        perm.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        perm.sort_by_key(|&i| unsafe{(*sp.get_unchecked(i) % ewidth,*sp.get_unchecked(i)/ewidth)});

        let numecols = {
            let mut colidx = 0; let mut coli = usize::MAX;
            for (k,&ix) in sp.permute_by(perm).enumerate() {
                let j = ix % ewidth;
                if j != coli {
                    ecolptr[colidx] = k;
                    esubj[colidx] = j;
                    colidx += 1;
                    coli = j;
                }
            }
            ecolptr[colidx] = sp.len();
            colidx
        };
        let esubj = & esubj[..numecols];
        let ecolptr = & ecolptr[..numecols+1];

        // count result elements and nonzeros
        let mut rnelm = 0;
        let mut rnnz  = 0;
        for (&mp0,&mp1) in izip!(mrowptr.iter(),mrowptr[1..].iter()) {
            for (&ep0,&ep1) in izip!(ecolptr.iter(),ecolptr[1..].iter()) {
                let mut espi = izip!(sp.permute_by(&perm[ep0..ep1]),
                                     ptr.permute_by(&perm[ep0..ep1]),
                                     ptr[1..].permute_by(&perm[ep0..ep1])).peekable();
                let mut mspi = msparsity[mp0..mp1].iter().peekable();

                let mut ijnnz = 0;
                while let (Some((&ei,&p0,&p1)),Some(&mi)) = (espi.peek(),mspi.peek()) {
                    match (mi % mwidth).cmp(&(ei / ewidth)) {
                        std::cmp::Ordering::Less    => { let _ = mspi.next(); },
                        std::cmp::Ordering::Greater => { let _ = espi.next(); },
                        std::cmp::Ordering::Equal => { 
                            let _ = espi.next();
                            let _ = mspi.next();
                            ijnnz += p1-p0; 
                        }
                    }
                }
                rnnz += ijnnz;
                if ijnnz > 0 { rnelm += 1; }
            }
        }

        // build result
        let (rptr,mut rsp,rsubj,rcof) = rs.alloc_expr(&[mheight,ewidth],rnnz,rnelm);
        let mut nzi = 0;
        let mut elmi = 0;
        rptr[0] = 0;
        for (&i,&mp0,&mp1) in izip!(msubi.iter(),mrowptr.iter(),mrowptr[1..].iter()) {
            for (&j,&ep0,&ep1) in izip!(esubj.iter(),ecolptr.iter(),ecolptr[1..].iter()) {
                let mut espi = izip!(sp.permute_by(&perm[ep0..ep1]),
                                     ptr.permute_by(&perm[ep0..ep1]),
                                     ptr[1..].permute_by(&perm[ep0..ep1])).peekable();
                let mut mspi = izip!(msparsity[mp0..mp1].iter(),
                                     mdata[mp0..mp1].iter()).peekable();

                let mut ijnnz = nzi;
                while let (Some((&ei,&p0,&p1)),Some((&mi,&mc))) = (espi.peek(),mspi.peek()) {
                    let km = mi % mwidth;
                    let ke = ei / ewidth;

                    match km.cmp(&ke) {
                        std::cmp::Ordering::Less    => { let _ = mspi.next(); },
                        std::cmp::Ordering::Greater => { let _ = espi.next(); },
                        std::cmp::Ordering::Equal => {
                            rsubj[ijnnz..ijnnz+p1-p0].clone_from_slice(&subj[p0..p1]);
                            rcof[ijnnz..ijnnz+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c*mc );
                            let _ = espi.next();
                            let _ = mspi.next();
                            ijnnz += p1-p0;
                        },
                    }
                }
                if ijnnz > nzi {
                    nzi = ijnnz;
                    if let Some(ref mut rsp) = rsp { rsp[elmi] = i * ewidth + j; }
                    rptr[elmi+1] = nzi;
                    elmi += 1;
                }
            }
        }
    }
    else {
        let rnelm = mheight * ewidth;
        let (xptr,_) = xs.alloc(rnelm+1,0);
        xptr.fill(0);

        for &mspi in msparsity.iter() {
            let (mi,mj) = (mspi / mwidth,mspi % mwidth);

            for (j,&p0,&p1) in izip!(0..ewidth,
                                     ptr[mj*ewidth..(mj+1)*ewidth].iter(),
                                     ptr[mj*ewidth+1..(mj+1)*ewidth+1].iter()) {
                unsafe{ *xptr.get_unchecked_mut(mi*ewidth+j+1) += p1-p0 };
            }
        }
        let _ = xptr.iter_mut().fold(0,|v,p| { *p += v; *p });

        let &rnnz = xptr.last().unwrap();
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[mheight,ewidth],rnnz,rnelm);
        rptr.clone_from_slice(xptr);

        for (&mspi,&mv) in msparsity.iter().zip(mdata.iter()) {
            let (mi,mj) = (mspi / mwidth,mspi % mwidth);

            for (j,&p0,&p1) in izip!(0..ewidth,
                                     ptr[mj*ewidth..(mj+1)*ewidth].iter(),
                                     ptr[mj*ewidth+1..(mj+1)*ewidth+1].iter()) {
                let dst = unsafe{*xptr.get_unchecked(mi*ewidth+j)};
                rsubj[dst..dst+p1-p0].clone_from_slice(&subj[p0..p1]);
                rcof[dst..dst+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c*mv);

                unsafe{*xptr.get_unchecked_mut(mi*ewidth+j) += p1-p0};
            }
        }
    }
    Ok(())
}

// expr x matrix
pub fn mul_right_sparse(mheight : usize,
                               mwidth : usize,
                               msparsity : &[usize],
                               mdata : &[f64],
                               rs : & mut WorkStack,
                               ws : & mut WorkStack,
                               xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    if shape.len() != 1 && shape.len() != 2 {
        return Err(ExprEvalError::new(file!(),line!(),"Invalid operand shapes: Expr is not 1- or 2-dimensional"));
    }

    let (eheight,ewidth) =
        if let Some(&d) = shape.get(1) { (shape[0],d) }
        else { (1,shape[0]) };

    if ewidth != mheight {
        panic!("Incompatible operand shapes: {:?} * {:?}",shape,&[mheight,mwidth]);
    }
    //println!("Operand shapes: {:?} * {:?}",shape,&[mheight,mwidth]);

    // tranpose matrix
    let (us,mcof) = xs.alloc(eheight+1 // erowptr
                             +(mwidth+1)*2 // mptr
                             + mwidth // msubj
                             + msparsity.len(), // msubi
                             msparsity.len());
    let (erowptr,us)  = us.split_at_mut(eheight+1);
    let (mcolptr1,us) = us.split_at_mut(mwidth+1);
    let (mcolptr,us)  = us.split_at_mut(mwidth+1);
    let (msubj,msubi) = us.split_at_mut(mwidth);

    mcolptr1.fill(0);
    for &i in msparsity.iter() { unsafe{ *mcolptr1.get_unchecked_mut(i % mwidth + 1) += 1; } }
    let _ = mcolptr1.iter_mut().fold(0,|v,p| { *p += v; *p });
    for (&i,&c) in msparsity.iter().zip(mdata.iter()) {
        let j = i % mwidth;
        let p = unsafe{ *mcolptr1.get_unchecked(j) };
        unsafe{ *msubi.get_unchecked_mut(p) = i / mwidth; }
        unsafe{ *mcof.get_unchecked_mut(p) = c; }
        unsafe{ *mcolptr1.get_unchecked_mut(j) += 1; }
    }
    let _ = mcolptr1.iter_mut().fold(0,|v,p| { let prev = *p; *p -= v; prev });
    mcolptr[0] = 0;
    let _ = izip!(mcolptr1.iter().enumerate().filter(|(_,&n)| n > 0),
                  msubj.iter_mut(),
                  mcolptr[1..].iter_mut()).fold(0,|v,((j,&n),rj,p)| { *p = v+n; *rj = j; v+n });
    let mnumnzcol = mcolptr1[..mwidth].iter().filter(|&&n| n > 0).count();
    let mcolptr = &mcolptr[..mnumnzcol+1];
    let msubj   = &msubj[..mnumnzcol];

    if let Some(sp) = sp {
        let mut rnelm    = 0;
        let mut rnnz     = 0;

        erowptr.fill(0);
        sp.iter().for_each(|&i| unsafe{*erowptr.get_unchecked_mut(i / ewidth+1) += 1});
        let _ = erowptr.iter_mut().fold(0,|v,p| { *p += v; *p  });
        // count nonzeros and elements
        iproduct!(izip!(erowptr.iter(), erowptr[1..].iter())
                      .filter_map(|(&rp0,&rp1)| if rp0 < rp1 { Some((&sp[rp0..rp1],&ptr[rp0..rp1],&ptr[rp0+1..rp1+1])) } else { None }),
                  izip!(mcolptr.iter(), mcolptr[1..].iter())
                      .map(|(&mp0,&mp1)| &msubi[mp0..mp1]))
            .for_each(|((espis,ep0s,ep1s),mis)|{
                let mut ei = izip!(espis.iter().map(|&v| v % ewidth),ep0s.iter(),ep1s.iter()).peekable();
                let mut mi = mis.iter().peekable();

                let mut elmnnz = 0;
                while let (Some((ke,&p0,&p1)),Some(km)) = (ei.peek(),mi.peek()) {
                    match ke.cmp(km) {
                        std::cmp::Ordering::Less => { let _ = ei.next(); },
                        std::cmp::Ordering::Greater => { let _ = mi.next(); },
                        std::cmp::Ordering::Equal => {
                            let _ = ei.next();
                            elmnnz += p1-p0; 
                            let _ = mi.next();
                        },
                    }
                }
                if elmnnz > 0 {
                    rnnz += elmnnz  ;
                    rnelm += 1;
                }

            });
        // alloc result
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&[eheight,mwidth],rnnz,rnelm);

        // build result
        rptr[0] = 0;
        let mut nzi = 0;
        let ii =
            // 1. build iterator over the outer product of nonzero rows in expr and nonzero columns
            //    in matrix.
            iproduct!(izip!(0..eheight,erowptr.iter(), erowptr[1..].iter())
                          .filter_map(|(ri,&rp0,&rp1)| if rp0 < rp1 { Some((ri,&sp[rp0..rp1],&ptr[rp0..rp1],&ptr[rp0+1..rp1+1])) } else { None }),
                      izip!(msubj.iter(),mcolptr.iter(), mcolptr[1..].iter())
                          .map(|(&rj,&mp0,&mp1)| (rj, &msubi[mp0..mp1],&mcof[mp0..mp1])))
            // 2. Compute the inner product of row and column, filtering out the empty results
                .filter_map(|((ri,espis,ep0s,ep1s),(rj,mcolsubi,mcolcof))|{
                    let mut ei = izip!(espis.iter().map(|&v| v % ewidth),ep0s.iter(),ep1s.iter()).peekable();
                    let mut mi = izip!(mcolsubi.iter(),mcolcof.iter()).peekable();


                    let nzi0 = nzi;
                    while let (Some((ke,&p0,&p1)),Some((&km,&mv))) = (ei.peek(),mi.peek()) {
                        match ke.cmp(&km) {
                            std::cmp::Ordering::Less    => { let _ = ei.next(); },
                            std::cmp::Ordering::Greater => { let _ = mi.next(); },
                            std::cmp::Ordering::Equal   => {
                                let _ = ei.next();
                                let _ = mi.next();

                                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                                rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c*mv);

                                nzi += p1-p0;
                            },
                        }
                    }
                    if nzi0 < nzi {
                        Some((nzi,ri*mwidth+rj))
                    }
                    else {
                        None
                    }

                })
            // 3. zip the resulting nonzero elements with rptr and compute the ptr array, passing
            //    the sparsity index in in the iterator
                .zip(rptr[1..].iter_mut())
                .map(|((nnz,rk),rp)| { *rp = nnz; rk} );

        // 4. Finally compute the sparsity pattern, or, if not sparse, consume the iterator to
        //    effectuate computation of ptr, subj, and cof
        if let Some(rsp) = rsp {
            rsp.iter_mut().zip(ii).for_each(|(spi,k)| *spi = k);
        }
        else {
            ii.for_each(| _ |{});
        }
    }
    else {
        // count nonzeros
        let rnelm = mnumnzcol * eheight;
        let mut rnnz = 0;
        for (p0s,p1s) in izip!(ptr.chunks(ewidth),ptr[1..].chunks(ewidth)) {
            for (&mp0,&mp1) in izip!(mcolptr.iter(),mcolptr[1..].iter()) {

                let mcolsubi = &msubi[mp0..mp1];
                rnnz += izip!(p0s.permute_by(mcolsubi),
                              p1s.permute_by(mcolsubi)).map(|(&p0,&p1)| p1-p0).sum::<usize>();
            }
        }

        // alloc result
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&[eheight,mwidth],rnnz,rnelm);
        let mut nzi = 0;

        rptr[0] = 0;
        izip!(ptr.chunks(ewidth),ptr[1..].chunks(ewidth))
            .map(|(p0s,p1s)| izip!(std::iter::repeat(p0s),
                                   std::iter::repeat(p1s),
                                   mcolptr.iter(),
                                   mcolptr[1..].iter()))
            .flatten()
            .zip(rptr[1..].iter_mut())
            .for_each(|((p0s,p1s,&mp0,&mp1),rp)| {
                let mcolsubi = &msubi[mp0..mp1];
                izip!(p0s.permute_by(mcolsubi),
                      p1s.permute_by(mcolsubi),
                      mcof[mp0..mp1].iter()).for_each(|(&p0,&p1,&mv)| {
                          rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                          rcof[nzi..nzi+p1-p0].iter_mut().zip(&cof[p0..p1]).for_each(|(rc,&c)| *rc = c * mv);
                          nzi += p1-p0;
                      });
                *rp = nzi;
            });
        if let Some(rsp) = rsp {
            izip!(rsp.iter_mut(), iproduct!(0..eheight,msubj.iter()))
                .for_each(|(k,(i,&j))| { *k = i*mwidth+j; })
        }
    }
    Ok(())
}

pub fn dot_sparse(_sparsity : &[usize],
                  _data     : &[f64],
                  _rs       : & mut WorkStack,
                  ws       : & mut WorkStack,
                  _xs      : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (_shape,_ptr,_sp,_subj,_cof) = ws.pop_expr();
   
    return Err(ExprEvalError::new(file!(),line!(),"TODO"));
}

pub fn dot_vec(data : &[f64],
               rs   : & mut WorkStack,
               ws   : & mut WorkStack,
               _xs  : & mut WorkStack) -> Result<(),ExprEvalError> {

    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nd   = shape.len();
    let nnz  = subj.len();

    if nd != 1 || shape[0] != data.len() {
        return Err(ExprEvalError::new(file!(),line!(),"Mismatching operands"));
    }

    if let Some(sp) = sp {
        let rnnz = nnz;
        let rnelm = 1;
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[],rnnz,rnelm);

        rsubj.clone_from_slice(subj);
        rptr[0] = 0;
        rptr[1] = rnnz;
        for (&i,&p0,&p1) in izip!(sp.iter(),ptr[0..ptr.len()-1].iter(),ptr[1..].iter()) {
            let v = data[i];
            for (&c,rc) in cof[p0..p1].iter().zip(rcof.iter_mut()) {
                *rc = c*v;
            }
        }
    }
    else {
        let rnnz = nnz;
        let rnelm = 1;
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[],rnnz,rnelm);

        rsubj.clone_from_slice(subj);
        rptr[0] = 0;
        rptr[1] = rnnz;
        for (&p0,&p1,v) in izip!(ptr[0..ptr.len()-1].iter(),ptr[1..].iter(),data.iter()) {
            for (&c,rc) in cof[p0..p1].iter().zip(rcof[p0..p1].iter_mut()) {
                *rc = c*v;
            }
        }
    }
    rs.check();
    Ok(())
} // dot_vec

pub fn stack(dim : usize, n : usize, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    // println!("{}:{}: eval::stack n={}, dim={}",file!(),line!(),n,dim);
    let exprs = ws.pop_exprs(n);

    // check shapes
    //println!("vec = {:?}",exprs.iter().map(|v| v.0).collect::<Vec<&[usize]>>());
    if ! exprs.iter().zip(exprs[1..].iter()).any(|((s0,_,_,_,_),(s1,_,_,_,_))| s0.iter().zip(s1.iter()).enumerate().all(|(i,(&d0,&d1))| i == dim || d0 == d1)) {
        return Err(ExprEvalError::new(file!(),line!(),"Mismatching shapes or stacking dimension"));
    }

    let nd = (dim+1).max(exprs.iter().map(|(shape,_,_,_,_)| shape.len()).max().unwrap());
    let (rnnz,rnelm,ddim) = exprs.iter().fold((0,0,0),|(nnz,nelm,d),(shape,ptr,_sp,_subj,_cof)| (nnz+ptr.last().unwrap(),nelm+ptr.len()-1,d+shape.get(dim).copied().unwrap_or(1)));
    let mut rshape = exprs.first().unwrap().0.to_vec();
    rshape[dim] = ddim;

    let (rptr,mut rsp,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(),rnnz,rnelm);

    // Stacking in any number of dimensions can always be reduced to
    // stacking in 3 dimensions, with the dimension sizes computed as:
    let d0 : usize  = rshape[..dim].iter().product();
    let d1 = rshape[dim];
    let d2 : usize = if dim+1 < nd { rshape[dim+1..].iter().product() } else { 1 };

    // Special case: Stacking in dimension 0 means we can basically
    // concatenate the expressions. We test d0==1 meaning that the
    // product up to the stacking dimension are all 1, so effectively
    // that means stacking in the first (non-one) dimension.
    if d0 == 1 {
        // println!("{}:{}: eval::stack CASE 1",file!(),line!());
        let mut elmi : usize = 0;
        let mut nzi  : usize = 0;
        let mut ofs  : usize = 0;

        rptr[0] = 0;
        for (shape,ptr,sp,subj,cof) in exprs.iter() {
            // println!("{}:{}: shape = {:?}",file!(),line!(),shape);
            let nnz = ptr.last().unwrap();
            let nelm = ptr.len()-1;
            rsubj[nzi..nzi+nnz].clone_from_slice(subj);
            rcof[nzi..nzi+nnz].clone_from_slice(cof);
            izip!(rptr[elmi+1..elmi+nelm+1].iter_mut(),
                  ptr.iter(),
                  ptr[1..].iter()).for_each(|(rp,&p0,&p1)| *rp = p1-p0);

            if let Some(ref mut rsp) = rsp {
                if let Some(sp) = sp {
                    rsp[elmi..elmi+nelm].iter_mut().zip(sp.iter()).for_each(|(rsp,&sp)| *rsp = sp+ofs);
                }
                else {
                    rsp[elmi..elmi+nelm].iter_mut().zip(0..nelm).for_each(|(rsp,sp)| *rsp = sp+ofs);
                }
                ofs += shape.iter().product::<usize>();
            }
            nzi += ptr.last().unwrap();

            elmi += ptr.len()-1;
        }
        //println!("{}:{}: rptr = {:?}",file!(),line!(),rptr);
        let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
    }
    // Case 2: The result is sparse, implying that at least one
    // operand is sparse. Strategy:
    // 1. Create intermediate xsp and xptr
    // 2. Loop through each expression and add all sparsity index and
    //    number of nonzeros for each element to xsp and xptr
    //    respectively.
    // 3. Compute the permutation, xperm, that sort the result
    //    sparsity, xsp, and cummulate xptr ordered by xperm. Now
    //    xptr[i] points to the location in rsubj and rcof where the
    //    i'th element starts.
    // 4. Loop though each expression and copy nonzeros, then copy ptr and sp.
    else if let Some(rsp) = rsp {
        let (us,_fs) = xs.alloc(3*rnelm,0);
        let (xptr,us) = us.split_at_mut(rnelm);
        let (xsp,xperm) = us.split_at_mut(rnelm);
        // build ptr
        let mut elmi = 0;
        let mut ofs = 0;
        for (shape,ptr,sp,_,_) in exprs.iter() {
            let vd1 = shape.get(dim).copied().unwrap_or(1);
            if let Some(sp) = sp {
                for (xi,xp,&i,&p0,&p1) in izip!(xsp[elmi..elmi+sp.len()].iter_mut(),
                                                xptr[elmi..elmi+sp.len()].iter_mut(),
                                                sp.iter(),
                                                ptr.iter(),
                                                ptr[1..].iter()) {
                    let (i0,i1,i2) = (i/(vd1*d2),(i/d2)%vd1,i%d2);
                    *xp = p1-p0;
                    *xi = (i0*d1+i1+ofs)*d2+i2;
                }
            }
            else {
                for (xi,xp,i,&p0,&p1) in izip!(xsp[elmi..elmi+ptr.len()-1].iter_mut(),
                                               xptr[elmi..elmi+ptr.len()-1].iter_mut(),
                                               iproduct!(0..d0,ofs..ofs+vd1,0..d2).map(|(i0,i1,i2)| (i0*d1+i1)*d2+i2),
                                               ptr.iter(),
                                               ptr[1..].iter()) {
                    *xp = p1-p0;
                    *xi = i;
                }
            }
            elmi += *ptr.last().unwrap();
            ofs += vd1;
        }
        xperm.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        xperm.sort_by_key(|&p| unsafe{xsp.get_unchecked(p)});
        let _ = xperm.iter().fold(0,|v,p| { let prev = unsafe{*xptr.get_unchecked(*p)}; unsafe{*xptr.get_unchecked_mut(*p) = v}; prev+v});

        // build subj/cof/ptr/sp
        let mut elmi = 0;
        for (_shape,ptr,_sp,subj,cof) in exprs.iter() {
            let nelm = ptr.len()-1;
            for (xp,&p0,&p1) in izip!(xptr[elmi..elmi+nelm].iter_mut(),
                                      ptr.iter(),
                                      ptr[1..].iter()) {
                let n = p1-p0;
                rsubj[*xp..*xp+n].clone_from_slice(&subj[p0..p1]);
                rcof[*xp..*xp+n].clone_from_slice(&cof[p0..p1]);
                *xp += n;
            }
            elmi += *ptr.last().unwrap();
        }

        rptr[0] = 0;
        izip!(xperm.iter(),rptr[1..].iter_mut(),rsp.iter_mut())
            .for_each(|(&p,rp,ri)| {
                *ri = unsafe{*xsp.get_unchecked(p)};
                *rp = unsafe{*xptr.get_unchecked(p)};
            });
    }
    // Case 3: All operands and the result is dense.
    else {
        let rblocksize = d1*d2;
        let mut ofs  : usize = 0;
        // Build the result ptr
        // println!("{}:{}: Stack: Dense, rshape = {:?}, stack dim = {}",file!(),line!(),rshape,dim);
        rptr.iter_mut().for_each(|v| *v = 0);
        for (shape,ptr,_,_,_) in exprs.iter() {
            let vd1 = shape.get(dim).copied().unwrap_or(1);
            let blocksize = vd1*d2;
            // println!("{}:{}: blocksize = {}",file!(),line!(),blocksize);
            for (rps,p0s,p1s) in izip!(rptr[1..].chunks_mut(rblocksize),
                                       ptr.chunks(blocksize),
                                       ptr[1..].chunks(blocksize)) {
                rps[ofs..].iter_mut()
                    .zip(p0s.iter().zip(p1s.iter()).map(|(&p0,&p1)| p1-p0))
                    .for_each(|(rp,n)| *rp = n);
            }

            ofs += vd1;
        }
        let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
        // Then copy nonzeros
        let mut ofs : usize = 0;
        rsubj.fill(0);
        for (shape,ptr,_,subj,cof) in exprs.iter() {
            let vd1 = shape.get(dim).copied().unwrap_or(1);
            let blocksize = vd1*d2;
            izip!(rptr.chunks(rblocksize),
                  ptr.chunks(blocksize),
                  ptr[1..].chunks(blocksize))
                .for_each(|(rps,p0s,p1s)| {
                    let p0 = *p0s.first().unwrap();
                    let p1 = *p1s.last().unwrap();

                    let rp = rps[ofs];
                    rsubj[rp..rp+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[rp..rp+p1-p0].clone_from_slice(&cof[p0..p1]);

                });

            ofs += vd1;
        }
    }
    rs.check();
    Ok(())
}

pub fn sum_last(num : usize, rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let d = shape[shape.len()-num..].iter().product();
    let mut rshape = shape.to_vec();
    rshape[shape.len()-num..].iter_mut().for_each(|s| *s = 1);

    if let Some(sp) = sp {
        let rnelm =
            if sp.len() == 0 {
                0
            }
            else {
                sp.iter().zip(sp[1..].iter()).filter(|(&i0,&i1)| i0/d < i1/d).count()+1
            };
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(), subj.len(), rnelm);
        
        rptr[0] = 0;
        if let Some(rsp) = rsp {
            for (rp,r,(p,v)) in izip!(rptr[1..].iter_mut(),
                                      rsp.iter_mut(),
                                      izip!(ptr[1..].iter(),
                                            sp.iter(),
                                            sp[1..].iter().chain(once(&usize::MAX)))
                                        .filter(|(_,&i0,&i1)| i0/d < i1/d)
                                        .map(|(&p,&i0,_)| (p,i0/d))) {
                *r = v;
                *rp = p
            }
        } else {
            for (rp,(p,_v)) in izip!(rptr[1..].iter_mut(),
                                    izip!(ptr[1..].iter(),
                                          sp.iter(),
                                          sp[1..].iter().chain(once(&usize::MAX)))
                                      .filter(|(_,&i0,&i1)| i0/d < i1/d)
                                      .map(|(&p,&i0,_)| (p,i0/d))) {
                *rp = p
            }
            
        }
        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);
    } 
    else {
        let rnelm = shape.iter().product::<usize>()/d; 
        let (rptr,_,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(),subj.len(),rnelm);

        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);
        rptr.iter_mut().zip(ptr.iter().step_by(d)).for_each(|(rp,&p)| *rp = p );
    }
    rs.check();
    Ok(())
}

pub fn mul_elem(datashape : &[usize],
                        datasparsity : Option<&[usize]>,
                        data : &[f64],
                        rs : & mut WorkStack,
                        ws : & mut WorkStack,
                        xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let &nnz = ptr.last().unwrap();
    let nelm = ptr.len()-1;

    if shape.iter().zip(datashape.iter()).any(|(&s0,&s1)| s0 != s1) { return Err(ExprEvalError::new(file!(),line!(),"Mismatching operand shapes in mul_elm")); }

    match (datasparsity,sp) {
        (Some(msp),Some(esp)) => {
            let (upart,xcof) = xs.alloc(esp.len()*2+1+subj.len(),subj.len());
            let (xsp,upart) = upart.split_at_mut(esp.len());
            let (xptr,xsubj) = upart.split_at_mut(esp.len()+1);

            xptr[0] = 0;
            let mut mit   = msp.iter().zip(data.iter()).peekable();
            let mut eit   = izip!(esp.iter(),ptr.iter(),ptr[1..].iter()).peekable();
            let mut rnelm = 0usize;
            let mut rnnz  = 0usize;

            while let (Some((&mi,&mc)),Some((ei,&p0,&p1))) = (mit.peek(),eit.peek()) {
                match mi.cmp(ei) {
                    std::cmp::Ordering::Less => _ = mit.next(),
                    std::cmp::Ordering::Greater => _ = eit.next(),
                    std::cmp::Ordering::Equal => {
                        xsp[rnelm] = mi;                            
                        xsubj[rnnz..rnnz+p1-p0].clone_from_slice(&subj[p0..p1]);                            
                        xcof[rnnz..rnnz+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(tc,&sc)| *tc = sc * mc);

                        rnelm += 1;
                        rnnz += p1-p0;
                        xptr[rnelm] = rnnz;

                        _ = mit.next();
                        _ = eit.next();
                    }
                }
            }

            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape, rnnz, rnelm);
            rptr.clone_from_slice(&xptr[..rnelm+1]);
            assert!(rptr[0] == 0);
            if let Some(rsp) = rsp { rsp.clone_from_slice(&xsp[..rnelm]) };
            rsubj.clone_from_slice(&xsubj[..rnnz]);
            rcof.clone_from_slice(&xcof[..rnnz]);
            xs.clear();
        }
        (Some(msp),None) =>  {
            // count result size
            let rnelm = msp.len();
            let rnnz = msp.iter().map(|&i| ptr[i+1]-ptr[i]).sum();
            //println!("ExprMulElm::eval(): nelm = {}, nnz = {}, rnelm = {}, rnnz = {}",nelm,nnz, rnelm,rnnz);
            //println!("ExprMulElm::eval():\nSource\n\tptr = {:?}\n\tsubj = {:?}",ptr,subj);
            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape, rnnz, rnelm);
            rptr[0] = 0;
            let mut nzi = 0usize;

            let rsp = rsp.unwrap();
            for (ri,rp,&i,&mc) in izip!(rsp.iter_mut(),
                                        rptr[1..].iter_mut(),
                                        msp.iter(),
                                        data.iter()) {
                let p0 = ptr[i];
                let p1 = ptr[i+1];

                //println!("  p0 = {}, p1 = {}",p0,p1);
                *ri = i;
                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c * mc);
                nzi += p1-p0;
                *rp = nzi; 
            }
            //println!("ExprMulElm::eval(): rptr = {:?}",rptr);
        }
        (None,Some(esp)) => {
            let rnnz = nnz;
            let rnelm = nelm;

            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape, rnnz, rnelm);

            if let Some(rsp) = rsp {
                rsp.clone_from_slice(esp);
            }
            rsubj.clone_from_slice(subj);
            rptr.clone_from_slice(ptr);
            rcof.clone_from_slice(cof);
            for (&p0,&p1,&i) in izip!(ptr.iter(),ptr[1..].iter(),esp.iter()) {
                let mc = data[i];
                rcof[p0..p1].iter_mut().for_each(|c| *c *= mc);
            }
        }
        (None,None) => {
            let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(shape, nnz, nelm);
            rptr.clone_from_slice(ptr);
            rsubj.clone_from_slice(subj);
            rcof.clone_from_slice(cof);
            for (&p0,&p1,&c) in izip!(ptr.iter(),ptr[1..].iter(),data.iter()) {
                rcof[p0..p1].iter_mut().for_each(|t| *t *= c );
            }
        }
    } // match (sparsity,sp)
    rs.check();
    Ok(())
}

pub fn scalar_expr_mul
(   datashape : &[usize],
    datasparsity : Option<&[usize]>,
    data : &[f64],
    rs : & mut WorkStack, 
    ws : & mut WorkStack, 
    _xs : & mut WorkStack) -> Result<(),ExprEvalError> 
{
    use std::iter::repeat;
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    assert_eq!(shape.len(), 0);
    if let Some(_sp) = sp {
        let (rptr,_rsp,_rsubj,_rcof) = rs.alloc_expr(&datashape, 0, 0);
        rptr[0] = 0;
    }
    else {
        let nnz = *ptr.last().unwrap();
        let rnelem = data.len();
        let rnnz = nnz * rnelem;
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&datashape, rnnz, rnelem);
        
        rptr.iter_mut().fold(0,|c,r| { *r = c; c + nnz });
        if let (Some(rsp),Some(dsp)) = (rsp,datasparsity) {
            rsp.copy_from_slice(dsp);
        }
        rsubj.iter_mut().zip(subj[0..nnz].iter().cycle()).for_each(| (r,&s)| *r = s);
        izip!(rcof.iter_mut(),
              data.iter().flat_map(|&v| repeat(v).take(nnz)),  
              cof[0..nnz].iter().cycle())
            .for_each(| (r,s0,&s1)| *r = s0 * s1);
    }
    rs.check();
    Ok(())
}



pub fn into_symmetric(dim : usize, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    use std::iter::repeat;
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    // check
    if dim + 2 > shape.len() { return Err(ExprEvalError::new(file!(),line!(),"Invalid dimension for symmetrization")); }
    
    let n : usize = {
        let d = shape[dim]*shape[dim+1];
        // n*(n+1) = 2d 
        // n^2 + n - 2d = 0 
        // n = (-1 + sqrt(1 + 8d)/2 

        let n = (((1.0+8.0*d as f64).sqrt()-1.0)/2.0).floor() as usize;
        if n*(n+1)/2 != d {
            return Err(ExprEvalError::new(file!(),line!(),"Specified symmetrization dimensions do not match a symmetric size"));
        }
        n
    };

    let mut strides = vec![0usize; shape.len()]; strides.iter_mut().zip(shape.iter()).rev().fold(1,|st,(s,&d)| { *s = st; st * d });
    let mut rshape = shape.to_vec();
    rshape[dim] = n;
    rshape[dim+1] = n;
    
    let d0 = shape[..dim].iter().product();
    let d1 = shape[dim]*shape[dim+1];
    let d2 = shape[dim+2..].iter().product();

    if let Some(sp) = sp {
        // count rnnz,rnelm
        let nelm = ptr.len()-1;
        let mut rnnz = 0;
        let mut rnelm = 0;
        let (urest,_) = xs.alloc(6*nelm,0);
        let (xsp,urest) = urest.split_at_mut(2*nelm);
        let (xperm,urest) = urest.split_at_mut(2*nelm);
        let (xsrc,_) = urest.split_at_mut(2*nelm);

        izip!(sp.iter(),ptr.iter(),ptr[1..].iter()).enumerate().for_each(|(idx,(&i,pb,pe))| {
            let i0 = i / (d1*d2);
            let k  = (i / d2) % d1;
            let i2 = i % d2;
            let ii  = ((((1+8*k) as f64).sqrt()-1.0)/2.0).floor() as usize;
            let jj  = k - ii*(ii+1)/2;
            if ii == jj {
                //xsp[rnelm] = i0*d1*d2 + (ii*(ii+1)/2+jj)*d2 + i2;
                xsp[rnelm] = i0*d1*d2 + ii * n * d2 + jj * d2 + i2;
                xsrc[rnelm]  = idx;

                rnelm += 1;
                rnnz  += pe-pb;
            }
            else {
                //xsp[rnelm]   = i0*d1*d2 + (ii*(ii+1)/2+jj)*d2 + i2;
                //xsp[rnelm+1] = i0*d1*d2 + (jj*(jj+1)/2+ii)*d2 + i2;
                xsp[rnelm]   = i0*d1*d2 + ii * n * d2 + jj * d2 + i2;
                xsp[rnelm+1] = i0*d1*d2 + jj * n * d2 + ii * d2 + i2;
                xsrc[rnelm]    = idx;
                xsrc[rnelm+1]  = idx;

                rnelm += 2;
                rnnz  += 2*(pe-pb);
            }
        });
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rshape.as_ref(), rnnz, rnelm);

        xperm.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        xperm[..rnelm].sort_by_key(|&i| xsp[i]);

        if let Some(rsp) = rsp {
            xsp.permute_by(xperm).zip(rsp.iter_mut()).for_each(|(&i,spi)| *spi = i);            
        }
        let xperm2 = xsp; 
        xsrc.permute_by(xperm).zip(xperm2.iter_mut()).for_each(|(&si,i)| *i = si);

        rptr[0] = 0;
        for (&pb,&pe,p) in izip!(ptr.permute_by(xperm2),
                                 ptr[1..].permute_by(xperm2),
                                 rptr[1..].iter_mut()) {
            *p = pe-pb;
        }
        rptr.iter_mut().fold(0,|c,p| { *p += c; *p });
        for (&pb,&pe,&rpb,&rpe) in izip!(ptr.permute_by(xperm2),
                                         ptr[1..].permute_by(xperm2),
                                         rptr.iter(),
                                         rptr[1..].iter()) {
            rsubj[rpb..rpe].copy_from_slice(&subj[pb..pe]);
            rcof[rpb..rpe].copy_from_slice(&cof[pb..pe]);
        }
    }
    else {
        let rnelm : usize = rshape.iter().product();
        // count nnz
        let rnnz : usize = 
            izip!(iproduct!(0..d0,0..n).flat_map(|(i0,i1a)| izip!(repeat(i0),repeat(i1a),0..i1a+1)),
                  ptr.iter().step_by(d2),
                  ptr[d2..].iter().step_by(d2))
                .map(|((_,i1a,i1b),pb,pe)| if i1a == i1b { pe-pb } else { 2*(pe-pb) } )
                .sum();
        
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(), rnnz, rnelm);

        rptr[0] = 0;
        let mut nzi : usize = 0;
        rptr[1..].iter_mut().enumerate().for_each(|(i,p)| { 
            let i0  = i/(n*n*d2);
            let i1a = (i / (d2*n)) % n;
            let i1b = (i / d2) % n;
            let i2  = i % d2;

            let (i1a,i1b) = (i1a.max(i1b),i1a.min(i1b));
            let si = i0 * d1 * d0 + (i1a*(i1a+1)/2 + i1b)*d2 + i2;

            let p0 = ptr[si];
            let p1 = ptr[si+1];
            let rownzi = p1-p0;
            
            rsubj[nzi..nzi+rownzi].copy_from_slice(&subj[p0..p1]);
            rcof[nzi..nzi+rownzi].copy_from_slice(&cof[p0..p1]);
        
            nzi += rownzi;
            *p = nzi;
        });

//        let d01 = d * (d+1) / 2;
//        let num : usize = tshape[0..tshape.len()-2].product();
//
//        let rnelm = d01 * num;
//        let rnnz = izip!(ptr.chunks(d01),
//                         ptr[1..].chunks(d01))
//            .map(|(pb,pe)| 
//                 izip!((0..d).flat_map(|i| std::iter::repeat(i).zip(0..i+1)), 
//                       pb.iter(),
//                       pe.iter())
//                    .map(|((i,j),&b,&e)| if i==j { e-b } else { 2*(e-b) }))
//            .sum();
//        let (rptr,rsp,rsubj,rval) = rs.alloc_expr(tshape,rnnz, rnelm);
//        rptr[0] = 0;
//        izip!(ptr.chunks(d01),
//              ptr[1..].chunks(d01),
//              rptr[1..].dchunks(d*d))
//            .map(|(pb,pe)| 
//                 izip!(pb.iter(),
//                       pe.iter())
//                    .map(|((i,j),&b,&e)| if i==j { e-b } else { 2*(e-b) }))
//            .sum();
//
//        `
    }
    rs.check();
    Ok(())
}


pub fn inplace_reduce_shape(m : usize,rs : & mut WorkStack, xs : & mut WorkStack)  -> Result<(),ExprEvalError>
{
    rs.validate_top().unwrap();
    let (rshape,_) = xs.alloc(m,0);
    {
        let (shape,_,_,_,_) = rs.peek_expr();
        if m <= shape.len() {
            rshape[..m-1].clone_from_slice(&shape[0..m-1]);
            *rshape.last_mut().unwrap() = shape[m-1..].iter().product();
        }
        else {
            rshape[0..shape.len()].clone_from_slice(shape);
            rshape[shape.len()..].iter_mut().for_each(|s| *s = 1);
        }
    }
    rs.inline_reshape_expr(rshape).or_else(|s| Err(ExprEvalError::new(file!(),line!(),s)))?;
    rs.check();
    Ok(())
}

pub fn inplace_reshape_one_row(m : usize, dim : usize, rs : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> 
{
    if dim >= m { return Err(ExprEvalError::new(file!(),line!(),"Invalid dimension given")); }
        
    let (newshape,_) = xs.alloc(m,0);
    newshape.iter_mut().for_each(|s| *s = 1 );
    newshape[dim] = {
        let (shp,_,_,_,_) = rs.peek_expr();
        shp.iter().product()
    };
    rs.inline_reshape_expr(newshape).unwrap();
    rs.check();
    Ok(())
}

pub fn inplace_reshape(rshape : &[usize],rs : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> 
{
    rs.validate_top().unwrap();
    {
        let (shape,_,_,_,_) = rs.peek_expr();
        if shape.iter().product::<usize>() != rshape.iter().product() {
            return Err(ExprEvalError::new(file!(),line!(),"Cannot reshape expression into given shape"));
        }
    }
    rs.inline_reshape_expr(rshape).unwrap();
    rs.check();
    Ok(())
}

pub fn scatter(rshape : &[usize], sparsity : &[usize], rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> 
{
    let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();

    if ptr.len()-1 != sparsity.len() {
        return Err(ExprEvalError::new(file!(),line!(),"Sparsity pattern does not match number of elements in expression"));
    }

    let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rshape,ptr.len()-1,subj.len());

    rptr.copy_from_slice(ptr);
    rsubj.copy_from_slice(subj);
    rcof.copy_from_slice(cof);

    if let Some(rsp) = rsp {
        rsp.copy_from_slice(sparsity);
    }
    rs.check();
    Ok(())
}

pub fn gather(rshape : &[usize], rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> 
{
    let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();

    if ptr.len()-1 != rshape.iter().product::<usize>() {
        return Err(ExprEvalError::new(file!(),line!(),"Shape does not match number of elements in expression"));
    }

    let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(rshape,ptr.len()-1,subj.len());

    rptr.clone_from_slice(ptr);
    rsubj.clone_from_slice(subj);
    rcof.clone_from_slice(cof);
    rs.check();
    Ok(())
}

pub fn gather_to_vec(rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();
    let rnelm = ptr.len()-1;
    let rnnz  = subj.len();        

    let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr( &[rnelm],rnnz,rnelm);

    rptr.clone_from_slice(ptr);
    rsubj.clone_from_slice(subj);
    rcof.clone_from_slice(cof);
    Ok(())
}





pub fn eval_finalize(rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let rnnz  = subj.len();
    let rnelm = shape.iter().product();
    let (rptr,_,rsubj,rcof) = rs.alloc_expr(shape,rnnz,rnelm);
 
    let (urest,_frest) = xs.alloc(rnnz+ptr.len(),0);
    let (perm,xptr) = urest.split_at_mut(rnnz); 
    xptr[0] = 0;
    perm.iter_mut().enumerate().for_each(|(i,p)| *p = i);
    for (pp,xptr) in perm.chunks_ptr_mut(ptr, &ptr[1..]).zip(xptr[1..].iter_mut())
    {
        pp.sort_by_key(|&p| unsafe{ *subj.get_unchecked(p) });
        *xptr = 
            if pp.len() > 1 {
                subj.permute_by(pp).zip(subj.permute_by(&pp[1..])).filter(|(&a,&b)| a != b ).count()+1
            }
            else {
                pp.len()
            };
    }
    xptr.iter_mut().fold(0usize,|c,p| { *p += c; *p });
    //println!("subj = {:?}",subj);
    //println!("ptr = {:?}",ptr);
    //println!("xptr = {:?}",xptr);
    //println!("xptr.last() = {}, rsubj.len() = {}",xptr.last().unwrap(),rsubj.len());
    //println!("perm = {:?}",perm);

    

    for (rsubj,rcof,perm) in izip!(rsubj.chunks_ptr_mut(xptr,&xptr[1..]),
                                   rcof.chunks_ptr_mut(xptr,&xptr[1..]),
                                   perm.chunks_ptr(ptr))
    {
        //println!("sub perm = {:?}",perm);
        
        let mut sit = subj.permute_by(perm).zip(cof.permute_by(perm));
        let mut tit = rsubj.iter_mut().zip(rcof.iter_mut()).peekable();
        
        if let Some((j,c)) = sit.next() {
            if let Some((jj,cc)) = tit.peek_mut() {
                **jj = *j;
                **cc = *c;
            }
        }

        for (j,c) in sit {
            if let Some((jj,cc)) = tit.peek_mut() {
                if **jj == *j {
                    **cc += *c;
                }
                else {
                    _ = tit.next();
                    if let Some((jj,cc)) = tit.peek_mut() {
                        **jj = *j;
                        **cc = *c;
                    }
                }
            }
        }
    }

    if let Some(sp) = sp {
        rptr.iter_mut().for_each(|p| *p = 0);
        for (i,ni) in sp.iter().zip(ptr.iter().zip(ptr[1..].iter()).map(|v| *v.1-*v.0)) {
            rptr[i+1] = ni;
        }

        rptr.iter_mut().fold(0usize,|c,p| { *p += c; *p } );
    }
    else {
        rptr.copy_from_slice(xptr);
    }

//---
//---
//---    let maxj = subj.iter().max().unwrap_or(&0);
//---    let (jj,ff) = xs.alloc(maxj*2+2,maxj+1);
//---    let (jj,jjind) = jj.split_at_mut(maxj+1);
//---
//---
//---    let mut ii  = 0;
//---    let mut nzi = 0;
//---    rptr[0] = 0;
//---
//---    subj.iter().for_each(|&j| unsafe{ *jjind.get_unchecked_mut(j) = 0; });
//---
//---    if let Some(sp) = sp {
//---        println!("eval_finalize(), Sparse");
//---        
//---        rsubj.copy_from_slice(subj);
//---        rcof.copy_from_slice(cof);
//---        
//---        let (strides,_) = xs.alloc(nd,0);
//---        strides.iter_mut().zip(shape.iter()).fold(1usize, |c,(s,&d)| { *s = c; d*c });
//---
//---
//---
//---
//---
//---        for (&i,&p0,&p1) in izip!(sp, ptr[0..ptr.len()-1].iter(), ptr[1..].iter()) {
//---            if ii < i { rptr[ii+1..i+1].fill(nzi); ii = i; }
//---
//---            let mut rownzi : usize = 0;
//---            subj[p0..p1].iter().for_each(|&j| unsafe{ *jjind.get_unchecked_mut(j) = 0; });
//---            subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&j,&c)| {
//---                if c == 0.0 {}
//---                else if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
//---                    unsafe{
//---                        *jjind.get_unchecked_mut(j)   = 1;
//---                        *jj.get_unchecked_mut(rownzi) = j;
//---                        *ff.get_unchecked_mut(j)      = c;
//---                    }
//---                    rownzi += 1;
//---                }
//---                else {
//---                    unsafe{
//---                        *ff.get_unchecked_mut(j) += c;
//---                    }
//---                }
//---            });
//---
//---            izip!(jj[0..rownzi].iter(),
//---                  rsubj[nzi..nzi+rownzi].iter_mut(),
//---                  rcof[nzi..nzi+rownzi].iter_mut())
//---                .for_each(|(&j,rj,rc)| {
//---                    *rc = unsafe{ *ff.get_unchecked(j) };
//---                    unsafe{ *jjind.get_unchecked_mut(j) = 0; };
//---                    *rj = j;
//---                });
//---
//---            nzi += rownzi;
//---            // println!("ExprTrait::eval_finalize sparse: nzi = {}",nzi);
//---            rptr[i+1] = nzi;
//---            ii += 1;
//---        }
//---    }
//---    else {
//---        for (&p0,&p1,rp) in izip!(ptr[0..ptr.len()-1].iter(),ptr[1..].iter(),rptr[1..].iter_mut()) {
//---
//---            let mut rownzi : usize = 0;
//---
//---            subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&j,&c)| {
//---                // println!("-- j = {}, c = {}, ind = {}",j,c,jjind[j]);
//---                if c == 0.0 {}
//---                else if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
//---                    unsafe{
//---                        *jjind.get_unchecked_mut(j)   = 1;
//---                        *jj.get_unchecked_mut(rownzi) = j;
//---                        *ff.get_unchecked_mut(j)      = c;
//---                    }
//---                    rownzi += 1;
//---                }
//---                else {
//---                    unsafe{ *ff.get_unchecked_mut(j) += c; }
//---                }
//---            });
//---
//---            izip!(jj[0..rownzi].iter(),
//---                  rsubj[nzi..nzi+rownzi].iter_mut(),
//---                  rcof[nzi..nzi+rownzi].iter_mut()).for_each(|(&j,rj,rc)| {
//---                      *rc = unsafe{ *ff.get_unchecked(j) };
//---                      unsafe{ *jjind.get_unchecked_mut(j) = 0; };
//---                      *rj = j;
//---                  });
//---
//---            jj[0..rownzi].fill(0);
//---
//---            nzi += rownzi;
//---            // println!("ExprTrait::eval_finalize dense: nzi = {}",nzi);
//---            *rp = nzi;
//---            ii += 1;
//---        }
//---    }
    rs.check();
    Ok(())
} // eval_finalize





#[cfg(test)]
mod test {
    use crate::expr::workstack::WorkStack;
    #[test]
    fn test_mul_matrix_expr_transpose_dd() {
        let mut rs = WorkStack::new(1024);
        let mut ws = WorkStack::new(1024);
        let mut xs = WorkStack::new(1024);
   
        println!("Dense x Dense");
        {
            let (ptr,_sp,subj,cof) = ws.alloc_expr(&[3,3], 18, 9);
            ptr.iter_mut().zip((0..20).step_by(2)).for_each(|(p,i)| *p = i);
            subj.iter_mut().step_by(2).zip(0..9).for_each(|(j,i)| *j = i);
            subj[1..].iter_mut().step_by(2).zip(9..18).for_each(|(j,i)| *j = i);
            cof.iter_mut().enumerate().for_each(|(i,c)| *c = (i+1) as f64);

            super::mul_matrix_expr_transpose(
                (3,3),
                None,
                &[1.1,1.2,1.3,2.1,2.2,2.3,3.1,3.2,3.3],
                & mut rs,& mut ws,&mut xs).unwrap();
            let (rshape,rptr,rsp,rsubj,rcof) = rs.pop_expr();
            // | 1.1 1.2 1.3 |   | 1(x0+x9)  2(x1+x10) 3(x2+x11) | 
            // | 2.1 2.2 2.3 | x | 4(x3+x12) 5(x4+x13) 6(x5+x14) |
            // | 3.1 3.2 3.3 |   | 7(x6+x15) 8(x7+x16) 9(x8+x17) |

            assert_eq!(rshape.len(),2);
            assert_eq!(rshape[0],3);
            assert_eq!(rshape[1],3);
            assert_eq!(rptr.len(),10);
            assert_eq!(rsubj.len(),6*9);
            assert!(rsubj[0..18].iter().zip([0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17].iter()).all(|(&a,&b)| a == b));
            //assert!(rcof[0..18].iter().step_by(2).zip([1.1,1.2*2.0,1.3*3.0, 1.1*4.0,1.2*5.0,1.3*6.0, 1.1*7.0,1.2*8.0,1.3*9.0 ].iter()).all(|(&a,&b)| (a-b).abs() < 1e-8));
        }
    }
    #[test]
    fn test_mul_matrix_expr_transpose_sd() {
        let mut rs = WorkStack::new(1024);
        let mut ws = WorkStack::new(1024);
        let mut xs = WorkStack::new(1024);
   
        println!("Sparse x Dense");
        {
            let (ptr,_sp,subj,cof) = ws.alloc_expr(&[3,3], 18, 9);
            ptr.iter_mut().zip((0..20).step_by(2)).for_each(|(p,i)| *p = i);
            subj.iter_mut().step_by(2).zip(0..9).for_each(|(j,i)| *j = i);
            subj[1..].iter_mut().step_by(2).zip(9..18).for_each(|(j,i)| *j = i);
            cof.iter_mut().enumerate().for_each(|(i,c)| *c = (i+1) as f64);

            // | 1.1     1.3 |   | 1(x0+x9)  2(x1+x10) 3(x2+x11) | 
            // |             | x | 4(x3+x12) 5(x4+x13) 6(x5+x14) |
            // |     3.2     |   | 7(x6+x15) 8(x7+x16) 9(x8+x17) |

            super::mul_matrix_expr_transpose(
                (3,3),
                Some(&[0,2,7]),
                &[1.1,1.3,3.2],
                & mut rs,& mut ws,&mut xs).unwrap();
            
            let (rshape,rptr,rsp,rsubj,rcof) = rs.pop_expr();

            assert_eq!(rshape,&[3,3]);
            assert_eq!(rptr,&[0,4,8,12,14,16,18]);
            assert_eq!(rsubj,&[0,9,2,11,3,12,5,14,6,15,8,17, 1,10,4,13,7,16]);
            assert_eq!(rsp.unwrap(),&[0,1,2,6,7,8]);
            //assert!(rcof.iter().step_by(2).zip([1.1, 1.3*3.0, 1.1*4.0, 1.3*6.0, 3.2*2.0, 3.2*5.0, 3.2*8.0].iter()).all(|(&a,&b)| (a-b).abs() < 1e-8));
        }
    }
    #[test]
    fn test_mul_matrix_expr_transpose_ds() {
        let mut rs = WorkStack::new(1024);
        let mut ws = WorkStack::new(1024);
        let mut xs = WorkStack::new(1024);
   
        println!("Dense x Sparse");
        {
            let (ptr,sp,subj,cof) = ws.alloc_expr(&[3,3], 6, 3);
            ptr.copy_from_slice(&[0,2,4,6]);
            sp.unwrap().copy_from_slice(&[0,2,7]);
            subj.copy_from_slice(&[0,9,2,11,7,16]);
            cof.iter_mut().enumerate().for_each(|(i,c)| *c = (i+1) as f64);

            // | 1.1 1.2 1.3 |   | 1(x0+x9)            2(x2+x11) | 
            // | 2.1 2.2 2.3 | x |                               |
            // | 3.1 3.2 3.3 |   |           3(x7+x16)           |

            super::mul_matrix_expr_transpose(
                (3,3),
                None,
                &[1.1,1.2,1.3,2.1,2.2,2.3,3.1,3.2,3.3],
                & mut rs,& mut ws,&mut xs).unwrap();
            
            let (rshape,rptr,rsp,rsubj,rcof) = rs.pop_expr();

            assert_eq!(rshape.len(),2);
            assert_eq!(rshape[0],3);
            assert_eq!(rshape[1],3);
            assert_eq!(rptr,&[0,4,6,10,12,16,18]);
            let expect_subj = &[0,9,2,11, 7,16, 0,9,2,11, 7,16, 0,9,2,11, 7,16];
            println!("rsubj = {:?}\n        {:?}",rsubj,expect_subj);
            assert_eq!(rsp.unwrap(),&[0,2,3,5,6,8]);
            assert_eq!(rsubj,expect_subj);
            //assert!(rcof.iter().step_by(2).zip([1.1,1.3*2.0, 1.2*3.0, 2.1*1.0, 2.3*2.0, 2.2*3.0, 3.1*1.0, 3.3*2.0, 3.2*3.0].iter()).all(|(&a,&b)| (a-b).abs() < 1e-8));
        }
    }
    #[test]
    fn test_mul_matrix_expr_transpose_ss() {
        let mut rs = WorkStack::new(1024);
        let mut ws = WorkStack::new(1024);
        let mut xs = WorkStack::new(1024);
   
        println!("Sparse x Sparse");
        {
            let (ptr,sp,subj,cof) = ws.alloc_expr(&[3,3], 6, 3);
            ptr.copy_from_slice(&[0,2,4,6]);
            sp.unwrap().copy_from_slice(&[0,2,7]);
            subj.copy_from_slice(&[0,9,2,11,7,16]);
            cof.iter_mut().enumerate().for_each(|(i,c)| *c = (i+1) as f64);

            super::mul_matrix_expr_transpose(
                (3,3),
                Some(&[0,2,4,5,7]),
                &[1.1,1.3,2.2,2.3,3.2],
                & mut rs,& mut ws,&mut xs).unwrap();
            
            let (rshape,rptr,rsp,rsubj,rcof) = rs.pop_expr();
            // | 1.1     1.3 |   | 1(x0+x9)            2(x2+x11) | 
            // |             | x |           3(x4+x13) 4(x5+x14) |
            // |     3.2     |   |           5(x7+x16)           |

            assert_eq!(rshape.len(),2);
            assert_eq!(rshape[0],3);
            assert_eq!(rshape[1],3);
            assert_eq!(rptr.len(),5);
            assert_eq!(rsubj.len(),10);
            assert!(rsp.unwrap().iter().zip([0,1,7,8].iter()).all(|(&a,&b)| a==b));
            assert!(rsubj.iter().zip([0,9,2,11, 5,14, 4,13, 7,16].iter()).all(|(&a,&b)| a==b));
            //assert!(rcof.iter().step_by(2).zip([1.1, 1.3*2.0, 1.3*4.0, 3.2*3.0, 3.2*5.0].iter()).all(|(&a,&b)| (a-b).abs() < 1e-8));
        }
    }
}

