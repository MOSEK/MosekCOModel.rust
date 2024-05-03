// This module defines the generic expression evaluation, i.e. the
// evaluation part that is not concerned with expression sub-part
// types.

use std::iter::once;
use super::*;
use super::super::utils;
use super::workstack::WorkStack;

use itertools::{izip,iproduct};

pub(super) fn repeat(dim : usize, num : usize, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    if dim >= shape.len() {
        panic!("Invalid stacking dimension");
    }
    let mut rshape = shape.to_vec();
    rshape[dim] *= num;
    let nelm = ptr.len()-1;
    let rnnz = ptr.last().unwrap()*num;
    let rnelm = (ptr.len()-1)*num;

    println!("repeat: \n\tshape = {:?}\n\tptr = {:?}\n\tsubj = {:?}",shape,ptr,subj);
    println!("repeat: rnnz = {}, rnelm = {}", rnnz,rnelm);

    let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(), rnnz, rnelm);

    if let (Some(ref sp),Some(rsp)) = (sp,rsp) {
        let d0 : usize = shape[..dim].iter().product();
        let d1 : usize = shape[dim];
        let d2 : usize = shape[dim+1..].iter().product();
        let rd1 = rshape[dim];
        let (uslice,_) = xs.alloc(rnelm*3,0);
        let (xsp,uslice) = uslice.split_at_mut(rnelm);
        let (xidx,perm)  = uslice.split_at_mut(rnelm);

        for (xspi,xi,(k,spi,i)) in izip!(xsp.iter_mut(),
                                         xidx.iter_mut(),
                                         (0..num).map(|i| izip!(0..nelm,sp.iter(),std::iter::repeat(i))).flatten()) {
            let (i0,i1,i2) = (spi / (d1*d2), (spi / d2) % d1, spi % d2);
            *xspi = i0 * rd1 * d2 + (i1 + i * d1) * d2 + i2;
            *xi = k;
            println!("sp i {} -> ({},{},{}) -> {}",spi,i0,i1,i2,*xspi);
        }

        perm.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        perm.sort_by_key(|&i| xsp[i]);

        rptr.iter_mut().for_each(|p| *p = 0);
        rsp.iter_mut().zip(perm_iter(perm,xsp)).for_each(|(t,&s)| *t = s);

        println!("xidx = {:?}",xidx);
        let mut p = 0usize;
        for (rptr,&i) in izip!(rptr[1..].iter_mut(), perm_iter(perm,xidx)) {
            let ptrb = ptr[i];
            let ptre = ptr[i+1];
            let n = ptre-ptrb;
            println!("Copy [{}..{}] -> [{}..{}]",ptrb,ptre,p,p+n);
            *rptr = n;
            rsubj[p..p+n].copy_from_slice(&subj[ptrb..ptre]);
            rcof[p..p+n].copy_from_slice(&cof[ptrb..ptre]);
            p += n;
        }
        _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
        println!("repeat sparse:\n\trptr = {:?}\n\trsubj = {:?}\n\trcof = {:?}\n\trsp = {:?}",rptr,rsubj,rcof,rsp);
    } 
    else { // dense
        let d0 : usize = num * shape[..dim].iter().product::<usize>();
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
}


pub(super) fn permute_axes(perm : &[usize],
                           rs : & mut WorkStack,
                           ws : & mut WorkStack,
                           xs : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nelem = ptr.len()-1;
    let nd = shape.len();

    if perm.len() != nd || *perm.iter().max().unwrap() >= nd {
        panic!("Mismatching permutation and shape");
    }
    let mut rshape = vec![usize::MAX; shape.len()];
    perm.iter().for_each(|&i| {
        unsafe {
            if i >= shape.len() || *rshape.get_unchecked(i) < usize::MAX  {
                panic!("Invalid permutation");
            }
            *rshape.get_unchecked_mut(i) = *shape.get_unchecked(i);
        }
    });

    let nd = shape.len();
    let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(),subj.len(),*ptr.last().unwrap());
    rptr[0] = 0;
        
    let (uslice,_)           = xs.alloc(nelem*2+shape.len()*3,0);
    let (spx,uslice)         = uslice.split_at_mut(nelem);
    let (elmperm,uslice)     = uslice.split_at_mut(nelem);
    let (strides,uslice)     = uslice.split_at_mut(nd);
    let (rstrides,prstrides) = uslice.split_at_mut(nd);


    strides.iter_mut().rev().zip(shape.iter().rev().fold_map0(1usize,|cum,d| d*cum )).for_each(|(t,s)| *t = s);
    rstrides.iter_mut().rev().zip(rshape.iter().rev().fold_map0(1,|cum,d| d*cum )).for_each(|(t,s)| *t = s);
    prstrides.iter_mut().zip(perm.iter()).for_each(|(s,&p)| *s = rstrides[p] );

    if let Some(sp) = sp {
        //println!("permute_axes: sparse, strides = {:?}, rstrides = {:?}, prstrides = {:?}",strides,rstrides,prstrides);
        spx.iter_mut().zip(sp.iter()).for_each(|(ix,&i)| {
            let (_,ri) = strides.iter().zip(prstrides.iter()).fold((i,0),|(v,r),(&s,&rs)| (v%s,r+(v/s)*rs));
            *ix = ri;
        });

        elmperm.iter_mut().enumerate().for_each(|(i,t)| *t = i);
        elmperm.sort_by_key(|&i| unsafe{* spx.get_unchecked(i) });
       
        // apply permutation
        if let Some(rsp) = rsp { 
            for (t,&s) in rsp.iter_mut().zip(perm_iter(elmperm,sp)) {
                (_,*t) = izip!(strides.iter(),prstrides.iter()).fold((s,0),|(sv,r),(&s,&ps)| (sv % s,r + (sv/s)*ps)  );
            }
            //println!("\trsp = {:?}",rsp);
        };
        //let _ = rptr.iter_mut().zip(perm_iter(ptr[0..nelem].iter()).zip(ptr[1..].iter())).fold(0,|c,(t,(&p0,&p1))| { *t = c+p1-p0; *t });
        rptr.iter_mut().for_each(|p| *p = 0);

        { 
            let mut nzi = 0;
    
            for (&p0,&p1,p) in izip!(perm_iter(elmperm,&ptr[0..nelem]),perm_iter(elmperm,&ptr[1..nelem+1]),rptr[1..].iter_mut()) {
                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                *p = p1-p0;
                nzi += p1-p0;
            }
            _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p } );
            println!("permute_axes: sparse\n\trsubj = {:?}\n\trcof = {:?}\n\trptr = {:?}",rsubj,rcof,rptr);
        }
    }
    else {
        //println!("permute_axes: dense");
        for (si,n) in ptr.iter().zip(ptr[1..].iter()).map(|(&p0,&p1)| p1-p0).enumerate() {
            let (_,ti) = strides.iter().zip(prstrides.iter()).fold((si,0),|(v,r),(&s,&rs)| (v%s,r+(v/s)*rs));
            rptr[ti+1] = n
        }
        let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
        
        for (si,(ssubj,scof)) in izip!(subj.chunks_by(ptr),cof.chunks_by(ptr)).enumerate() {
            let (_,ti) = strides.iter().zip(prstrides.iter()).fold((si,0),|(v,r),(&s,&rs)| (v%s,r+(v/s)*rs));
            let n = ssubj.len();
            let nzi = rptr[ti];

            rsubj[nzi..nzi+n].clone_from_slice(ssubj);
            rcof[nzi..nzi+n].clone_from_slice(scof);
        }
    }
    //println!("eval::permute_axes: end");
}


/// Add `n` expression residing on `ws`. Result pushed to `rs`.
pub(super) fn add(n  : usize,
                  rs : & mut WorkStack,
                  ws : & mut WorkStack,
                  xs : & mut WorkStack) {
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
                izip!(subj.chunks_by(ptr),
                      cof.chunks_by(ptr),
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
                izip!(subj.chunks_by(ptr),
                      cof.chunks_by(ptr),
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
            let mut h = utils::IndexHashMap::new(hdata,hindex,hnext,hbucket,0);

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
                                         perm_iter(perm, hindex),
                                         perm_iter(perm, hdata)) {
                *ri = si;
                *rp = sd;
            }
            hindex[..rnelm].clone_from_slice(rsp);
        }
        else {
            for (rp,&sd) in izip!(rptr[1..].iter_mut(), perm_iter(perm, hdata)) {
                *rp = sd;
            }
            hindex[..rnelm].iter_mut().enumerate().for_each(|(i,x)| *x = i);
        }
        // cummulate ptr
        _ = rptr.iter_mut().fold(0,|c,p| { *p += c; *p });
                 
        // Create a mapping from linear sparsity index into sp/ptr index
        hdata.iter_mut().enumerate().for_each(|(i,d)| *d = i);
        let h = utils::IndexHashMap::with_data(&mut hdata[..rnelm],
                                               &mut hindex[..rnelm],
                                               &mut hnext[..rnelm],
                                               &mut hbucket[..rnelm],
                                               0);

        for (_,ptr,sp,subj,cof) in exprs.iter() {
            if let Some(sp) = sp {
                for (&i,sj,sc) in izip!(sp.iter(),
                                        subj.chunks_by(ptr),
                                        cof.chunks_by(ptr)) {
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
        //println!("rptr = {:?}",rptr); 
    }
    //println!("eval::add: end");
} // add

/// Evaluates `lhs` * expr.
pub(super) fn mul_left_dense(mdata : &[f64],
                             mdimi : usize,
                             mdimj : usize,
                             rs    : & mut WorkStack,
                             ws    : & mut WorkStack,
                             _xs    : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let nd   = shape.len();
    let nnz  = subj.len();

    if nd != 2 && nd != 1{ panic!("Invalid shape for multiplication") }
    if mdimj != shape[0] { panic!("Mismatching shapes for multiplication") }

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

    //println!("{}:{}: rnelm = {}, rnnz = {}",file!(),line!(),rnelm,rnnz);

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
} // mul_left_dense

pub(super) fn mul_right_dense(mdata : &[f64],
                              mdimi : usize,
                              mdimj : usize,
                              rs    : & mut WorkStack,
                              ws    : & mut WorkStack,
                              _xs    : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let nd   = shape.len();
    let nnz  = subj.len();
    //let nelm = ptr.len()-1;
    if nd != 2 && nd != 1{ panic!("Invalid shape for multiplication") }
    let (edimi,edimj) = if let Some(d2) = shape.get(1) {
        (shape[0],*d2)
    }
    else {
        (1,shape[0])
    };

    if mdimi != edimj { panic!("Mismatching shapes for multiplication") }

    let rdimi = edimi;
    let rdimj = mdimj;
    let rnnz = nnz * mdimj;
    let rnelm = rdimi * rdimj;

    //println!("{}:{}: dimi = {}, dimj = {}",file!(),line!(),edimi,edimj);

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
} // mul_right_dense


pub(super) fn mul_left_sparse(mheight : usize,
                              mwidth : usize,
                              msparsity : &[usize],
                              mdata : &[f64],
                              rs : & mut WorkStack,
                              ws : & mut WorkStack,
                              xs : & mut WorkStack) {

    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nd = shape.len();
    if nd != 1 && nd != 2 { panic!("Expression is incorrect shape for multiplication: {:?}",shape); }
    if shape[0] != mwidth { panic!("Mismatching operand shapes for multiplication"); }
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
            for (k,&ix) in perm_iter(perm,sp).enumerate() {
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
                let mut espi = izip!(perm_iter(&perm[ep0..ep1],sp),
                                     perm_iter(&perm[ep0..ep1],ptr),
                                     perm_iter(&perm[ep0..ep1],&ptr[1..])).peekable();
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
                let mut espi = izip!(perm_iter(&perm[ep0..ep1],sp),
                                     perm_iter(&perm[ep0..ep1],ptr),
                                     perm_iter(&perm[ep0..ep1],&ptr[1..])).peekable();
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
}

// expr x matrix
pub(super) fn mul_right_sparse(mheight : usize,
                               mwidth : usize,
                               msparsity : &[usize],
                               mdata : &[f64],
                               rs : & mut WorkStack,
                               ws : & mut WorkStack,
                               xs : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    if shape.len() != 1 && shape.len() != 2 {
        panic!("Invalid operand shapes: Expr is not 1- or 2-dimensional");
    }

    let (eheight,ewidth) =
        if let Some(&d) = shape.get(1) { (shape[0],d) }
        else { (1,shape[0]) };

    if ewidth != mheight {
        panic!("Incompatible operand shapes");
    }

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
                rnnz += izip!(perm_iter(mcolsubi,p0s),
                              perm_iter(mcolsubi,p1s)).map(|(&p0,&p1)| p1-p0).sum::<usize>();
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
                izip!(perm_iter(mcolsubi,p0s),
                      perm_iter(mcolsubi,p1s),
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
}

pub(super) fn dot_sparse(sparsity : &[usize],
                         data     : &[f64],
                         rs : & mut WorkStack,
                         ws : & mut WorkStack,
                         _xs : & mut WorkStack) {
    let (_shape,ptr,sp,subj,cof) = ws.pop_expr();
    

}

pub(super) fn dot_vec(data : &[f64],
                      rs : & mut WorkStack,
                      ws : & mut WorkStack,
                      _xs : & mut WorkStack) {

    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    let nd   = shape.len();
    let nnz  = subj.len();

    if nd != 1 || shape[0] != data.len() {
        panic!("Mismatching operands");
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
} // dot_slice



pub(super) fn stack(dim : usize, n : usize, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
    // println!("{}:{}: eval::stack n={}, dim={}",file!(),line!(),n,dim);
    let exprs = ws.pop_exprs(n);

    for (i,e)  in exprs.iter().enumerate() {
        //println!("stack expr {}: shape = {:?}",i,e.0);
    }

    // check shapes
    if ! exprs.iter().zip(exprs[1..].iter()).any(|((s0,_,_,_,_),(s1,_,_,_,_))| shape_eq_except(s0,s1,dim)) {
        panic!("Mismatching shapes or stacking dimension");
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
        rptr[0] = 0;
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
        // println!("{}:{}: rptr = {:?}",file!(),line!(),rptr);
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

                    // println!("{}:{}:\n\trptr[...] = {:?}\n\tptr[...] = {:?}\n\tptr+1[...] = {:?}, ofs = {}",
                    //          file!(),line!(),&rps,p0s,p1s,ofs);

                    let rp = rps[ofs];
                    rsubj[rp..rp+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[rp..rp+p1-p0].clone_from_slice(&cof[p0..p1]);

                });

            ofs += vd1;
        }
    }
}

pub(super) fn sum_last(num : usize, rs : & mut WorkStack, ws : & mut WorkStack, _xs : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let d = shape[shape.len()-num..].iter().product();
    let mut rshape = shape.to_vec();
    rshape[shape.len()-num..].iter_mut().for_each(|s| *s = 1);

//    println!("sum_last: shape = {:?}",shape);
//    println!("sum_last: ptr = {:?}",ptr);
//    println!("sum_last: sp = {:?}",sp);
//    println!("sum_last: subj = {:?}",subj);
//    println!("sum_last: cof = {:?}",cof);


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
            for (rp,(p,v)) in izip!(rptr[1..].iter_mut(),
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
        //println!("sum_last sparse");
        //println!("sum_last: rnelm = {:?}",rnelm);
        //println!("sum_last: rsubj = {:?}",rsubj);
        //println!("sum_last: rcof = {:?}",rcof);
        //println!("sum_last: rptr = {:?}",rptr);
    } 
    else {
        let rnelm = shape.iter().product::<usize>()/d; 
        let (rptr,_,rsubj,rcof) = rs.alloc_expr(rshape.as_slice(),subj.len(),rnelm);

        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);
        rptr.iter_mut().zip(ptr.iter().step_by(d)).for_each(|(rp,&p)| *rp = p );
        
        //println!("sum_last dense");
        //println!("sum_last: rnelm = {:?}",rnelm);
        //println!("sum_last: rsubj = {:?}",rsubj);
        //println!("sum_last: rcof = {:?}",rcof);
        //println!("sum_last: rptr = {:?}",rptr);
    }
    //println!("eval::sum_last: end");
}

pub(super) fn eval_finalize(rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();

    let nnz  = subj.len();
    let nelm = shape.iter().product();
    let (rptr,_,rsubj,rcof) = rs.alloc_expr(shape,nnz,nelm);
 
    let maxj = subj.iter().max().unwrap_or(&0);
    let (jj,ff) = xs.alloc(maxj*2+2,maxj+1);
    let (jj,jjind) = jj.split_at_mut(maxj+1);


    let mut ii  = 0;
    let mut nzi = 0;
    rptr[0] = 0;

    subj.iter().for_each(|&j| unsafe{ *jjind.get_unchecked_mut(j) = 0; });

    if let Some(sp) = sp {
        for (&i,&p0,&p1) in izip!(sp, ptr[0..ptr.len()-1].iter(), ptr[1..].iter()) {
            if ii < i { rptr[ii+1..i+1].fill(nzi); ii = i; }

            let mut rownzi : usize = 0;
            subj[p0..p1].iter().for_each(|&j| unsafe{ *jjind.get_unchecked_mut(j) = 0; });
            subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&j,&c)| {
                if c == 0.0 {}
                else if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
                    unsafe{
                        *jjind.get_unchecked_mut(j)   = 1;
                        *jj.get_unchecked_mut(rownzi) = j;
                        *ff.get_unchecked_mut(j)      = c;
                    }
                    rownzi += 1;
                }
                else {
                    unsafe{
                        *ff.get_unchecked_mut(j) += c;
                    }
                }
            });

            izip!(jj[0..rownzi].iter(),
                  rsubj[nzi..nzi+rownzi].iter_mut(),
                  rcof[nzi..nzi+rownzi].iter_mut())
                .for_each(|(&j,rj,rc)| {
                    *rc = unsafe{ *ff.get_unchecked(j) };
                    unsafe{ *jjind.get_unchecked_mut(j) = 0; };
                    *rj = j;
                });

            nzi += rownzi;
            // println!("ExprTrait::eval_finalize sparse: nzi = {}",nzi);
            rptr[i+1] = nzi;
            ii += 1;
        }
    }
    else {
        for (&p0,&p1,rp) in izip!(ptr[0..ptr.len()-1].iter(),ptr[1..].iter(),rptr[1..].iter_mut()) {

            // izip!(subj[p0..p1].iter().map(|&v| v)
            //       cof[p0..p1].iter().map(|&v| v))
            //     .partial_fold_map(|(j0,v0),(j,v)| if j0 == j { Some(v0+v) } else { None })
            //     .for_each(|(j0,v0)| {
            //         .....
            //     });
            let mut rownzi : usize = 0;

            subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&j,&c)| {
                // println!("-- j = {}, c = {}, ind = {}",j,c,jjind[j]);
                if c == 0.0 {}
                else if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
                    unsafe{
                        *jjind.get_unchecked_mut(j)   = 1;
                        *jj.get_unchecked_mut(rownzi) = j;
                        *ff.get_unchecked_mut(j)      = c;
                    }
                    rownzi += 1;
                }
                else {
                    unsafe{ *ff.get_unchecked_mut(j) += c; }
                }
            });

            izip!(jj[0..rownzi].iter(),
                  rsubj[nzi..nzi+rownzi].iter_mut(),
                  rcof[nzi..nzi+rownzi].iter_mut()).for_each(|(&j,rj,rc)| {
                      *rc = unsafe{ *ff.get_unchecked(j) };
                      unsafe{ *jjind.get_unchecked_mut(j) = 0; };
                      *rj = j;
                  });

            jj[0..rownzi].fill(0);

            nzi += rownzi;
            // println!("ExprTrait::eval_finalize dense: nzi = {}",nzi);
            *rp = nzi;
            ii += 1;
        }
    }
}


