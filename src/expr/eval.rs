// This module defines the generic expression evaluation, i.e. the
// evaluation part that is not concerned with expression sub-part
// types.

use super::*;
use super::super::utils;
use super::workstack::WorkStack;

use itertools::{izip};

pub(super) fn add(n  : usize,
       rs : & mut WorkStack,
       ws : & mut WorkStack,
       xs : & mut WorkStack) {
    // check that shapes match
    let exprs = ws.pop_exprs(n);
    
    if exprs.iter().map(|(shape,_,_,_,_)| shape)
        .zip(exprs[1..].iter().map(|(shape,_,_,_,_)| shape))
        .any(|(shape1,shape2)| shape1.len() != shape2.len() || shape1.iter().zip(shape2.iter()).any(|(&d1,&d2)| d1 != d2)) {
            panic!("Mismatching operand shapes");
        }

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
                        rsubj[p0..p0+nnz].clone_from_slice(&js);
                        rcof[p0..p0+nnz].clone_from_slice(&cs);

                        unsafe{ *rptr.get_unchecked_mut(i) += nnz };
                    });
            }
            else {
                izip!(subj.chunks_by(ptr),
                      cof.chunks_by(ptr),
                      rptr.iter())
                    .for_each(|(js,cs,&p0)| {
                        let nnz = js.len();
                        rsubj[p0..p0+nnz].clone_from_slice(&js);
                        rcof[p0..p0+nnz].clone_from_slice(&cs);
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
        // cummulate
        let _ = perm.iter().fold(0,|v,&p| {
            let d = unsafe{ &mut*hdata.get_unchecked_mut(p) };
            let prev = *d;
            unsafe{ *rptr.get_unchecked_mut(p+1) = prev+v; }
            *d = v;
            prev + v
        });
        if let Some(sp) = rsp {
            let _ = izip!(hperm.iter(),sp.iter_mut()).for_each(|(&p,sp)| {
                *sp = unsafe{ *hindex.get_unchecked(p) };
            });
        }

        let mut h = utils::IndexHashMap::with_data(&mut hdata[..rnelm],
                                                   &mut hindex[..rnelm],
                                                   &mut hnext[..rnelm],
                                                   &mut hbucket[..rnelm],
                                                   0);

        for (_,ptr,sp,subj,cof) in exprs.iter() {
            if let Some(sp) = sp {
                izip!(sp.iter(),ptr.iter(),ptr[1..].iter())
                    .for_each(|(&i,&p0,&p1)| {
                        let d = h.at_mut(i);
                        let n = p1-p0;
                        rsubj[*d..*d+n].clone_from_slice(&subj[p0..p1]);
                        rcof[*d..*d+n].clone_from_slice(&cof[p0..p1]);
                        *d += n;
                    });
            }
        }
    }

    // let rnelm =
    //     match (sp0,sp1) {
    //         (None,_) | (_,None) => ptr0.len()+ptr1.len()-2,
    //         (Some(sp0),Some(sp1)) => {
    //             let mut i0 = sp0.iter().peekable();
    //             let mut i1 = sp1.iter().peekable();
    //             let mut n = 0;
    //             while match (i0.peek(),i1.peek()) {
    //                 (Some(_j0),None) => { let _ =  i0.next(); n += 1; true },
    //                 (None,Some(_j1)) => { let _ =  i1.next(); n += 1; true },
    //                 (Some(&j0),Some(&j1)) => {
    //                     n += 1;
    //                     if j0 < j1 { let _ = i0.next(); }
    //                     else if j1 < j0 { let _ = i1.next(); }
    //                     else { let _ = i0.next(); let _ = i1.next(); }
    //                     true
    //                 },
    //                 _ => false
    //             }{ /*empty loop body*/}
    //             n
    //         }
    //     };
}


pub(super) fn mul_left(lhs   : &Matrix,
            rs    : & mut WorkStack,
            ws    : & mut WorkStack,
            xs    : & mut WorkStack) {
    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    
    let nd   = shape.len();
    let nnz  = subj.len();
    let nelm = ptr.len()-1;
    let (mdimi,mdimj) = lhs.dim;


    if nd != 2 && nd != 1{ panic!("Invalid shape for multiplication") }
    if mdimj != shape[0] { panic!("Mismatching shapes for multiplication") }

    let rdimi = mdimi;
    let rdimj = if nd == 1 { 1 } else { shape[1] };
    let edimi = shape[0];
    let rnnz = nnz * mdimi;
    let rnelm = mdimi * rdimj;

    let (perm_spptr,mrowdata) = xs.alloc(nelm+rdimj+1,lhs.data.len());
    if lhs.rows { mrowdata.clone_from_slice(lhs.data.as_slice()); }
    else {
        iproduct!((0..mdimi),(0..mdimj)).zip(mrowdata.iter_mut())
            .for_each(|((i,j),dst)| *dst = unsafe { * lhs.data.get_unchecked(j*mdimi+i) });
    }

    let (rptr,_rsp,rsubj,rcof) = if nd == 2 {
        rs.alloc_expr(&[rdimi,rdimj],rnnz,rnelm)
    }
    else {
        rs.alloc_expr(&[rdimi],rnnz,rnelm)
    };

    rptr[0] = 0;
    let mut elmi = 0;
    let mut nzi  = 0;

    rptr[0] = 0;

    if let Some(sp) = sp {
        let (perm,spptr) = perm_spptr.split_at_mut(sp.len());
        spptr.iter_mut().for_each(|v| *v = 0);
        perm.iter_mut().enumerate().for_each(|(i,pi)| *pi = i );
        perm.sort_by_key(|&k| {
            let spi = unsafe{ sp.get_unchecked(k) };
            let ii = spi / rdimj; let jj = spi - ii*rdimj;
            unsafe { *spptr.get_unchecked_mut(jj+1) += 1 };
            jj * edimi + ii
        });

        { let mut cum = 0; spptr.iter_mut().for_each(|v| { let tv = *v; *v = cum; cum = tv; } ); }

        // loop over matrix rows x expr columns
        iproduct!(mrowdata.chunks(mdimj),
                  spptr[..rdimj].iter().zip(spptr[1..].iter()))
            .for_each(|(mrow,(&sp0,&sp1))| {
                izip!(sp[sp0..sp1].iter(),ptr[sp0..sp1].iter(),ptr[sp0+1..sp1+1].iter()).for_each(|(&spi,&p0,&p1)| {
                    let spi_i = spi / rdimj;
                    //let spi_j = spi % rdimj;
                    let v = unsafe { *mrow.get_unchecked(spi_i) };
                    izip!(subj[p0..p1].iter(),
                          cof[p0..p1].iter(),
                          rsubj[nzi..nzi+p1-p0].iter_mut(),
                          rcof[nzi..nzi+p1-p0].iter_mut())
                        .for_each(|(&j,&c,rj,rc)| { *rj = j; *rc = v*c; });
                    nzi += p1-p0;
                    elmi += 1;
                    rptr[nelm] = nzi;
                });
            });
    }
    else {
        mrowdata.chunks(mdimj).for_each(|mrowi| { // for each matrix row
            (0..rdimj).for_each(|j| { // for each expression column
                izip!(mrowi.iter(), ptr[j..].iter().step_by(rdimj), ptr[j+1..].iter().step_by(rdimj)).for_each(|(&c,&p0,&p1)| {
                    rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rv,&v)| *rv = c*v );
                    nzi += p1-p0;
                });
                elmi += 1;
                rptr[elmi] = nzi;
                // println!("rptr[{}] = {}",elmi,nzi);
            });
        });
    }
}


pub(super) fn dot_slice(data : &[f64],
             rs : & mut WorkStack,
             ws : & mut WorkStack,
             _xs : & mut WorkStack) {

    let (shape,ptr,sp,subj,cof) = ws.pop_expr();
    println!("ExprDot::eval: subj = {:?}, cof = {:?}",subj,cof);
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
        println!("ExprDot::eval: result nnz = {}, nelm = {}, ptr = {:?}, subj = {:?}, data = {:?}",rnnz,rnelm,rptr,rsubj,data);
        for (&p0,&p1,v) in izip!(ptr[0..ptr.len()-1].iter(),ptr[1..].iter(),data.iter()) {
            for (&c,rc) in cof[p0..p1].iter().zip(rcof[p0..p1].iter_mut()) {
                *rc = c*v;
            }
        }
    }
}
