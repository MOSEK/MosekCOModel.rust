extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};
extern crate mosekmodel;

use mosekmodel::{*, expr::workstack::WorkStack};

// Operation specific benchmarks

fn make_6d_variable(m : & mut Model, sp : bool, n : usize) -> Variable<6> {
    let shape = [n,n,n,n,n,n];
    if ! sp {
        m.variable(None,&shape)
    }
    else {
        let nelm = shape.iter().product();
        let mut strides = [0usize;6]; strides.iter_mut().zip(shape.iter()).rev().fold(1usize,|c,(s,&d)| { *s = c; c*d });
        //println!("shape = {:?}, strides = {:?}", shape,strides);
        let sparsity = (0..nelm).step_by(13).map(|i| { let mut r = [0usize;6]; r.iter_mut().zip(strides.iter()).fold(i,|i,(r,&s)| {*r = i/s; i%s}); r }).collect::<Vec<[usize;6]>>();
        //println!("sparsiy = {:?}", &sparsity[..100]);
        m.variable(None,unbounded().with_shape_and_sparsity(&shape, &sparsity))
    }
}


// ExprTrait::axispermute
fn bench_axis_permute(c : & mut Criterion, sp : bool, n : usize) {
    let mut m = Model::new(None);
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let v = make_6d_variable(&mut m, sp, n);

    c.bench_function(
        format!("axispermute-{}-{}",if sp {"sparse"} else {"dense"},n).as_str(), 
        |b| b.iter(|| {
            rs.clear(); ws.clear(); xs.clear();
            let v = v.clone();
            v.axispermute(&[3,2,1,0,4,5]).axispermute(&[2,0,1,4,3,5]).axispermute(&[5,4,3,2,1,0])
                .eval_finalize(&mut rs,&mut ws,&mut xs).unwrap()
        }));
}

// ExprTrait::sum_on 
fn bench_sum_on( c : &mut Criterion, sp : bool, n : usize, axes : [usize;3]) {
    let mut m = Model::new(None);
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);
    let v = make_6d_variable(&mut m, sp, n);
    
    c.bench_function(
        format!("sumon-{}-{}-{}{}{}",if sp {"sparse"} else {"dense"},n,axes[0],axes[1],axes[2]).as_str(),
        |b| b.iter(|| {
            rs.clear(); ws.clear(); xs.clear();
            let e = v.clone().add(v.clone().axispermute(&[3,2,1,0,4,5])).add(v.clone().axispermute(&[2,0,1,4,3,5]));
            e.sum_on(&axes).eval_finalize(& mut rs, &mut ws, &mut xs).unwrap();
        }));
}


fn bench_stack(c : & mut Criterion, d : usize, sp : bool, n : usize) {
    use utils::ShapeToStridesEx;
    c.bench_function(
        format!("stack-{}-{}-{}",if sp {"sparse"} else {"dense"},d,n).as_str(), 
        |b| {
            let mut m = Model::new(None);
            let shape = [n,n,n];
            let v = 
                if sp {
                    let st = shape.to_strides();
                    let sp = (0..shape.iter().product()).step_by(10).map(|i| st.to_index(i)).collect::<Vec<[usize;3]>>();
                    m.variable(None, unbounded().with_shape_and_sparsity(&shape, sp.as_slice()))
                }
                else {
                    m.variable(None,&shape)
                };
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            b.iter(|| {
                rs.clear(); ws.clear(); xs.clear();
                v.clone().stack(d,v.clone()).stack(d,v.clone()).eval_finalize(& mut rs,& mut ws, & mut xs).unwrap();
            });
        });
}







fn bench_repeat(c : & mut Criterion, sp : bool, d : usize, n : usize, rep : usize) {
    use utils::ShapeToStridesEx;
    c.bench_function(
        format!("repeat-{}-{}-{}-{}",if sp {"sparse"} else {"dense"},d,n,rep).as_str(), 
        |b| {
            let mut m = Model::new(None);
            let shape = [n,n,n];
            let v = 
                if sp {
                    let st = shape.to_strides();
                    let sp = (0..shape.iter().product()).step_by(10).map(|i| st.to_index(i)).collect::<Vec<[usize;3]>>();
                    m.variable(None, unbounded().with_shape_and_sparsity(&shape, sp.as_slice()))
                }
                else {
                    m.variable(None,&shape)
                };
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            b.iter(|| {
                rs.clear(); ws.clear(); xs.clear();
                v.clone().repeat(d,rep).eval_finalize(& mut rs,& mut ws, & mut xs).unwrap();
            });
        });
}



#[derive(Debug,Clone,Copy)]
enum Sparsity {
    Sparse,
    Dense
}
fn bench_mul(c : & mut Criterion, vsp : Sparsity, dsp : Sparsity, rev : bool, n : usize) {
    use utils::ShapeToStridesEx;
    c.bench_function(
        format!("mul-{:?}-{:?}-{}",vsp,dsp,if rev {"rev"} else {"fwd"}).as_str(), 
        |b| {
            let mut m = Model::new(None);
            let shape = [n,n];
            let st = shape.to_strides();
            let (v,w) = 
                match vsp {
                    Sparsity::Sparse => {
                        let sp = (0..shape.iter().product()).step_by(10).map(|i| st.to_index(i)).collect::<Vec<[usize;2]>>();
                        (m.variable(None, unbounded().with_shape_and_sparsity(&shape, sp.as_slice())),
                         m.variable(None, unbounded().with_shape_and_sparsity(&shape, sp.as_slice())).transpose())
                    },
                    Sparsity::Dense => (m.variable(None,&shape),
                                        m.variable(None,&shape).transpose()),
                };
            let mx = 
                match dsp {
                    Sparsity::Sparse =>  {
                        let sp = (0..n*n).step_by(7).map(|i| st.to_index(i)).collect::<Vec<[usize;2]>>();
                        let num = sp.len();
                        matrix::sparse( [n,n], sp, vec![1.0; num])
                    },
                    Sparsity::Dense => matrix::dense([n,n], vec![1.0;n*n]),
                };

            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);


            if rev {
                b.iter(|| {
                    rs.clear(); ws.clear(); xs.clear();
                    v.clone().add(w.clone()).mul(mx.clone()).eval_finalize(& mut rs,& mut ws, & mut xs).unwrap();
                });
            }
            else {
                b.iter(|| {
                    rs.clear(); ws.clear(); xs.clear();
                    mx.clone().mul(v.clone().add(w.clone())).eval_finalize(& mut rs,& mut ws, & mut xs).unwrap();
                });
            }
        });
}

fn bench_mul_diag(c : & mut Criterion, vsp : Sparsity, dsp : Sparsity, rev : bool, n : usize) {
    use utils::ShapeToStridesEx;
    c.bench_function(
        format!("mul-diag-{:?}-{:?}-{}",vsp,dsp,if rev {"rev"} else {"fwd"}).as_str(), 
        |b| {
            let mut m = Model::new(None);
            let shape = [n,n];
            let st = shape.to_strides();
            let (v,w) = 
                match vsp {
                    Sparsity::Sparse => {
                        let sp = (0..shape.iter().product()).step_by(10).map(|i| st.to_index(i)).collect::<Vec<[usize;2]>>();
                        (m.variable(None, unbounded().with_shape_and_sparsity(&shape, sp.as_slice())),
                         m.variable(None, unbounded().with_shape_and_sparsity(&shape, sp.as_slice())).transpose())
                    },
                    Sparsity::Dense => (m.variable(None,&shape),
                                        m.variable(None,&shape).transpose()),
                };
            let mx = 
                match dsp {
                    Sparsity::Sparse =>  {
                        let sp = (0..n*n).step_by(7).map(|i| st.to_index(i)).collect::<Vec<[usize;2]>>();
                        let num = sp.len();
                        matrix::sparse( [n,n], sp, vec![1.0; num])
                    },
                    Sparsity::Dense => matrix::dense([n,n], vec![1.0;n*n]),
                };

            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            if rev {
                b.iter(|| {
                    rs.clear(); ws.clear(); xs.clear();
                    v.clone().add(w.clone()).dot_rows(mx.clone().transpose()).eval_finalize(& mut rs,& mut ws, & mut xs).unwrap();
                });
            }
            else {
                b.iter(|| {
                    rs.clear(); ws.clear(); xs.clear();
                    v.clone().add(w.clone()).transpose().dot_rows(mx.clone()).eval_finalize(& mut rs,& mut ws, & mut xs).unwrap();
                });
            }
        });
}








const N : usize = 10;

fn bench_axis_permute_dense_10( c : & mut Criterion) { bench_axis_permute(c,false,N)}
fn bench_axis_permute_sparse_10(c : & mut Criterion) { bench_axis_permute(c,true,N) }

fn bench_sum_on_dense_10_012(   c : & mut Criterion) { bench_sum_on(c,false,N,[0,1,2]) }
fn bench_sum_on_dense_10_345(   c : & mut Criterion) { bench_sum_on(c,false,N,[3,4,5]) }
fn bench_sum_on_dense_10_135(   c : & mut Criterion) { bench_sum_on(c,false,N,[1,3,5]) }
fn bench_sum_on_sparse_10_012(  c : & mut Criterion) { bench_sum_on(c,true,N,[0,1,2]) }
fn bench_sum_on_sparse_10_345(  c : & mut Criterion) { bench_sum_on(c,true,N,[3,4,5]) }
fn bench_sum_on_sparse_10_135(c : & mut Criterion) { bench_sum_on(c,true,N,[1,3,5]) }

fn bench_stack_dense_0_256(c : & mut Criterion) { bench_stack(c,0,false,256) }
fn bench_stack_dense_1_256(c : & mut Criterion) { bench_stack(c,1,false,256) }
fn bench_stack_dense_2_256(c : & mut Criterion) { bench_stack(c,2,false,256) }
fn bench_stack_sparse_0_374(c : & mut Criterion) { bench_stack(c,0,true,374) }
fn bench_stack_sparse_1_374(c : & mut Criterion) { bench_stack(c,1,true,374) }
fn bench_stack_sparse_2_374(c : & mut Criterion) { bench_stack(c,2,true,374) }

fn bench_repeat_dense_0_256(c : & mut Criterion)  { bench_repeat(c,false,0,256,3) }
fn bench_repeat_dense_1_256(c : & mut Criterion)  { bench_repeat(c,false,1,256,3) }
fn bench_repeat_dense_2_256(c : & mut Criterion)  { bench_repeat(c,false,2,256,3) }
fn bench_repeat_sparse_0_374(c : & mut Criterion) { bench_repeat(c,true, 0,374,3) }
fn bench_repeat_sparse_1_374(c : & mut Criterion) { bench_repeat(c,true, 1,374,3) }
fn bench_repeat_sparse_2_374(c : & mut Criterion) { bench_repeat(c,true, 2,374,3) }

fn bench_mul_dense_dense_fwd(c : & mut Criterion)   { bench_mul(c,Sparsity::Dense, Sparsity::Dense, false,256) }
fn bench_mul_dense_dense_rev(c : & mut Criterion)   { bench_mul(c,Sparsity::Dense, Sparsity::Dense, true, 128) }
fn bench_mul_dense_sparse_fwd(c : & mut Criterion)  { bench_mul(c,Sparsity::Dense, Sparsity::Sparse,false,512) }
fn bench_mul_dense_sparse_rev(c : & mut Criterion)  { bench_mul(c,Sparsity::Dense, Sparsity::Sparse,true, 512) }
fn bench_mul_sparse_dense_fwd(c : & mut Criterion)  { bench_mul(c,Sparsity::Sparse,Sparsity::Dense, false,512) }
fn bench_mul_sparse_dense_rev(c : & mut Criterion)  { bench_mul(c,Sparsity::Sparse,Sparsity::Dense, true, 512) }
fn bench_mul_sparse_sparse_fwd(c : & mut Criterion) { bench_mul(c,Sparsity::Sparse,Sparsity::Sparse,false,512) }
fn bench_mul_sparse_sparse_rev(c : & mut Criterion) { bench_mul(c,Sparsity::Sparse,Sparsity::Sparse,true, 512) }

fn bench_mul_diag_dense_dense_fwd(c : & mut Criterion)   { bench_mul_diag(c,Sparsity::Dense, Sparsity::Dense, false,512) }
fn bench_mul_diag_dense_dense_rev(c : & mut Criterion)   { bench_mul_diag(c,Sparsity::Dense, Sparsity::Dense, true, 512) }
fn bench_mul_diag_dense_sparse_fwd(c : & mut Criterion)  { bench_mul_diag(c,Sparsity::Dense, Sparsity::Sparse,false,512) }
fn bench_mul_diag_dense_sparse_rev(c : & mut Criterion)  { bench_mul_diag(c,Sparsity::Dense, Sparsity::Sparse,true, 512) }
fn bench_mul_diag_sparse_dense_fwd(c : & mut Criterion)  { bench_mul_diag(c,Sparsity::Sparse,Sparsity::Dense, false,512) }
fn bench_mul_diag_sparse_dense_rev(c : & mut Criterion)  { bench_mul_diag(c,Sparsity::Sparse,Sparsity::Dense, true, 512) }
fn bench_mul_diag_sparse_sparse_fwd(c : & mut Criterion) { bench_mul_diag(c,Sparsity::Sparse,Sparsity::Sparse,false,512) }
fn bench_mul_diag_sparse_sparse_rev(c : & mut Criterion) { bench_mul_diag(c,Sparsity::Sparse,Sparsity::Sparse,true, 512) }

criterion_group!(
    name=benches;
    config=Criterion::default().sample_size(10);
    targets=
        bench_axis_permute_dense_10,
        bench_axis_permute_sparse_10,
        bench_sum_on_dense_10_012,
        bench_sum_on_dense_10_345,
        bench_sum_on_dense_10_135,
        bench_sum_on_sparse_10_012,
        bench_sum_on_sparse_10_345,
        bench_sum_on_sparse_10_135,

        bench_stack_dense_0_256,
        bench_stack_dense_1_256,
        bench_stack_dense_2_256,
        bench_stack_sparse_0_374,
        bench_stack_sparse_1_374,
        bench_stack_sparse_2_374,

        bench_repeat_dense_0_256, 
        bench_repeat_dense_1_256, 
        bench_repeat_dense_2_256, 
        bench_repeat_sparse_0_374,
        bench_repeat_sparse_1_374,
        bench_repeat_sparse_2_374,

        bench_mul_dense_dense_fwd,
        bench_mul_dense_dense_rev,
        bench_mul_dense_sparse_fwd,
        bench_mul_dense_sparse_rev,
        bench_mul_sparse_dense_fwd,
        bench_mul_sparse_dense_rev,
        bench_mul_sparse_sparse_fwd,
        bench_mul_sparse_sparse_rev,

        bench_mul_diag_dense_dense_fwd,
        bench_mul_diag_dense_dense_rev,
        bench_mul_diag_dense_sparse_fwd,
        bench_mul_diag_dense_sparse_rev,
        bench_mul_diag_sparse_dense_fwd,
        bench_mul_diag_sparse_dense_rev,
        bench_mul_diag_sparse_sparse_fwd,
        bench_mul_diag_sparse_sparse_rev
    );
criterion_main!(benches);
