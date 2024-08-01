all: target/release/libmosekmodel.rlib
	make -C ./examples/lowner-john-outer all
	make -C ./examples/lowner-john-outer-3d all
	make -C ./examples/ellipsoid_approximation all

test: target/release/libmosekmodel.rlib 
	cargo test

target/release/libmosekmodel.rlib: src/matrix.rs src/lib.rs src/domain.rs src/expr src/expr/add.rs src/expr/workstack.rs src/expr/eval.rs src/expr/mul.rs src/expr/dot.rs src/expr/mod.rs src/utils.rs src/variable.rs
	cargo build --release

clean:
	rm -rf target/release
	make -C ./examples/lowner-john-outer clean
	make -C ./examples/lowner-john-outer-3d clean
	make -C ./examples/ellipsoid_approximation clean

.PHONY: all test

.DEFAULT: all
