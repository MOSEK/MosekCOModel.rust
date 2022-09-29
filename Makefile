all: 
	PATH="/remote/public/linux/64-x86/rust/current/bin:$(PATH)" cargo build

test: 
	PATH="/remote/public/linux/64-x86/rust/current/bin:$(PATH)" cargo test
	
.PHONY: all test
