# Comparative performance tests

Comparing expression evaluation performance for MosekCOModel vs Java Fusion.

Run a series of performance tests, printing a table with the results.
Optionally, also compiles and runs the corresponding examples as implemented in
Java (requires `javac` and `java`).

To run tests and print a Markdown formatted table:
```
cargo run --release --bin compare -- --compare --style md -cp PATH/TO/mosek.jar
```

To run individual examples (useful for profiling), say `mul1`:
```
cargo run --release --bin runtest -- mul1
```
