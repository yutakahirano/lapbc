# Locality-aware Pauli-based computation

This repository contains the code used for the paper ["Locality-aware Pauli-based computation for local magic state preparation" (arXiv:2504.12091)](https://arxiv.org/abs/2504.12091).

The implementation is written in **Rust** and **Python**.

### Directory Structure

The `scripts/` directory contains Python scripts for generating QASM programs and mapping configurations.

 - **`scripts/rcs.py`** generates Random circuit sampling QASM programs.
 - **`scripts/trotter.py`** generates QASM programs for Hamiltonian simulations with Trotterization.
 - **`scripts/mapping_generator.py`** generates standard and sparse qubit mapping configurations.

The `src/` directory contains the Rust implementation of the scheduler and the runtime simulator.

### Building

See [https://rust-lang.org/tools/install/](https://rust-lang.org/tools/install/) for installing the Rust toolchain.
Building the scheduler and simulator is easy:
```bash
[yhirano@host lapbc]$ cargo build
```

or 

```bash
[yhirano@host lapbc]$ cargo build --release
```


### Running
Using `cargo run [--release]` is the simplest way:
```bash
[yhirano@host lapbc]$ cargo run --release -- --filename=in.qasm --mapping-filename=data/6x6.mapping.json
```

Alternatively, you can directly run the compiled binary:
```bash
[yhirano@host lapbc]$ target/release/lapbc --filename=in.qasm --mapping-filename=data/6x6.mapping.json
```

Here, `in.qasm` is the input QASM program, and `data/6x6.mapping.json` is the qubit mapping configuration.

The available command-line options are defined in the struct `Args` in [main.rs](src/main.rs).

