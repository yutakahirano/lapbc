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
Args:
  filename = in.qasm
  mapping filename = data/6x6.mapping.json
  output source filename = None
  file format = None
  schedule output filename = None
  print_operations = false
  use_pi_over_8_rotation_block = false
  code_distance = 15
  magic_state_distillation_cost = 21
  magic_state_distillation_success_probability = 0.5
  num_distillations_for_pi_over_8_rotation = 6
  num_distillations_for_pi_over_8_rotation_block = 3
  single_qubit_pi_over_8_rotation_block_depth_ratio = 1.1
  single_qubit_arbitrary_angle_rotation_precision = 1e-10
  preferable_distillation_area_size = 5
  parallelism = 1
  num_executions = 10

Parsing the QASM file...done.
num lapbc ops = 871
spc_ops.len = 271, len * distance = 4065, spc cycles = 4065
num cycles (without blocks) = 2881
Scheduling is done.
Run (WithoutBlocks): runtime cycle = 2881, delay = 0
Run (WithoutBlocks): runtime cycle = 2881, delay = 0
Run (WithoutBlocks): runtime cycle = 2890, delay = 9
Run (WithoutBlocks): runtime cycle = 2881, delay = 0
Run (WithoutBlocks): runtime cycle = 2889, delay = 8
Run (WithoutBlocks): runtime cycle = 2881, delay = 0
Run (WithoutBlocks): runtime cycle = 2881, delay = 0
Run (WithoutBlocks): runtime cycle = 2881, delay = 0
Run (WithoutBlocks): runtime cycle = 2881, delay = 0
Run (WithoutBlocks): runtime cycle = 2930, delay = 49
Average runtime cycles[without blocks] = 2887.6
Average delay[without blocks] = 6.6
```

Here, `in.qasm` is the input QASM program, and `data/6x6.mapping.json` is the qubit mapping configuration.

The available command-line options are defined in the struct `Args` in [main.rs](src/main.rs).

### Caveats
 - You need to prepare `qelib1.inc` and place it in the top directory.
 - The QASM loading library we use is very slow. For larger programs, it is recommended to convert the QASM file into an internal JSON representation.
   This can be done with the following command:
   ```
   [yhirano@host lapbc]$ target/release/lapbc ... --filename=in.qasm --output-source-filename=out.json
   ```
   Then the JSON representation is stored in `out.json`, which can be reused for subsequent runs:
   ```
   [yhirano@host lapbc]$ target/release/lapbc ... --filename=out.json --file-format=JSON
   ```

