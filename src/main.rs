extern crate clap;
// extern crate oq3_lexer;
// extern crate oq3_parser;
// extern crate oq3_source_file;
extern crate qasm;

use board::BoardOccupancy;
use board::Configuration;
use board::OperationWithAdditionalData;
use clap::Parser;
use pbc::Operation;
use rand::seq::SliceRandom;
use rand_distr::Distribution;
use rand_distr::Normal;
use std::env;
use std::io::IsTerminal;
use std::io::Write as _;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;

mod board;
mod lapbc;
mod mapping;
mod pbc;
mod runner;

#[cfg(test)]
mod testutils;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The filename of the QASM file to be translated.
    #[arg(short, long)]
    filename: String,

    #[arg(short, long)]
    mapping_filename: String,

    #[arg(short, long)]
    output_source_filename: Option<String>,

    #[arg(short, long)]
    file_format: Option<String>,

    #[arg(short, long)]
    schedule_output_filename: Option<String>,

    #[arg(short, long, default_value_t = false)]
    print_operations: bool,

    #[arg(short, long, default_value_t = false)]
    use_pi_over_8_rotation_block: bool,

    #[arg(short, long, default_value_t = 15_u32)]
    code_distance: u32,

    #[arg(short, long, default_value_t = 21_u32)]
    magic_state_distillation_cost: u32,

    #[arg(short, long, default_value_t = 0.5)]
    magic_state_distillation_success_probability: f64,

    #[arg(short, long, default_value_t = 6)]
    num_distillations_for_pi_over_8_rotation: u32,

    #[arg(short, long, default_value_t = 3)]
    num_distillations_for_pi_over_8_rotation_block: u32,

    #[arg(short, long, default_value_t = 1.1)]
    single_qubit_pi_over_8_rotation_block_depth_ratio: f64,

    #[arg(short, long, default_value_t = 1e-10)]
    single_qubit_arbitrary_angle_rotation_precision: f64,

    #[arg(short, long, default_value_t = 5)]
    preferable_distillation_area_size: u32,

    #[arg(short, long, default_value_t = 10)]
    num_executions: u32,

    #[arg(short, long, default_value_t = 1)]
    parallelism: u32,
}

use pbc::Angle;
use pbc::Axis;
use pbc::Mod8;
use pbc::Pauli;
use pbc::PauliRotation;

#[derive(serde::Deserialize, serde::Serialize)]
struct Registers {
    qregs: Vec<(String, u32)>,
    cregs: Vec<(String, u32)>,
}

impl Registers {
    fn new() -> Self {
        Registers {
            qregs: Vec::new(),
            cregs: Vec::new(),
        }
    }

    fn add_qreg(&mut self, name: String, size: u32) {
        assert!(!self.is_qreg(&name));
        assert!(!self.is_creg(&name));
        self.qregs.push((name, size));
    }

    fn add_creg(&mut self, name: String, size: u32) {
        assert!(!self.is_qreg(&name));
        assert!(!self.is_creg(&name));
        self.cregs.push((name, size));
    }

    fn is_qreg(&self, name: &str) -> bool {
        self.qregs.iter().any(|(n, _)| n == name)
    }
    fn is_creg(&self, name: &str) -> bool {
        self.cregs.iter().any(|(n, _)| n == name)
    }

    fn qubit_index(&self, name: &str, index: u32) -> Option<u32> {
        let mut qubit_index = 0;
        for (n, size) in &self.qregs {
            if n == name {
                return if index < *size {
                    Some(qubit_index + index)
                } else {
                    None
                };
            }
            qubit_index += size;
        }
        None
    }

    fn classical_bit_index(&self, name: &str, index: u32) -> Option<u32> {
        let mut bit_index = 0;
        for (n, size) in &self.cregs {
            if n == name {
                return if index < *size {
                    Some(bit_index + index)
                } else {
                    None
                };
            }
            bit_index += size;
        }
        None
    }

    fn num_qubits(&self) -> u32 {
        self.qregs.iter().map(|(_, size)| *size).sum()
    }
}

fn extract_qubit(
    args: &[qasm::Argument],
    args_index: u32,
    registers: &Registers,
    context: &str,
) -> Result<u32, String> {
    if let qasm::Argument::Qubit(qubit, index) = &args[args_index as usize] {
        if *index < 0 {
            return Err(format!("{}: args[{}] must be non-negative", context, args_index));
        }
        if let Some(index) = registers.qubit_index(qubit, *index as u32) {
            Ok(index)
        } else {
            Err(format!("{}: there is no qubit {}[{}]", context, qubit, index))
        }
    } else {
        Err(format!("{}: args[{}] must be a qubit", context, args_index))
    }
}

fn extract_classical_bit(
    args: &[qasm::Argument],
    args_index: u32,
    registers: &Registers,
    context: &str,
) -> Result<u32, String> {
    if let qasm::Argument::Qubit(qubit, index) = &args[args_index as usize] {
        if *index < 0 {
            return Err(format!("{}: args[{}] must be non-negative", context, args_index));
        }
        if let Some(index) = registers.classical_bit_index(qubit, *index as u32) {
            Ok(index)
        } else {
            Err(format!("{}: there is no classical bit {}[{}]", context, qubit, index))
        }
    } else {
        Err(format!("{}: args[{}] must be a classical bit", context, args_index))
    }
}

// Extracts `s` as an angle.
// The input is a QASM-style string, e.g., an argument for a RZ gate.
// The output is in Litinski's style. This function accounts for the style difference,
// so extract_angle(" pi / 2 ") returns Ok(Angle::PiOver8(Two)), for instance.
fn extract_angle(s: &str, context: &str) -> Result<Angle, String> {
    use Mod8::*;
    let pattern =
        regex::Regex::new(r"^ *(?<sign>-)? *((?<n>[0-9]+) *\*)? *pi *(/ *(?<m>[0-9]+))? *$")
            .unwrap();
    let arbitrary_angle_pattern =
        regex::Regex::new(r"^ *(?<sign>-)? *(?<a>[0-9]+\.[0-9]+) *$").unwrap();
    if s.trim() == "" {
        Err(format!("{}: angle must not be empty", context))
    } else if s.trim() == "0" {
        Ok(Angle::PiOver8(Zero))
    } else if let Some(caps) = pattern.captures(s) {
        let has_minus = caps.name("sign").is_some();
        let n = caps
            .name("n")
            .map_or(Ok(1), |n| n.as_str().parse::<u32>())
            .map_err(|e| format!("{}: invalid angle: {}", context, e))?;
        let m = caps
            .name("m")
            .map_or(Ok(1), |m| m.as_str().parse::<u32>())
            .map_err(|e| format!("{}: invalid angle: {}", context, e))?;
        let n_with_pi_over_8 = match m {
            1 => 4 * n % 8,
            2 => 2 * n % 8,
            4 => n % 8,
            _ => {
                return Err(format!("{}: invalid angle: {}", context, s));
            }
        };
        if has_minus {
            Ok(Angle::PiOver8(-Mod8::from(n_with_pi_over_8)))
        } else {
            Ok(Angle::PiOver8(Mod8::from(n_with_pi_over_8)))
        }
    } else if let Some(caps) = arbitrary_angle_pattern.captures(s) {
        let sign = if caps.name("sign").is_some() {
            -1.0
        } else {
            1.0
        };
        let a = caps
            .name("a")
            .unwrap()
            .as_str()
            .parse::<f64>()
            .map_err(|e| format!("{}: invalid angle: {}", context, e))?;
        Ok(Angle::Arbitrary(sign * a / 2.0))
    } else {
        Err(format!("{}: invalid angle: {}", context, s))
    }
}

fn translate_gate(
    name: &str,
    args: &[qasm::Argument],
    angle_args: &[String],
    registers: &Registers,
    output: &mut Vec<Operation>,
) -> Result<(), String> {
    use Mod8::*;
    use Operation::Measurement as M;
    use Operation::PauliRotation as R;
    let num_qubits = registers.num_qubits();
    match name {
        "x" | "y" | "z" => {
            if args.len() != 1 {
                return Err(format!("Invalid number of arguments for {}: {}", name, args.len()));
            }
            if !angle_args.is_empty() {
                return Err(format!(
                    "Invalid number of angle arguments for {}: {}",
                    name,
                    angle_args.len()
                ));
            }
            let pauli = match name {
                "x" => Pauli::X,
                "y" => Pauli::Y,
                "z" => Pauli::Z,
                _ => unreachable!(),
            };
            output.push(R(PauliRotation::new(
                Axis::new_with_pauli(
                    extract_qubit(args, 0, registers, name)? as usize,
                    num_qubits as usize,
                    pauli,
                ),
                Angle::PiOver8(Four),
            )));
            return Ok(());
        }
        "rz" => {
            if args.len() != 1 {
                return Err(format!("Invalid number of arguments for rz: {}", args.len()));
            }
            if angle_args.len() != 1 {
                return Err(format!(
                    "Invalid number of angle arguments for rz: {}",
                    angle_args.len()
                ));
            }
            let qubit = extract_qubit(args, 0, registers, "rz")?;
            let angle = extract_angle(&angle_args[0], "rz")?;
            if angle != Angle::PiOver8(Zero) {
                let axis = Axis::new_with_pauli(qubit as usize, num_qubits as usize, Pauli::Z);
                output.push(R(PauliRotation::new(axis, angle)));
            }
        }
        "ry" => {
            if args.len() != 1 {
                return Err(format!("Invalid number of arguments for ry: {}", args.len()));
            }
            if angle_args.len() != 1 {
                return Err(format!(
                    "Invalid number of angle arguments for ry: {}",
                    angle_args.len()
                ));
            }
            let qubit = extract_qubit(args, 0, registers, "ry")?;
            let angle = extract_angle(&angle_args[0], "ry")?;
            if angle != Angle::PiOver8(Zero) {
                let axis = Axis::new_with_pauli(qubit as usize, num_qubits as usize, Pauli::Y);
                output.push(R(PauliRotation::new(axis, angle)));
            }
        }
        "sx" => {
            if args.len() != 1 {
                return Err(format!("Invalid number of arguments for sx: {}", args.len()));
            }
            if !angle_args.is_empty() {
                return Err(format!(
                    "Invalid number of angle arguments for sx: {}",
                    angle_args.len()
                ));
            }
            let qubit = extract_qubit(args, 0, registers, "sx")?;
            let axis = Axis::new_with_pauli(qubit as usize, num_qubits as usize, Pauli::X);
            output.push(R(PauliRotation::new(axis, Angle::PiOver8(Two))));
        }
        "h" => {
            if args.len() != 1 {
                return Err(format!("Invalid number of arguments for h: {}", args.len()));
            }
            if !angle_args.is_empty() {
                return Err(format!(
                    "Invalid number of angle arguments for h: {}",
                    angle_args.len()
                ));
            }
            let qubit = extract_qubit(args, 0, registers, "h")?;
            let axis_x = Axis::new_with_pauli(qubit as usize, num_qubits as usize, Pauli::X);
            let axis_z = Axis::new_with_pauli(qubit as usize, num_qubits as usize, Pauli::Z);
            // H = S * S_x * S
            output.push(R(PauliRotation::new(axis_z.clone(), Angle::PiOver8(Two))));
            output.push(R(PauliRotation::new(axis_x, Angle::PiOver8(Two))));
            output.push(R(PauliRotation::new(axis_z, Angle::PiOver8(Two))));
        }
        "cx" => {
            if args.len() != 2 {
                return Err(format!("Invalid number of arguments for cx: {}", args.len()));
            }
            if !angle_args.is_empty() {
                return Err(format!(
                    "Invalid number of angle arguments for cx: {}",
                    angle_args.len()
                ));
            }
            let control = extract_qubit(args, 0, registers, "cx")?;
            let target = extract_qubit(args, 1, registers, "cx")?;
            if control == target {
                return Err("cx: control and target must be different".to_string());
            }
            let axis = Axis::new_with_pauli(control as usize, num_qubits as usize, Pauli::Z);
            output.push(R(PauliRotation::new(axis, -Angle::PiOver8(Two))));

            let axis = Axis::new_with_pauli(target as usize, num_qubits as usize, Pauli::X);
            output.push(R(PauliRotation::new(axis, -Angle::PiOver8(Two))));

            let mut axis = vec![Pauli::I; num_qubits as usize];
            axis[control as usize] = Pauli::Z;
            axis[target as usize] = Pauli::X;
            output.push(R(PauliRotation::new(Axis::new(axis), Angle::PiOver8(Two))));
        }
        "measure" => {
            if args.len() != 2 {
                return Err(format!("Invalid number of arguments for measure: {}", args.len()));
            }
            if !angle_args.is_empty() {
                return Err(format!(
                    "Invalid number of angle arguments for measure: {}",
                    angle_args.len()
                ));
            }
            let qubit = extract_qubit(args, 0, registers, "measure")?;
            let _ = extract_classical_bit(args, 1, registers, "measure")?;
            let mut axis = vec![Pauli::I; num_qubits as usize];
            axis[qubit as usize] = Pauli::Z;
            output.push(M(Axis::new(axis)));
        }
        _ => {
            return Err(format!("Unrecognized gate: {}", name));
        }
    }
    Ok(())
}

fn extract(nodes: &[qasm::AstNode]) -> Option<(Vec<Operation>, Registers)> {
    use qasm::AstNode;
    let mut registers = Registers::new();
    if !nodes.iter().all(|node| match node {
        AstNode::QReg(..) => true,
        AstNode::CReg(..) => true,
        AstNode::Barrier(..) => false,
        AstNode::Reset(..) => true,
        AstNode::Measure(..) => true,
        AstNode::ApplyGate(..) => true,
        AstNode::Opaque(..) => false,
        AstNode::Gate(..) => true,
        AstNode::If(..) => false,
    }) {
        eprintln!("Unrecognized node in the AST");
        return None;
    }

    let nodes = nodes
        .iter()
        .filter(|node| match node {
            AstNode::QReg(..) => true,
            AstNode::CReg(..) => true,
            AstNode::Reset(..) => true,
            AstNode::Measure(..) => true,
            AstNode::ApplyGate(..) => true,

            AstNode::Gate(..) => false,
            AstNode::If(..) => false,
            _ => unreachable!("We mustn't be here as we've already checked unsupported nodes."),
        })
        .collect::<Vec<_>>();

    // Let's construct the registers first.
    for node in &nodes {
        match node {
            AstNode::QReg(name, num_qubits) => {
                if registers.is_qreg(name) || registers.is_creg(name) {
                    println!("Duplicate register name: {}", name);
                    return None;
                }
                if *num_qubits < 0 {
                    println!("The number of qubits in a register must be non-negative");
                    return None;
                }
                let num_qubits = *num_qubits as u32;
                registers.add_qreg(name.clone(), num_qubits);
            }
            AstNode::CReg(name, num_bits) => {
                if registers.is_qreg(name) || registers.is_creg(name) {
                    println!("Duplicate register name: {}", name);
                    return None;
                }
                if *num_bits < 0 {
                    println!("The number of qubits in a register must be non-negative");
                    return None;
                }
                let num_bits = *num_bits as u32;
                registers.add_creg(name.clone(), num_bits);
            }
            _ => (),
        }
    }
    let mut ops = Vec::new();
    for node in &nodes {
        match node {
            AstNode::ApplyGate(name, args, angle_args) => {
                if let Err(e) = translate_gate(name, args, angle_args, &registers, &mut ops) {
                    println!("{}", e);
                    return None;
                }
            }
            AstNode::Measure(arg1, arg2) => {
                if let Err(e) = translate_gate(
                    "measure",
                    &[arg1.clone(), arg2.clone()],
                    &[],
                    &registers,
                    &mut ops,
                ) {
                    println!("{}", e);
                    return None;
                }
            }
            _ => (),
        }
    }
    Some((ops, registers))
}

fn print_line_potentially_with_colors(line: &str) {
    if std::io::stdout().is_terminal() {
        let re = regex::Regex::new(r"([IXYZSH][IXYZSH]+)").unwrap();
        if let Some(caps) = re.captures(line) {
            let m = caps.get(1).unwrap();
            let mut colored_text = String::new();
            for c in m.as_str().chars() {
                match c {
                    'I' => colored_text.push_str("\x1b[38;5;8mI\x1b[0m"),
                    'X' => colored_text.push_str("\x1b[38;5;9mX\x1b[0m"),
                    'Y' => colored_text.push_str("\x1b[38;5;10mY\x1b[0m"),
                    'Z' => colored_text.push_str("\x1b[38;5;12mZ\x1b[0m"),
                    'S' => colored_text.push_str("\x1b[38;5;11mS\x1b[0m"),
                    'H' => colored_text.push_str("\x1b[38;5;13mH\x1b[0m"),
                    _ => colored_text.push(c),
                }
            }
            println!("{}{}{}", &line[..m.start()], colored_text, &line[m.end()..]);
        } else {
            println!("{}", line);
        }
    } else {
        println!("{}", line);
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
struct OperationsAndRegisters {
    ops: Vec<Operation>,
    registers: Registers,
}

fn generate_random_pauli_axes_for_arbitrary_angle_rotations(
    ops: &[Operation],
    precision: f64,
) -> Vec<(f64, Vec<Pauli>, Vec<Pauli>)> {
    use Pauli::*;
    let mut angle_map = Vec::<(f64, Vec<Pauli>, Vec<Pauli>)>::new();
    for op in ops {
        let angle = match op {
            Operation::PauliRotation(PauliRotation {
                angle: Angle::Arbitrary(angle),
                ..
            }) => angle,
            _ => {
                continue;
            }
        };
        if angle_map
            .iter()
            .any(|(a, _, _)| (angle - a).abs() < precision)
        {
            continue;
        }
        let mean = -1.55 * precision.log2() + 3.0;
        assert!(mean > 0.0);
        let stddev = mean / 16.0;
        let distribution = Normal::<f64>::new(mean, stddev).unwrap();
        // Let `pi_over_8_rotation_axes` be a sequence of random Paulis.
        let mut rng = rand::thread_rng();
        let len = distribution.sample(&mut rng).round() as usize;
        let mut pi_over_8_rotation_axes = vec![];
        while pi_over_8_rotation_axes.len() < len {
            let axis = *([X, Y, Z].choose(&mut rng).unwrap());
            if let Some(last) = pi_over_8_rotation_axes.last() {
                if *last == axis {
                    continue;
                }
            }
            pi_over_8_rotation_axes.push(axis);
        }
        let mut pi_over_4_rotation_axes = vec![];
        let first = *[I, X, Y, Z].choose(&mut rng).unwrap();
        if first != I {
            pi_over_4_rotation_axes.push(first);
        }
        let second = *[I, X, Y, Z].choose(&mut rng).unwrap();
        if second != I && second != first {
            pi_over_4_rotation_axes.push(second);
        }
        angle_map.push((*angle, pi_over_8_rotation_axes, pi_over_4_rotation_axes));
    }
    angle_map
}

fn translate_arbitrary_angle_rotations(
    ops: &[Operation],
    angle_map: &[(f64, Vec<Pauli>, Vec<Pauli>)],
    conf: &Configuration,
) -> Vec<Operation> {
    let mut new_ops = Vec::new();
    for op in ops {
        match op {
            Operation::PauliRotation(PauliRotation {
                axis,
                angle: Angle::Arbitrary(angle),
            }) => {
                if let Some((_, pi_over_8_rotation_axes, pi_over_4_rotation_axes)) =
                    angle_map.iter().find(|(a, _, _)| {
                        (*a - angle).abs() < conf.single_qubit_arbitrary_angle_rotation_precision
                    })
                {
                    assert_eq!(axis.iter().filter(|p| *p != &Pauli::I).count(), 1);
                    let target_position = axis.iter().position(|p| p != &Pauli::I).unwrap();

                    match axis[target_position] {
                        Pauli::I => unreachable!(),
                        //  S_Y * Z * (S_Y)dg = X
                        Pauli::X => new_ops.push(Operation::PauliRotation(PauliRotation {
                            axis: Axis::new_with_pauli(target_position, axis.len(), Pauli::Y),
                            angle: Angle::PiOver8(Mod8::Six),
                        })),
                        // (S_X)dg * Z * S_X = Y
                        Pauli::Y => new_ops.push(Operation::PauliRotation(PauliRotation {
                            axis: Axis::new_with_pauli(target_position, axis.len(), Pauli::X),
                            angle: Angle::PiOver8(Mod8::Two),
                        })),
                        Pauli::Z => {}
                    }

                    for pauli in pi_over_8_rotation_axes {
                        new_ops.push(Operation::PauliRotation(PauliRotation {
                            axis: Axis::new_with_pauli(target_position, axis.len(), *pauli),
                            angle: Angle::PiOver8(Mod8::One),
                        }));
                    }
                    for pauli in pi_over_4_rotation_axes {
                        new_ops.push(Operation::PauliRotation(PauliRotation {
                            axis: Axis::new_with_pauli(target_position, axis.len(), *pauli),
                            angle: Angle::PiOver8(Mod8::Two),
                        }));
                    }

                    match axis[target_position] {
                        Pauli::I => unreachable!(),
                        Pauli::X => new_ops.push(Operation::PauliRotation(PauliRotation {
                            axis: Axis::new_with_pauli(target_position, axis.len(), Pauli::Y),
                            angle: Angle::PiOver8(Mod8::Two),
                        })),
                        Pauli::Y => new_ops.push(Operation::PauliRotation(PauliRotation {
                            axis: Axis::new_with_pauli(target_position, axis.len(), Pauli::X),
                            angle: Angle::PiOver8(Mod8::Six),
                        })),
                        Pauli::Z => {}
                    }
                } else {
                    unreachable!();
                }
            }
            _ => new_ops.push(op.clone()),
        };
    }
    new_ops
}

fn schedule(board: &mut board::Board, ops: &[Operation], print_operations: bool) {
    let mut start = 0_usize;
    let mut layers = Vec::new();
    while start < ops.len() {
        let mut end = start + 1;
        while end < ops.len()
            && ops
                .iter()
                .take(end)
                .skip(start)
                .all(|op| op.axis().commutes_with(ops[end].axis()))
        {
            end += 1;
        }
        layers.push((start, end));
        start = end;
    }
    let mut schedule: Vec<(usize, Operation, u32)> = Vec::new();

    for (layer_index, (start, end)) in layers.iter().enumerate() {
        let mut indices = (0..end - start).collect::<Vec<_>>();
        if layer_index + 1 < layers.len() {
            let (next_start, next_end) = layers[layer_index + 1];
            assert_eq!(next_start, *end);
            let num_commuting_ops_in_successive_layer = (*start..*end)
                .map(|i| {
                    (next_start..next_end)
                        .filter(|&j| ops[i].axis().commutes_with(ops[j].axis()))
                        .count()
                })
                .collect::<Vec<_>>();

            indices.sort_by(|&i, &j| {
                let ci = num_commuting_ops_in_successive_layer[i];
                let cj = num_commuting_ops_in_successive_layer[j];
                ci.cmp(&cj)
            });
        }
        let indices = indices;
        let mut scheduled = vec![false; end - start];
        let cycle = ops[*start..*end]
            .iter()
            .map(|op| {
                let support_qubits = op.axis().iter().enumerate().filter_map(|(i, p)| {
                    if *p == Pauli::I {
                        None
                    } else {
                        Some(mapping::Qubit::new(i))
                    }
                });

                support_qubits
                    .map(|q| board.get_earliest_available_cycle_at(q))
                    .max()
                    .unwrap()
            })
            .min()
            .unwrap();
        board.set_cycle(cycle);

        while scheduled.iter().any(|&b| !b) {
            let mut scheduled_on_this_cycle = false;

            for &i in &indices {
                if scheduled[i] {
                    continue;
                }

                let op = &ops[start + i];
                if board.schedule(op) {
                    let cycle = board.cycle();
                    if print_operations {
                        let line =
                            format!("Schedule ops[{:3}] ({}) at cycle {}", start + i, op, cycle);
                        print_line_potentially_with_colors(&line);
                    }
                    scheduled[i] = true;
                    scheduled_on_this_cycle = true;
                    schedule.push((start + i, op.clone(), cycle));
                }
            }

            if !scheduled_on_this_cycle {
                board.increment_cycle();
            }
        }
    }
    // These are commented out because they are too slow.
    // Check the validity of the schedule.
    // schedule.sort_by_key(|&(index, _, _)| index);
    // for (i, &(index, _, _)) in schedule.iter().enumerate() {
    //     assert_eq!(i, index);
    // }
    // for (i, op, cycle) in &schedule {
    //     for (_, op2, cycle2) in schedule.iter().skip(*i + 1) {
    //         assert!(op.axis().commutes_with(op2.axis()) || *cycle < *cycle2);
    //     }
    // }
}

fn num_spc_cycles(
    ops: &[Operation],
    angle_map: &[(f64, Vec<Pauli>, Vec<Pauli>)],
    conf: &Configuration,
) -> u32 {
    use Mod8::*;
    let mut cycles = 0_u32;
    for op in ops {
        cycles += match op {
            Operation::PauliRotation(PauliRotation {
                angle: Angle::PiOver8(Zero) | Angle::PiOver8(Four),
                ..
            }) => 0,
            Operation::PauliRotation(PauliRotation {
                angle: Angle::PiOver8(Two) | Angle::PiOver8(Six),
                ..
            }) => conf.code_distance,
            Operation::PauliRotation(PauliRotation {
                angle: Angle::PiOver8(One) | Angle::PiOver8(Seven),
                ..
            }) => conf.code_distance,
            Operation::PauliRotation(PauliRotation {
                angle: Angle::PiOver8(Three) | Angle::PiOver8(Five),
                ..
            }) => unreachable!(),
            Operation::PauliRotation(PauliRotation {
                angle: Angle::Arbitrary(angle),
                ..
            }) => {
                let eps = conf.single_qubit_arbitrary_angle_rotation_precision;
                if let Some((_, pi_over_8_rotation_axes, _)) =
                    angle_map.iter().find(|(a, _, _)| (*a - angle).abs() < eps)
                {
                    pi_over_8_rotation_axes.len() as u32 * conf.code_distance
                } else {
                    panic!("self.arbitrary_angle_rotation_map.get(&{}) is None", angle);
                }
            }
            Operation::Measurement(_) => conf.code_distance,
        };
    }
    cycles
}

fn output_schedule(board: &board::Board, filename: &str) -> Result<(), std::io::Error> {
    #[derive(serde::Serialize)]
    struct ScheduleEntry {
        x: u32,
        y: u32,
        occupancy: BoardOccupancy,
    }

    #[derive(serde::Serialize)]
    struct Schedule {
        operations: Vec<OperationWithAdditionalData>,
        schedule: Vec<Vec<ScheduleEntry>>,
        width: u32,
        height: u32,
    }

    let end_cycle = board.get_last_end_cycle();
    let width = board.width();
    let height = board.height();
    let schedule = (0..end_cycle)
        .map(|cycle| {
            let mut schedule_on_this_cycle = vec![];
            for y in 0..height {
                for x in 0..width {
                    let occupancy = board.get_occupancy(x, y, cycle);
                    schedule_on_this_cycle.push(ScheduleEntry { x, y, occupancy });
                }
            }
            schedule_on_this_cycle
        })
        .collect::<Vec<_>>();

    let schedule = Schedule {
        operations: board.operations().to_vec(),
        schedule,
        width,
        height,
    };
    let serialized = serde_json::to_string(&schedule)?;
    std::fs::write(filename, serialized)?;

    Ok(())
}

fn run(
    board_without_blocks: board::Board,
    board_with_blocks: Option<board::Board>,
    num_executions: u32,
    parallelism: u32,
) {
    if num_executions == 0 {
        return;
    }

    let runner_for_board_without_blocks = runner::Runner::new(&board_without_blocks);
    let runner_for_board_with_blocks = board_with_blocks.map(|b| runner::Runner::new(&b));
    let num_total_executions = if runner_for_board_with_blocks.is_some() {
        num_executions * 2
    } else {
        num_executions
    };

    #[derive(Debug)]
    enum Id {
        WithoutBlocks,
        WithBlocks,
    }
    enum Command {
        Run(Box<runner::Runner>, Id),
        Stop,
    }

    let (sender_for_command, receiver_for_command) = mpsc::channel::<Command>();
    let (sender_for_reply, receiver_for_reply) = mpsc::channel::<(Id, u32, u32)>();
    let receiver_for_command = Arc::new(Mutex::new(receiver_for_command));
    let sender_for_reply = Arc::new(Mutex::new(sender_for_reply));
    let mut join_handles = (0..parallelism)
        .map(|_| {
            let receiver = Arc::clone(&receiver_for_command);
            let sender = Arc::clone(&sender_for_reply);
            std::thread::spawn(move || loop {
                let command = receiver.lock().unwrap().recv().unwrap();
                match command {
                    Command::Run(mut runner, id) => {
                        let delay = runner.run();
                        let runtime_cycle = runner.runtime_cycle();
                        println!(
                            "Run ({:?}): runtime cycle = {}, delay = {}",
                            id, runtime_cycle, delay
                        );
                        sender
                            .lock()
                            .unwrap()
                            .send((id, runtime_cycle, delay))
                            .unwrap();
                    }
                    Command::Stop => break,
                }
            })
        })
        .collect::<Vec<_>>();

    for _ in 0..num_executions {
        let runner = Box::new(runner_for_board_without_blocks.clone());
        sender_for_command
            .send(Command::Run(runner, Id::WithoutBlocks))
            .unwrap();
    }

    if let Some(runner) = runner_for_board_with_blocks {
        for _ in 0..num_executions {
            let runner = Box::new(runner.clone());
            sender_for_command
                .send(Command::Run(runner, Id::WithBlocks))
                .unwrap();
        }
    }

    for _ in 0..parallelism {
        sender_for_command.send(Command::Stop).unwrap();
    }
    while let Some(handle) = join_handles.pop() {
        handle.join().unwrap();
    }

    let mut results_without_blocks = Vec::new();
    let mut results_with_blocks = Vec::new();

    for (id, cycle, delay) in receiver_for_reply
        .iter()
        .take(num_total_executions as usize)
    {
        match id {
            Id::WithoutBlocks => {
                results_without_blocks.push((cycle, delay));
            }
            Id::WithBlocks => {
                results_with_blocks.push((cycle, delay));
            }
        }
    }

    assert!(!results_without_blocks.is_empty());
    let average_runtime_cycles = results_without_blocks.iter().map(|(c, _)| c).sum::<u32>() as f64
        / results_without_blocks.len() as f64;
    let average_delay = results_without_blocks.iter().map(|(_, d)| d).sum::<u32>() as f64
        / results_without_blocks.len() as f64;
    println!("Average runtime cycles[without blocks] = {}", average_runtime_cycles);
    println!("Average delay[without blocks] = {}", average_delay);

    if !results_with_blocks.is_empty() {
        let averaget_runtime_cycles = results_with_blocks.iter().map(|(c, _)| c).sum::<u32>()
            as f64
            / results_with_blocks.len() as f64;
        let average_delay = results_with_blocks.iter().map(|(_, d)| d).sum::<u32>() as f64
            / results_with_blocks.len() as f64;
        println!("Average runtime cycles[with blocks] = {}", averaget_runtime_cycles);
        println!("Average delay[with blocks] = {}", average_delay);
    }
}

fn main() {
    let args = Args::parse();
    let source = std::fs::read_to_string(args.filename.clone()).unwrap();
    let mapping_source = std::fs::read_to_string(&args.mapping_filename).unwrap();

    // let result = syntax_to_semantics::parse_source_string(
    //     source.clone(),
    //     Some(args.filename.as_str()),
    //     None::<&[PathBuf]>,
    // );
    // println!("result.any_errors = {:?}", result.any_errors());
    // result.print_errors();

    println!("Args:");
    println!("  filename = {}", args.filename);
    println!("  mapping filename = {}", args.mapping_filename);
    println!("  output source filename = {:?}", args.output_source_filename);
    println!("  file format = {:?}", args.file_format);
    println!("  schedule output filename = {:?}", args.schedule_output_filename);
    println!("  print_operations = {}", args.print_operations);
    println!("  use_pi_over_8_rotation_block = {}", args.use_pi_over_8_rotation_block);
    println!("  code_distance = {}", args.code_distance);
    println!("  magic_state_distillation_cost = {}", args.magic_state_distillation_cost);
    println!(
        "  magic_state_distillation_success_probability = {}",
        args.magic_state_distillation_success_probability
    );
    println!(
        "  num_distillations_for_pi_over_8_rotation = {}",
        args.num_distillations_for_pi_over_8_rotation
    );
    println!(
        "  num_distillations_for_pi_over_8_rotation_block = {}",
        args.num_distillations_for_pi_over_8_rotation_block
    );
    println!(
        "  single_qubit_pi_over_8_rotation_block_depth_ratio = {}",
        args.single_qubit_pi_over_8_rotation_block_depth_ratio
    );
    println!(
        "  single_qubit_arbitrary_angle_rotation_precision = {:e}",
        args.single_qubit_arbitrary_angle_rotation_precision
    );
    println!(
        "  preferable_distillation_area_size = {}",
        args.preferable_distillation_area_size
    );
    println!("  parallelism = {}", args.parallelism);
    println!("  num_executions = {}", args.num_executions);
    println!();

    if args.parallelism == 0 {
        eprintln!("Error: parallelism must be a positive integer.");
        return;
    }
    if args.use_pi_over_8_rotation_block && args.parallelism == 1 {
        eprintln!("Error: parallelism must be greater than 1 when --use-pi-over-8-rotation-block is specified.");
        return;
    }

    let file_format = args
        .file_format
        .unwrap_or("qasm".to_string())
        .to_ascii_lowercase();
    let (ops, registers) = if file_format == "qasm" {
        let cwd = env::current_dir().unwrap();
        print!("Parsing the QASM file...");
        std::io::stdout().flush().unwrap();
        let processed_source = qasm::process(&source, &cwd);
        let mut tokens = qasm::lex(&processed_source);
        let ast = match qasm::parse(&mut tokens) {
            Ok(ast) => ast,
            Err(e) => {
                eprintln!("Error: {}", e);
                return;
            }
        };
        println!("done.");
        let (ops, registers) = match extract(&ast) {
            Some((ops, registers)) => (ops, registers),
            None => {
                eprintln!("Error in extracting the AST");
                return;
            }
        };
        (ops, registers)
    } else if file_format == "json" {
        print!("Parsing the JSON file...");
        std::io::stdout().flush().unwrap();
        let deserialized: OperationsAndRegisters = serde_json::from_str(&source).unwrap();
        println!("done.");
        (deserialized.ops, deserialized.registers)
    } else {
        eprintln!("Error: unknown file format: {}", file_format);
        return;
    };

    if let Some(filename) = args.output_source_filename {
        let out = OperationsAndRegisters { ops, registers };
        let serialized = serde_json::to_string(&out).unwrap();
        std::fs::write(filename.clone(), serialized).unwrap();
        println!("Serialized the source to {}", filename);
        return;
    }

    let mapping = mapping::DataQubitMapping::new_from_json(&mapping_source).unwrap();
    let conf = Configuration {
        width: mapping.width,
        height: mapping.height,
        code_distance: args.code_distance,
        magic_state_distillation_cost: args.magic_state_distillation_cost,
        magic_state_distillation_success_rate: args.magic_state_distillation_success_probability,
        num_distillations_for_pi_over_8_rotation: args.num_distillations_for_pi_over_8_rotation,
        num_distillations_for_pi_over_8_rotation_block: args
            .num_distillations_for_pi_over_8_rotation_block,
        single_qubit_pi_over_8_rotation_block_depth_ratio: args
            .single_qubit_pi_over_8_rotation_block_depth_ratio,
        single_qubit_arbitrary_angle_rotation_precision: args
            .single_qubit_arbitrary_angle_rotation_precision,
        preferable_distillation_area_size: args.preferable_distillation_area_size,
    };

    let angle_map = generate_random_pauli_axes_for_arbitrary_angle_rotations(
        &ops,
        conf.single_qubit_arbitrary_angle_rotation_precision,
    );
    let ops_with_arbitrary_angle_rotations = lapbc::lapbc_translation(&ops);
    let spc_ops = pbc::spc_translation(&ops_with_arbitrary_angle_rotations);
    let ops_without_arbitrary_angle_rotations =
        translate_arbitrary_angle_rotations(&ops_with_arbitrary_angle_rotations, &angle_map, &conf);
    let ops_without_arbitrary_angle_rotations =
        lapbc::lapbc_translation(&ops_without_arbitrary_angle_rotations);

    println!("num lapbc ops = {}", ops_with_arbitrary_angle_rotations.len());
    println!(
        "spc_ops.len = {}, len * distance = {}, spc cycles = {}",
        spc_ops.len(),
        spc_ops.len() * conf.code_distance as usize,
        num_spc_cycles(&spc_ops, &angle_map, &conf)
    );

    let num_qubits_in_registers = registers.qregs.iter().map(|(_, size)| *size).sum::<u32>();
    let qubit_ids_in_mapping = mapping
        .iter()
        .map(|(_, _, qubit)| qubit.qubit as u32)
        .collect::<Vec<_>>();

    if (0..num_qubits_in_registers).any(|qubit_id| !qubit_ids_in_mapping.contains(&qubit_id)) {
        eprintln!("Error: qubit IDs in the mapping file are out of range");
        return;
    }

    let mut board_without_blocks = board::Board::new(mapping.clone(), &conf);
    let receiver_and_join_handle = if args.use_pi_over_8_rotation_block {
        assert!(args.parallelism > 1);
        let (sender, receiver) = mpsc::channel();
        let sender = Arc::new(Mutex::new(sender));
        let angle_map = angle_map.clone();
        let conf = conf.clone();
        let ops_with_arbitrary_angle_rotations = ops_with_arbitrary_angle_rotations.clone();
        let print_operations = args.print_operations;
        let handle = std::thread::spawn(move || {
            let mut board_with_blocks = board::Board::new(mapping.clone(), &conf);
            board_with_blocks.set_arbitrary_angle_rotation_map(angle_map.clone());
            schedule(&mut board_with_blocks, &ops_with_arbitrary_angle_rotations, print_operations);
            sender.lock().unwrap().send(board_with_blocks).unwrap();
        });
        Some((receiver, handle))
    } else {
        None
    };
    schedule(
        &mut board_without_blocks,
        &ops_without_arbitrary_angle_rotations,
        args.print_operations,
    );
    println!("num cycles (without blocks) = {}", board_without_blocks.get_last_end_cycle());

    let board_with_blocks = if args.use_pi_over_8_rotation_block {
        if let Some((receiver, handle)) = receiver_and_join_handle {
            let board_with_blocks = receiver.recv().unwrap();
            handle.join().unwrap();
            println!("num cycles (with blocks) = {}", board_with_blocks.get_last_end_cycle());
            Some(board_with_blocks)
        } else {
            unreachable!();
        }
    } else {
        None
    };

    println!("Scheduling is done.");

    if let Some(schedule_output_filename) = args.schedule_output_filename {
        let board = if let Some(board_with_blocks) = &board_with_blocks {
            println!("Outputting the schedule with blocks to {}", schedule_output_filename);
            board_with_blocks
        } else {
            println!("Outputting the schedule without blocks to {}", schedule_output_filename);
            &board_without_blocks
        };
        output_schedule(board, schedule_output_filename.as_str()).unwrap();
    }

    run(board_without_blocks, board_with_blocks, args.num_executions, args.parallelism);
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use board::OperationId;
    use qasm::Argument;

    fn new_axis(axis_string: &str) -> Axis {
        let v = axis_string
            .chars()
            .map(|c| match c {
                'I' => Pauli::I,
                'X' => Pauli::X,
                'Y' => Pauli::Y,
                'Z' => Pauli::Z,
                _ => unreachable!(),
            })
            .collect();
        Axis::new(v)
    }

    fn new_qregs(size: u32) -> Registers {
        let mut regs = Registers::new();
        regs.add_qreg("q".to_string(), size);
        regs
    }

    fn default_conf() -> Configuration {
        Configuration {
            width: 5,
            height: 5,
            code_distance: 5,
            magic_state_distillation_cost: 10,
            num_distillations_for_pi_over_8_rotation: 1,
            magic_state_distillation_success_rate: 0.5,
            num_distillations_for_pi_over_8_rotation_block: 1,
            single_qubit_pi_over_8_rotation_block_depth_ratio: 1.2,
            single_qubit_arbitrary_angle_rotation_precision: 1e-10,
            preferable_distillation_area_size: 5,
        }
    }

    #[test]
    fn test_extract_angle() {
        use Angle::*;
        use Mod8::*;
        assert_eq!(extract_angle("0", "test"), Ok(PiOver8(Zero)));
        assert_eq!(extract_angle("", "test"), Err("test: angle must not be empty".to_string()));
        assert_eq!(extract_angle(" pi ", "test"), Ok(PiOver8(Four)));
        assert_eq!(extract_angle(" pi  /  2 ", "test"), Ok(PiOver8(Two)));
        assert_eq!(extract_angle(" -  pi  /  2 ", "test"), Ok(-PiOver8(Two)));
        assert_eq!(extract_angle(" pi / 4 ", "test"), Ok(PiOver8(One)));
        assert_eq!(extract_angle(" 3 * pi / 4 ", "test"), Ok(PiOver8(Three)));
        assert_eq!(extract_angle(" - 3 * pi / 4 ", "test"), Ok(-PiOver8(Three)));
        assert_eq!(
            extract_angle(" pi / 8 ", "test"),
            Err("test: invalid angle:  pi / 8 ".to_string())
        );
        assert_eq!(extract_angle("-1.25", "test"), Ok(-Angle::Arbitrary(0.625)));
        assert_eq!(extract_angle("1.25", "test"), Ok(Angle::Arbitrary(0.625)));
    }

    #[test]
    fn test_translate_pauli() {
        use Angle::*;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let args = vec![Argument::Qubit("q".to_string(), 1)];
        let regs = new_qregs(4);
        let angle_args = Vec::new();
        {
            let mut ops = Vec::new();
            translate_gate("x", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IXII"), PiOver8(Four)))]);
        }

        {
            let mut ops = Vec::new();
            translate_gate("y", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IYII"), PiOver8(Four)))]);
        }

        {
            let mut ops = Vec::new();
            translate_gate("z", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IZII"), PiOver8(Four)))]);
        }
    }

    #[test]
    fn test_translate_ry() {
        use Angle::*;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let args = [Argument::Qubit("q".to_string(), 2)];
        let angle_args = [" 3 * pi / 4 ".to_string()];
        let regs = new_qregs(4);

        {
            let mut ops = Vec::new();
            translate_gate("ry", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IIYI"), PiOver8(Three))),]);
        }

        {
            let mut ops = Vec::new();
            let args = [];
            let r = translate_gate("ry", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for ry: 0".to_string()));
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            let r = translate_gate("ry", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for ry: 2".to_string()));
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["0".to_string()];
            assert!(translate_gate("ry", &args, &angle_args, &regs, &mut ops).is_ok());
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["pi".to_string()];
            assert!(translate_gate("ry", &args, &angle_args, &regs, &mut ops).is_ok());
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IIYI"), PiOver8(Four))),]);
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["- pi / 2".to_string()];
            assert!(translate_gate("ry", &args, &angle_args, &regs, &mut ops).is_ok());
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IIYI"), -PiOver8(Two))),]);
        }

        {
            let mut ops = Vec::new();
            let angle_args = [];
            assert_eq!(
                translate_gate("ry", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of angle arguments for ry: 0".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["0".to_string(), "0".to_string()];
            assert_eq!(
                translate_gate("ry", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of angle arguments for ry: 2".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [Argument::Qubit("q".to_string(), 4)];
            assert_eq!(
                translate_gate("ry", &args, &angle_args, &regs, &mut ops),
                Err("ry: there is no qubit q[4]".to_string())
            );
            assert!(ops.is_empty());
        }
    }

    #[test]
    fn test_translate_rz() {
        use Angle::*;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let args = [Argument::Qubit("q".to_string(), 2)];
        let angle_args = vec![" - 5 * pi / 4 ".to_string()];
        let regs = new_qregs(4);

        {
            let mut ops = Vec::new();
            assert!(translate_gate("rz", &args, &angle_args, &regs, &mut ops).is_ok());
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IIZI"), -PiOver8(Five))),]);
        }

        {
            let mut ops = Vec::new();
            let args = [];
            assert_eq!(
                translate_gate("rz", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of arguments for rz: 0".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            assert_eq!(
                translate_gate("rz", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of arguments for rz: 2".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["0".to_string()];
            assert!(translate_gate("rz", &args, &angle_args, &regs, &mut ops).is_ok());
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["pi".to_string()];
            assert!(translate_gate("rz", &args, &angle_args, &regs, &mut ops).is_ok());
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IIZI"), PiOver8(Four)))]);
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["- pi / 2".to_string()];
            assert!(translate_gate("rz", &args, &angle_args, &regs, &mut ops).is_ok());
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IIZI"), -PiOver8(Two)))]);
        }

        {
            let mut ops = Vec::new();
            let angle_args = [];
            assert_eq!(
                translate_gate("rz", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of angle arguments for rz: 0".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["0".to_string(), "0".to_string()];
            assert_eq!(
                translate_gate("rz", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of angle arguments for rz: 2".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [Argument::Qubit("q".to_string(), 4)];
            assert_eq!(
                translate_gate("rz", &args, &angle_args, &regs, &mut ops),
                Err("rz: there is no qubit q[4]".to_string())
            );
            assert!(ops.is_empty());
        }
    }

    #[test]
    fn test_translate_sx() {
        use Angle::*;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let args = vec![Argument::Qubit("q".to_string(), 1)];
        let regs = new_qregs(4);
        let angle_args = Vec::new();
        {
            let mut ops = Vec::new();
            translate_gate("sx", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops, vec![R(PauliRotation::new(new_axis("IXII"), PiOver8(Two)))]);
        }
        {
            let mut ops = Vec::new();
            let args = vec![];
            assert_eq!(
                translate_gate("sx", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of arguments for sx: 0".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            assert_eq!(
                translate_gate("sx", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of arguments for sx: 2".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = vec!["0".to_string()];
            assert_eq!(
                translate_gate("sx", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of angle arguments for sx: 1".to_string())
            );
            assert!(ops.is_empty());
        }
        {
            let mut ops = Vec::new();
            let args = vec![Argument::Qubit("q".to_string(), 4)];
            assert_eq!(
                translate_gate("sx", &args, &angle_args, &regs, &mut ops),
                Err("sx: there is no qubit q[4]".to_string())
            );
            assert!(ops.is_empty());
        }
    }

    #[test]
    fn test_translate_h() {
        use Angle::*;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let args = [Argument::Qubit("q".to_string(), 1)];
        let regs = new_qregs(4);
        let angle_args = Vec::new();
        {
            let mut ops = Vec::new();
            assert!(translate_gate("h", &args, &angle_args, &regs, &mut ops).is_ok());
            assert_eq!(
                ops,
                vec![
                    R(PauliRotation::new(new_axis("IZII"), PiOver8(Two))),
                    R(PauliRotation::new(new_axis("IXII"), PiOver8(Two))),
                    R(PauliRotation::new(new_axis("IZII"), PiOver8(Two)))
                ]
            );
        }
        {
            let mut ops = Vec::new();
            let args = vec![];
            let r = translate_gate("h", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for h: 0".to_string()));
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            let r = translate_gate("h", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for h: 2".to_string()));
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = vec!["0".to_string()];
            let r = translate_gate("h", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for h: 1".to_string()));
            assert!(ops.is_empty());
        }
        {
            let mut ops = Vec::new();
            let args = vec![Argument::Qubit("q".to_string(), 4)];
            let r = translate_gate("h", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("h: there is no qubit q[4]".to_string()));
            assert!(ops.is_empty());
        }
    }

    #[test]
    fn test_translate_cx() {
        use Angle::*;
        use Mod8::*;
        use Operation::PauliRotation as R;
        let args = [
            Argument::Qubit("q".to_string(), 1),
            Argument::Qubit("q".to_string(), 3),
        ];
        let regs = new_qregs(4);
        let angle_args = Vec::new();

        {
            let mut ops = Vec::new();
            assert!(translate_gate("cx", &args, &angle_args, &regs, &mut ops).is_ok());
            assert_eq!(
                ops,
                vec![
                    R(PauliRotation::new(new_axis("IZII"), -PiOver8(Two))),
                    R(PauliRotation::new(new_axis("IIIX"), -PiOver8(Two))),
                    R(PauliRotation::new(new_axis("IZIX"), PiOver8(Two))),
                ]
            );
        }

        {
            let mut ops = Vec::new();
            let args = vec![Argument::Qubit("q".to_string(), 1)];
            assert_eq!(
                translate_gate("cx", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of arguments for cx: 1".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
                Argument::Qubit("q".to_string(), 3),
            ];
            assert_eq!(
                translate_gate("cx", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of arguments for cx: 3".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let angle_args = ["0".to_string()];
            assert_eq!(
                translate_gate("cx", &args, &angle_args, &regs, &mut ops),
                Err("Invalid number of angle arguments for cx: 1".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [
                Argument::Qubit("q".to_string(), 4),
                Argument::Qubit("q".to_string(), 3),
            ];
            assert_eq!(
                translate_gate("cx", &args, &angle_args, &regs, &mut ops),
                Err("cx: there is no qubit q[4]".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 4),
            ];
            assert_eq!(
                translate_gate("cx", &args, &angle_args, &regs, &mut ops),
                Err("cx: there is no qubit q[4]".to_string())
            );
            assert!(ops.is_empty());
        }

        {
            let mut ops = Vec::new();
            let args = [
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 1),
            ];
            assert_eq!(
                translate_gate("cx", &args, &angle_args, &regs, &mut ops),
                Err("cx: control and target must be different".to_string())
            );
            assert!(ops.is_empty());
        }
    }

    #[test]
    fn test_translate_measurement() {
        use Operation::Measurement;
        let mut ops = Vec::new();
        let args = vec![
            Argument::Qubit("q".to_string(), 1),
            Argument::Qubit("c".to_string(), 1),
        ];
        let mut regs = new_qregs(4);
        regs.add_creg("c".to_string(), 4);
        let angle_args = Vec::new();
        {
            translate_gate("measure", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], Measurement(new_axis("IZII")));
        }

        {
            let angle_args = vec!["0".to_string()];
            let r = translate_gate("measure", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for measure: 1".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![Argument::Qubit("q".to_string(), 0)];
            let r = translate_gate("measure", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for measure: 1".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 0),
                Argument::Qubit("c".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            let r = translate_gate("measure", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for measure: 3".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 0),
                Argument::Qubit("q".to_string(), 1),
            ];
            let r = translate_gate("measure", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("measure: there is no classical bit q[1]".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("c".to_string(), 0),
                Argument::Qubit("c".to_string(), 1),
            ];
            let r = translate_gate("measure", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("measure: there is no qubit c[0]".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 4),
                Argument::Qubit("c".to_string(), 1),
            ];
            let r = translate_gate("measure", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("measure: there is no qubit q[4]".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 3),
                Argument::Qubit("c".to_string(), 4),
            ];
            let r = translate_gate("measure", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("measure: there is no classical bit c[4]".to_string()));
            assert_eq!(ops.len(), 1);
        }
    }

    #[test]
    fn test_translate_unrecognized_gate() {
        let mut rotations = Vec::new();
        let args = vec![Argument::Qubit("q".to_string(), 1)];
        let regs = new_qregs(4);
        let angle_args = Vec::new();
        let r = translate_gate("p", &args, &angle_args, &regs, &mut rotations);
        assert_eq!(r, Err("Unrecognized gate: p".to_string()));
        assert_eq!(rotations.len(), 0);
    }

    #[test]
    fn test_translate_arbitrary_angle_rotations_with_single_length_one_map_entry() {
        use Operation::PauliRotation as R;
        use Pauli::*;

        let map = [(0.4, vec![Z], vec![])];

        let ops = [
            R(PauliRotation {
                axis: new_axis("XIII"),
                angle: Angle::Arbitrary(0.4),
            }),
            R(PauliRotation {
                axis: new_axis("YIII"),
                angle: Angle::Arbitrary(0.4),
            }),
            R(PauliRotation {
                axis: new_axis("ZIII"),
                angle: Angle::Arbitrary(0.4),
            }),
        ];

        let conf = Configuration {
            single_qubit_arbitrary_angle_rotation_precision: 0.005,
            ..default_conf()
        };
        let new_ops = pbc::spc_translation(&translate_arbitrary_angle_rotations(&ops, &map, &conf));
        assert_eq!(
            new_ops,
            vec![
                R(PauliRotation {
                    axis: new_axis("XIII"),
                    angle: Angle::PiOver8(Mod8::One)
                }),
                R(PauliRotation {
                    axis: new_axis("YIII"),
                    angle: Angle::PiOver8(Mod8::One)
                }),
                R(PauliRotation {
                    axis: new_axis("ZIII"),
                    angle: Angle::PiOver8(Mod8::One)
                }),
            ]
        );
    }

    #[test]
    fn test_translate_arbitrary_angle_rotations() {
        use Mod8::*;
        use Operation::PauliRotation as R;
        use Pauli::*;

        let map = [
            (0.01, vec![X, Y, Z], vec![X, Y]),
            (0.02, vec![Y, Z], vec![X, Z]),
            (0.03, vec![Z], vec![X]),
            (0.04, vec![Y, X, Y], vec![]),
        ];
        let ops = [
            R(PauliRotation {
                axis: new_axis("IXII"),
                angle: Angle::Arbitrary(0.021),
            }),
            R(PauliRotation {
                axis: new_axis("IIZI"),
                angle: Angle::Arbitrary(0.018),
            }),
            R(PauliRotation {
                axis: new_axis("YIII"),
                angle: Angle::Arbitrary(0.042),
            }),
        ];
        let conf = Configuration {
            single_qubit_arbitrary_angle_rotation_precision: 0.005,
            ..default_conf()
        };

        let new_ops = translate_arbitrary_angle_rotations(&ops, &map, &conf);

        assert_eq!(
            new_ops,
            vec![
                // The first rotation.
                R(PauliRotation {
                    axis: new_axis("IYII"),
                    angle: Angle::PiOver8(Six),
                }),
                R(PauliRotation {
                    axis: new_axis("IYII"),
                    angle: Angle::PiOver8(One),
                }),
                R(PauliRotation {
                    axis: new_axis("IZII"),
                    angle: Angle::PiOver8(One),
                }),
                R(PauliRotation {
                    axis: new_axis("IXII"),
                    angle: Angle::PiOver8(Two),
                }),
                R(PauliRotation {
                    axis: new_axis("IZII"),
                    angle: Angle::PiOver8(Two),
                }),
                R(PauliRotation {
                    axis: new_axis("IYII"),
                    angle: Angle::PiOver8(Two),
                }),
                // The second rotation.
                R(PauliRotation {
                    axis: new_axis("IIYI"),
                    angle: Angle::PiOver8(One),
                }),
                R(PauliRotation {
                    axis: new_axis("IIZI"),
                    angle: Angle::PiOver8(One),
                }),
                R(PauliRotation {
                    axis: new_axis("IIXI"),
                    angle: Angle::PiOver8(Two),
                }),
                R(PauliRotation {
                    axis: new_axis("IIZI"),
                    angle: Angle::PiOver8(Two),
                }),
                // The third rotation.
                R(PauliRotation {
                    axis: new_axis("XIII"),
                    angle: Angle::PiOver8(Two),
                }),
                R(PauliRotation {
                    axis: new_axis("YIII"),
                    angle: Angle::PiOver8(One),
                }),
                R(PauliRotation {
                    axis: new_axis("XIII"),
                    angle: Angle::PiOver8(One),
                }),
                R(PauliRotation {
                    axis: new_axis("YIII"),
                    angle: Angle::PiOver8(One),
                }),
                R(PauliRotation {
                    axis: new_axis("XIII"),
                    angle: Angle::PiOver8(Six),
                }),
            ]
        );
    }

    #[test]
    fn test_schedule() {
        use board::Position;
        use mapping::Qubit;
        use Angle::PiOver8;
        use BoardOccupancy::*;
        use Mod8::*;
        use OperationWithAdditionalData::PiOver4Rotation;
        use OperationWithAdditionalData::PiOver8Rotation;
        use Pauli::*;
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        let q2 = Qubit::new(2);
        let id0 = OperationId::new(0);
        let id1 = OperationId::new(1);
        let id2 = OperationId::new(2);
        let id3 = OperationId::new(3);

        let mut mapping = mapping::DataQubitMapping::new(20, 20);
        mapping.map(q0, 3, 3);
        mapping.map(q1, 3, 16);
        mapping.map(q2, 16, 16);

        let conf = Configuration {
            width: mapping.width,
            height: mapping.height,
            code_distance: 5,
            magic_state_distillation_cost: 10,
            num_distillations_for_pi_over_8_rotation: 1,
            magic_state_distillation_success_rate: 1.0,

            ..default_conf()
        };

        let mut board = board::Board::new(mapping, &conf);
        let ops = [
            Operation::PauliRotation(PauliRotation {
                axis: new_axis("ZII"),
                angle: PiOver8(One),
            }),
            Operation::PauliRotation(PauliRotation {
                axis: new_axis("XXI"),
                angle: PiOver8(Two),
            }),
            Operation::PauliRotation(PauliRotation {
                axis: new_axis("IZI"),
                angle: PiOver8(One),
            }),
            Operation::PauliRotation(PauliRotation {
                axis: new_axis("IIZ"),
                angle: PiOver8(One),
            }),
        ];

        schedule(&mut board, &ops, false);

        assert!(board.is_occupancy(3, 3, 0..10, IdleDataQubit));
        assert!(board.is_occupancy(3, 3, 10..15, DataQubitInOperation(id0)));
        assert!(board.is_occupancy(3, 3, 15..20, DataQubitInOperation(id1)));

        assert!(board.is_occupancy(3, 16, 0..15, IdleDataQubit));
        assert!(board.is_occupancy(3, 16, 15..20, DataQubitInOperation(id1)));
        assert!(board.is_occupancy(3, 16, 20..25, DataQubitInOperation(id3)));

        assert!(board.is_occupancy(16, 16, 0..10, IdleDataQubit));
        assert!(board.is_occupancy(16, 16, 10..15, DataQubitInOperation(id2)));

        assert_eq!(board.operations().len(), 4);
        assert!(matches!(&board.operations()[0], PiOver8Rotation{
            id,
            targets,
            ..
        } if *id == id0 && *targets == vec![(Position::new(3, 3), Z)]));
        assert!(matches!(&board.operations()[1], PiOver4Rotation{
            id,
            targets,
            ..
        } if *id == id1 && *targets == vec![(Position::new(3, 3), X), (Position::new(3, 16), X)]));
        assert!(matches!(&board.operations()[2], PiOver8Rotation{
            id,
            targets,
            ..
        } if *id == id2 && *targets == vec![(Position::new(16, 16), Z)]));
        assert!(matches!(&board.operations()[3], PiOver8Rotation{
            id,
            targets,
            ..
        } if *id == id3 && *targets == vec![(Position::new(3, 16), Z)]));
    }
}
