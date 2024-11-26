extern crate clap;
// extern crate oq3_lexer;
// extern crate oq3_parser;
// extern crate oq3_source_file;
extern crate qasm;

use board::BoardOccupancy;
use board::Configuration;
use clap::Parser;
use lapbc::LapbcCompactOperation;
use pbc::Operation;
use std::env;
use std::fmt::Write;
use std::io::IsTerminal;

mod board;
mod lapbc;
mod mapping;
mod pbc;
mod runner;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The filename of the QASM file to be translated.
    #[arg(short, long)]
    filename: String,

    #[arg(short, long)]
    mapping_filename: String,

    #[arg(short, long)]
    schedule_output_filename: Option<String>,
}

use pbc::Angle;
use pbc::Axis;
use pbc::Pauli;
use pbc::PauliRotation;

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
// so extract_angle(" pi / 2 ") returns Ok(Angle::PiOver4), for instance.
//
// Note also that we ignore the sign of the angle as long as it is a multiple of pi/8.
// This is due to the following reasons:
//   1. 0 == -0.
//   2. pi == -pi mod 2pi.
//   3. pi/2 and -pi/2 are equivalent, given Pauli operations are self-inverse.
//   4. pi/4 and -pi/4 are equivalent, because P(pi/4) = P(pi/2) * P(-pi/4) = PP(-pi/4) where P is
//      a Pauli operation. Pauli operations can be applied in the controlling classical computers
//      with the feed-forward mechanism.
//   5. Given that we use a magic state for a pi/8 rotation, a pi/8 rotation turns to a -pi/8
//      rotation with a 1/2 probability. In that sense, pi/8 and -pi/8 are equivalent.
fn extract_angle(s: &str, context: &str) -> Result<Angle, String> {
    let pattern = regex::Regex::new(r" pi */ *([0-9]+) *$").unwrap();
    let minus_pattern = regex::Regex::new(r"^ * - *([0-9]+\.[0-9]+) *$").unwrap();
    if s.trim() == "" {
        Err(format!("{}: angle must not be empty", context))
    } else if s.trim() == "0" {
        Ok(Angle::Zero)
    } else if s.trim() == "pi" || s.trim() == "-  pi" {
        // This apparent inconsistency is intentional: see above.
        Ok(Angle::PiOver2)
    } else if let Some(caps) = pattern.captures(s) {
        if caps.get(1).unwrap().as_str() == "2" {
            Ok(Angle::PiOver4)
        } else if caps.get(1).unwrap().as_str() == "4" {
            Ok(Angle::PiOver8)
        } else {
            Err(format!("{}: invalid angle: {}", context, s))
        }
    } else if let Some(caps) = minus_pattern.captures(s) {
        let angle = caps.get(1).unwrap().as_str().parse::<f64>();
        if let Ok(angle) = angle {
            Ok(Angle::Arbitrary(-angle))
        } else {
            Err(format!("{}: invalid angle: {}", context, s))
        }
    } else {
        let angle = s.trim().parse::<f64>();
        if let Ok(angle) = angle {
            Ok(Angle::Arbitrary(angle))
        } else {
            Err(format!("{}: invalid angle: {}", context, s))
        }
    }
}

fn translate_gate(
    name: &str,
    args: &[qasm::Argument],
    angle_args: &[String],
    registers: &Registers,
    output: &mut Vec<pbc::Operation>,
) -> Result<(), String> {
    use pbc::Operation::Measurement as M;
    use pbc::Operation::PauliRotation as R;
    let num_qubits = registers.num_qubits();
    match name {
        "x" | "y" | "z" => {
            // Pauli operations are dealt with classically.
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
            if angle != Angle::Zero && angle != Angle::PiOver2 {
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
            if angle != Angle::Zero && angle != Angle::PiOver2 {
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
            output.push(R(PauliRotation::new_clifford(axis)));
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
            let qubit = extract_qubit(args, 0, registers, "sx")?;
            let mut axis_x = vec![Pauli::I; num_qubits as usize];
            axis_x[qubit as usize] = Pauli::X;
            let mut axis_z = vec![Pauli::I; num_qubits as usize];
            axis_z[qubit as usize] = Pauli::X;
            output.push(R(PauliRotation::new_clifford(Axis::new(axis_z.clone()))));
            output.push(R(PauliRotation::new_clifford(Axis::new(axis_x))));
            output.push(R(PauliRotation::new_clifford(Axis::new(axis_z))));
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
            let mut axis = vec![Pauli::I; num_qubits as usize];
            axis[control as usize] = Pauli::Z;
            output.push(R(PauliRotation::new_clifford(Axis::new(axis))));

            let mut axis = vec![Pauli::I; num_qubits as usize];
            axis[target as usize] = Pauli::X;
            output.push(R(PauliRotation::new_clifford(Axis::new(axis))));

            let mut axis = vec![Pauli::I; num_qubits as usize];
            axis[control as usize] = Pauli::Z;
            axis[target as usize] = Pauli::X;
            output.push(R(PauliRotation::new_clifford(Axis::new(axis))));
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

#[allow(dead_code)]
fn extract_and_print(nodes: &[qasm::AstNode]) -> Option<(Vec<PauliRotation>, Registers)> {
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
        println!("Unrecognized node in the AST");
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
    println!("num ops = {}", ops.len());
    println!(
        "num single clifford ops = {}",
        ops.iter().filter(|r| r.is_single_qubit_clifford()).count()
    );
    println!(
        "num non-clifford rotations and measurements = {}",
        ops.iter()
            .filter(|r| r.is_non_clifford_rotation_or_measurement())
            .count()
    );
    println!(
        "num multi qubit clifford ops = {}",
        ops.iter().filter(|r| r.is_multi_qubit_clifford()).count()
    );

    let result = pbc::spc_translation(&ops);
    let cliffords = ops
        .iter()
        .filter_map(|op| match op {
            pbc::Operation::PauliRotation(r) => {
                if r.is_clifford() {
                    Some(r.clone())
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let num_qubits = match &result[0] {
        pbc::Operation::PauliRotation(r) => r.axis.len(),
        pbc::Operation::Measurement(a) => a.len(),
    };
    // Print logical operations.
    for i in 0..num_qubits {
        let mut a = Axis::new(vec![Pauli::I; num_qubits]);
        a[i] = Pauli::X;
        for c in cliffords.iter().rev() {
            a.transform(&c.axis);
        }
        let line = format!("X{:0>3} => {}", i, a);
        print_line_potentially_with_colors(&line);

        let mut a = Axis::new(vec![Pauli::I; num_qubits]);
        a[i] = Pauli::Z;
        for c in cliffords.iter().rev() {
            a.transform(&c.axis);
        }
        let line = format!("Z{:0>3} => {}", i, a);
        print_line_potentially_with_colors(&line);
    }

    println!();
    // Print SPC operations.
    for (i, op) in result.iter().enumerate() {
        let mut out = String::new();
        write!(&mut out, "{:>4} {:}", i, op).unwrap();
        print_line_potentially_with_colors(&out);
    }
    println!();

    // Print SPC compact operations.
    println!("SPC compact");
    let mut spc_compact_clocks = 0_u32;
    let compact_result = pbc::spc_compact_translation(&ops);
    for (i, (op, clocks)) in compact_result.iter().enumerate() {
        let mut out = String::new();
        write!(&mut out, "{:>4} {:} (+{})", i, op, clocks).unwrap();
        print_line_potentially_with_colors(&out);
        spc_compact_clocks += clocks + 1;
    }
    println!("SPC compact clocks = {}", spc_compact_clocks);

    println!("lapbc compact");
    let mut lapbc_compact_clocks = 0_u32;
    let mut lapbc_compact_num_spc_ops = 0_u32;
    let mut lapbc_compact_axis_permutations = 0_u32;

    let lapbc_result = lapbc::lapbc_compact_translation(&ops);
    for (i, op) in lapbc_result.iter().enumerate() {
        lapbc_compact_clocks += op.clocks();
        match op {
            LapbcCompactOperation::Operation(_) => {
                lapbc_compact_num_spc_ops += 1;
            }
            LapbcCompactOperation::AxisPermutation(_) => {
                lapbc_compact_axis_permutations += 1;
            }
            LapbcCompactOperation::Noop => (),
        }
        let mut out = String::new();
        write!(&mut out, "{:>4} {:}", i, op).unwrap();
        print_line_potentially_with_colors(&out);
    }
    println!("lapbc compact clocks: {}", lapbc_compact_clocks);
    println!(
        "SPC ops: {}, axis permutations: {}",
        lapbc_compact_num_spc_ops, lapbc_compact_axis_permutations
    );

    None
}

fn main() {
    let args = Args::parse();
    // Load the QASM file.
    let source = std::fs::read_to_string(args.filename.clone()).unwrap();
    let mapping_source = std::fs::read_to_string(&args.mapping_filename).unwrap();

    // let result = syntax_to_semantics::parse_source_string(
    //     source.clone(),
    //     Some(args.filename.as_str()),
    //     None::<&[PathBuf]>,
    // );
    // println!("result.any_errors = {:?}", result.any_errors());
    // result.print_errors();

    let cwd = env::current_dir().unwrap();
    let processed_source = qasm::process(&source, &cwd);
    let mut tokens = qasm::lex(&processed_source);
    let ast = qasm::parse(&mut tokens);
    let ast = match ast {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    };

    let (ops, registers) = match extract(&ast) {
        Some((ops, registers)) => (ops, registers),
        None => {
            eprintln!("Error in extracting the AST");
            return;
        }
    };

    let ops = lapbc::lapbc_translation(&ops);
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

    let mapping = mapping::DataQubitMapping::new_from_json(&mapping_source).unwrap();
    let num_qubits_in_registers = registers.qregs.iter().map(|(_, size)| *size).sum::<u32>();
    let qubit_ids_in_mapping = mapping
        .iter()
        .map(|(_, _, qubit)| qubit.qubit as u32)
        .collect::<Vec<_>>();

    if (0..num_qubits_in_registers).any(|qubit_id| !qubit_ids_in_mapping.contains(&qubit_id)) {
        eprintln!("Error: qubit IDs in the mapping file are out of range");
        return;
    }

    let conf = Configuration {
        width: mapping.width,
        height: mapping.height,
        code_distance: 15,
        magic_state_distillation_cost: 21,
        magic_state_distillation_success_rate: 0.5,
        num_distillations_for_pi_over_8_rotation: 6,
    };

    let mut board = board::Board::new(mapping, &conf);
    board.set_preferable_distillation_area_size(5);
    let mut schedule: Vec<(usize, Operation, u32)> = Vec::new();

    for (layer_index, (start, end)) in layers.iter().enumerate() {
        println!();
        let mut scheduled = vec![false; end - start];

        board.set_cycle(std::cmp::max(board.cycle(), 200) - 200);

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
        println!("indices = {:?} (+{})", indices, start);
        let indices = indices;

        while scheduled.iter().any(|&b| !b) {
            let mut scheduled_on_this_cycle = false;

            for &i in &indices {
                if scheduled[i] {
                    continue;
                }

                let op = &ops[start + i];
                if board.schedule(op) {
                    let cycle = board.cycle();
                    let line = format!("Schedule ops[{:3}] ({}) at cycle {}", start + i, op, cycle);
                    print_line_potentially_with_colors(&line);
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

    // Check the validity of the schedule.
    schedule.sort_by_key(|&(index, _, _)| index);
    for (i, &(index, _, _)) in schedule.iter().enumerate() {
        assert_eq!(i, index);
    }
    for (i, op, cycle) in &schedule {
        for (_, op2, cycle2) in schedule.iter().skip(*i + 1) {
            assert!(op.axis().commutes_with(op2.axis()) || *cycle < *cycle2);
        }
    }

    println!("num cycles = {}", board.get_last_end_cycle());

    let spc_ops = pbc::spc_translation(&ops);
    println!(
        "spc_ops.len = {}, cycles = {}",
        spc_ops.len(),
        spc_ops.len() * conf.code_distance as usize
    );

    let n = 50;
    let mut average_delay = 0.0;
    for _ in 0..n {
        let runner = runner::Runner::new(&board);
        let delay = runner.run();
        average_delay += (delay as f64) / n as f64;
    }
    println!("delay = {:.2}", average_delay);

    if let Some(schedule_output_filename) = args.schedule_output_filename {
        #[derive(serde::Serialize)]
        struct ScheduleEntry {
            x: u32,
            y: u32,
            occupancy: BoardOccupancy,
        }

        #[derive(serde::Serialize)]
        struct Schedule {
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
            schedule,
            width,
            height,
        };
        let serialized = serde_json::to_string(&schedule).unwrap();
        std::fs::write(schedule_output_filename, serialized).unwrap();
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn test_extract_angle() {
        assert_eq!(extract_angle("0", "test"), Ok(Angle::Zero));
        assert_eq!(extract_angle("", "test"), Err("test: angle must not be empty".to_string()));
        assert_eq!(extract_angle(" pi ", "test"), Ok(Angle::PiOver2));
        // We expect a space before "pi".
        assert_eq!(
            extract_angle("pi / 2 ", "test"),
            Err("test: invalid angle: pi / 2 ".to_string())
        );
        assert_eq!(extract_angle(" pi / 2 ", "test"), Ok(Angle::PiOver4));
        assert_eq!(extract_angle("- pi / 2 ", "test"), Ok(Angle::PiOver4));
        assert_eq!(extract_angle(" pi / 4 ", "test"), Ok(Angle::PiOver8));
        assert_eq!(
            extract_angle(" pi / 8 ", "test"),
            Err("test: invalid angle:  pi / 8 ".to_string())
        );
        assert_eq!(extract_angle("-1.25", "test"), Ok(Angle::Arbitrary(-1.25)));
    }

    #[test]
    fn test_translate_pauli() {
        let mut ops = Vec::new();
        let args = vec![Argument::Qubit("q".to_string(), 1)];
        let regs = new_qregs(4);
        let angle_args = Vec::new();
        {
            translate_gate("x", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 0);
        }

        {
            translate_gate("y", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 0);
        }

        {
            translate_gate("z", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 0);
        }
    }

    #[test]
    fn test_translate_ry() {
        use pbc::Operation::PauliRotation as R;
        let mut ops = Vec::new();
        let args = vec![Argument::Qubit("q".to_string(), 2)];
        let angle_args = vec![" 3 * pi / 4 ".to_string()];
        let regs = new_qregs(4);

        {
            translate_gate("ry", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], R(PauliRotation::new_pi_over_8(new_axis("IIYI"))));
        }

        {
            let args = vec![];
            let r = translate_gate("ry", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for ry: 0".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            let r = translate_gate("ry", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for ry: 2".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let angle_args = vec!["0".to_string()];
            translate_gate("ry", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], R(PauliRotation::new_pi_over_8(new_axis("IIYI"))));
        }

        {
            let angle_args = vec!["pi".to_string()];
            translate_gate("ry", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], R(PauliRotation::new_pi_over_8(new_axis("IIYI"))));
        }

        {
            let angle_args = vec!["- pi / 2".to_string()];
            translate_gate("ry", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 2);
            assert_eq!(ops[1], R(PauliRotation::new_clifford(new_axis("IIYI"))));
        }

        {
            let angle_args = vec![];
            let r = translate_gate("ry", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for ry: 0".to_string()));
            assert_eq!(ops.len(), 2);
        }

        {
            let angle_args = vec!["0".to_string(), "0".to_string()];
            let r = translate_gate("ry", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for ry: 2".to_string()));
            assert_eq!(ops.len(), 2);
        }
        {
            let args = vec![Argument::Qubit("q".to_string(), 4)];
            let r = translate_gate("ry", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("ry: there is no qubit q[4]".to_string()));
            assert_eq!(ops.len(), 2);
        }
    }

    #[test]
    fn test_translate_rz() {
        use pbc::Operation::PauliRotation as R;
        let mut ops = Vec::new();
        let args = vec![Argument::Qubit("q".to_string(), 2)];
        let angle_args = vec![" 3 * pi / 4 ".to_string()];
        let regs = new_qregs(4);

        {
            translate_gate("rz", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], R(PauliRotation::new_pi_over_8(new_axis("IIZI"))));
        }

        {
            let args = vec![];
            let r = translate_gate("rz", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for rz: 0".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            let r = translate_gate("rz", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for rz: 2".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let angle_args = vec!["0".to_string()];
            translate_gate("rz", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], R(PauliRotation::new_pi_over_8(new_axis("IIZI"))));
        }

        {
            let angle_args = vec!["pi".to_string()];
            translate_gate("rz", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], R(PauliRotation::new_pi_over_8(new_axis("IIZI"))));
        }

        {
            let angle_args = vec!["- pi / 2".to_string()];
            translate_gate("rz", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 2);
            assert_eq!(ops[1], R(PauliRotation::new_clifford(new_axis("IIZI"))));
        }

        {
            let angle_args = vec![];
            let r = translate_gate("rz", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for rz: 0".to_string()));
            assert_eq!(ops.len(), 2);
        }

        {
            let angle_args = vec!["0".to_string(), "0".to_string()];
            let r = translate_gate("rz", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for rz: 2".to_string()));
            assert_eq!(ops.len(), 2);
        }
        {
            let args = vec![Argument::Qubit("q".to_string(), 4)];
            let r = translate_gate("rz", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("rz: there is no qubit q[4]".to_string()));
            assert_eq!(ops.len(), 2);
        }
    }

    #[test]
    fn test_translate_sx() {
        use pbc::Operation::PauliRotation as R;
        let mut ops = Vec::new();
        let args = vec![Argument::Qubit("q".to_string(), 1)];
        let regs = new_qregs(4);
        let angle_args = Vec::new();
        {
            translate_gate("sx", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0], R(PauliRotation::new_clifford(new_axis("IXII"))));
        }
        {
            let args = vec![];
            let r = translate_gate("sx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for sx: 0".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
            ];
            let r = translate_gate("sx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for sx: 2".to_string()));
            assert_eq!(ops.len(), 1);
        }

        {
            let angle_args = vec!["0".to_string()];
            let r = translate_gate("sx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for sx: 1".to_string()));
            assert_eq!(ops.len(), 1);
        }
        {
            let args = vec![Argument::Qubit("q".to_string(), 4)];
            let r = translate_gate("sx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("sx: there is no qubit q[4]".to_string()));
            assert_eq!(ops.len(), 1);
        }
    }

    #[test]
    fn test_translate_measurement() {
        use pbc::Operation::Measurement;
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
    fn test_translate_cx() {
        use pbc::Operation::PauliRotation as R;
        let mut ops = Vec::new();
        let args = vec![
            Argument::Qubit("q".to_string(), 1),
            Argument::Qubit("q".to_string(), 3),
        ];
        let regs = new_qregs(4);
        let angle_args = Vec::new();

        {
            translate_gate("cx", &args, &angle_args, &regs, &mut ops).unwrap();
            assert_eq!(ops.len(), 3);
            assert_eq!(ops[0], R(PauliRotation::new_clifford(new_axis("IZII"))));
            assert_eq!(ops[1], R(PauliRotation::new_clifford(new_axis("IIIX"))));
            assert_eq!(ops[2], R(PauliRotation::new_clifford(new_axis("IZIX"))));
        }

        {
            let args = vec![Argument::Qubit("q".to_string(), 1)];
            let r = translate_gate("cx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for cx: 1".to_string()));

            assert_eq!(ops.len(), 3);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 2),
                Argument::Qubit("q".to_string(), 3),
            ];
            let r = translate_gate("cx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of arguments for cx: 3".to_string()));

            assert_eq!(ops.len(), 3);
        }

        {
            let angle_args = vec!["0".to_string()];
            let r = translate_gate("cx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("Invalid number of angle arguments for cx: 1".to_string()));
            assert_eq!(ops.len(), 3);
        }
        {
            let args = vec![
                Argument::Qubit("q".to_string(), 4),
                Argument::Qubit("q".to_string(), 3),
            ];
            let r = translate_gate("cx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("cx: there is no qubit q[4]".to_string()));
            assert_eq!(ops.len(), 3);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 4),
            ];
            let r = translate_gate("cx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("cx: there is no qubit q[4]".to_string()));
            assert_eq!(ops.len(), 3);
        }

        {
            let args = vec![
                Argument::Qubit("q".to_string(), 1),
                Argument::Qubit("q".to_string(), 1),
            ];
            let r = translate_gate("cx", &args, &angle_args, &regs, &mut ops);
            assert_eq!(r, Err("cx: control and target must be different".to_string()));
            assert_eq!(ops.len(), 3);
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
}
