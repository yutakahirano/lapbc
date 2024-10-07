#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Qubit {
    pub qubit: usize,
}

#[derive(Debug)]
pub struct Mapping {
    pub width: u32,
    pub height: u32,

    mapping: Vec<(u32, u32, Qubit)>,
}

impl Qubit {
    pub fn new(qubit: usize) -> Qubit {
        Qubit { qubit }
    }
}

impl Mapping {
    pub fn new(width: u32, height: u32) -> Mapping {
        Mapping {
            width,
            height,
            mapping: Vec::new(),
        }
    }

    pub fn new_from_json(json: &str) -> Result<Mapping, String> {
        let json: serde_json::Value = match serde_json::from_str(json) {
            Ok(json) => json,
            Err(e) => {
                return Err(format!("failed to parse json: {}", e));
            }
        };

        let width = json.get("width").ok_or("width is not found")?;
        let width: u32 = if let Some(width) = width.as_i64() {
            if width <= 0 {
                return Err(format!("width should be positive, but got {}", width));
            } else if width > u32::MAX as i64 {
                return Err("width is too large".to_string());
            }
            width as u32
        } else {
            return Err(format!("width should be an integer, but got {}", width))?;
        };
        let height = json.get("height").ok_or("height is not found")?;
        let height: u32 = if let Some(height) = height.as_i64() {
            if height <= 0 {
                return Err(format!("height should be positive, but got {}", height));
            } else if height > u32::MAX as i64 {
                return Err("height is too large".to_string());
            }
            height as u32
        } else {
            return Err(format!("height should be an integer, but got {}", height));
        };

        let mut result = Mapping::new(width, height);

        let mapping_list = if let Some(value) = json.get("mapping") {
            if let Some(mapping_list) = value.as_array() {
                mapping_list
            } else {
                return Err(format!("mapping should be a list, but got {}", value));
            }
        } else {
            return Ok(result);
        };

        for entry in mapping_list {
            if !entry.is_object() {
                return Err(format!("each mapping entry should be an object, but got {}", entry));
            }
            let x = entry
                .get("x")
                .ok_or(format!("x is not found in mapping entry {}", entry))?;
            let x: u32 = if let Some(x) = x.as_i64() {
                if x < 0 || x >= width as i64 {
                    return Err(format!("x({}) is out of bounds", x));
                }
                x as u32
            } else {
                return Err(format!("x should be an integer, but got {}", x));
            };
            let y = entry
                .get("y")
                .ok_or(format!("y is not found in mapping entry {}", entry))?;
            let y: u32 = if let Some(y) = y.as_i64() {
                if y < 0 || y >= height as i64 {
                    return Err(format!("y({}) is out of bounds", y));
                }
                y as u32
            } else {
                return Err(format!("y should be an integer, but got {}", y));
            };
            let qubit = entry
                .get("qubit")
                .ok_or(format!("qubit is not found in mapping entry {}", entry))?;
            let qubit = match qubit.as_u64() {
                Some(qubit) => Qubit::new(qubit as usize),
                None => {
                    return Err(format!("qubit should be an integer, but got {}", qubit));
                }
            };

            if result.is_assigned(qubit) {
                return Err(format!("qubit {} is already assigned", qubit.qubit));
            }
            if result.is_occupied(x, y) {
                return Err(format!("({}, {}) is already occupied", x, y));
            }
            result.map(qubit, x, y);
        }

        Ok(result)
    }

    pub fn is_assigned(&self, qubit: Qubit) -> bool {
        self.mapping.iter().any(|(_, _, q)| *q == qubit)
    }

    pub fn is_occupied(&self, x: u32, y: u32) -> bool {
        assert!(x < self.width);
        assert!(y < self.height);

        self.mapping.iter().any(|(x1, y1, _)| *x1 == x && *y1 == y)
    }

    pub fn get(&self, qubit: Qubit) -> Option<(u32, u32)> {
        self.mapping
            .iter()
            .find_map(|(x, y, q)| if *q == qubit { Some((*x, *y)) } else { None })
    }

    pub fn map(&mut self, qubit: Qubit, x: u32, y: u32) {
        assert!(x < self.width);
        assert!(y < self.height);

        assert!(!self.is_assigned(qubit));
        assert!(!self.is_occupied(x, y));

        self.mapping.push((x, y, qubit));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapping_init() {
        let mapping = Mapping::new(2, 3);
        assert_eq!(mapping.width, 2);
        assert_eq!(mapping.height, 3);
        assert_eq!(mapping.mapping.len(), 0);
        for x in 0..2 {
            for y in 0..3 {
                assert!(!mapping.is_occupied(x, y));
            }
        }
        assert!(!mapping.is_assigned(Qubit::new(4)));
        assert!(mapping.get(Qubit::new(4)).is_none());
    }

    #[test]
    fn test_mapping_add() {
        let mut mapping = Mapping::new(2, 3);

        mapping.map(Qubit::new(3), 1, 2);

        for x in 0..2 {
            for y in 0..3 {
                let expectation = x == 1 && y == 2;
                assert_eq!(mapping.is_occupied(x, y), expectation);
            }
        }
        assert!(!mapping.is_assigned(Qubit::new(0)));
        assert!(!mapping.is_assigned(Qubit::new(1)));
        assert!(!mapping.is_assigned(Qubit::new(2)));
        assert!(mapping.is_assigned(Qubit::new(3)));
        assert!(!mapping.is_assigned(Qubit::new(4)));

        assert!(mapping.get(Qubit::new(0)).is_none());
        assert!(mapping.get(Qubit::new(1)).is_none());
        assert!(mapping.get(Qubit::new(2)).is_none());
        assert_eq!(mapping.get(Qubit::new(3)), Some((1, 2)));
        assert!(mapping.get(Qubit::new(4)).is_none());
    }

    #[test]
    fn test_mapping_new_from_json_invalid_syntax() {
        let json = r#"{"width": 3, "height": 4, mapping: [], hogefuga}"#;
        let result = Mapping::new_from_json(json);
        let expectation = "failed to parse json: key must be a string at line 1 column 27";
        assert!(matches!(result, Err(s) if s == *expectation));
    }

    #[test]
    fn test_mapping_new_from_json_width_is_missing() {
        let json = r#"{"height": 3, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"width is not found"));
    }

    #[test]
    fn test_mapping_new_from_json_height_is_missing() {
        let json = r#"{"width": 3, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"height is not found"));
    }

    #[test]
    fn test_mapping_new_from_json_width_is_not_a_number() {
        let json = r#"{"width": "3", "height": 3, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"width should be an integer, but got \"3\""));
    }

    #[test]
    fn test_mapping_new_from_json_width_is_negative() {
        let json = r#"{"width": -1, "height": 3, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"width should be positive, but got -1"));
    }

    #[test]
    fn test_mapping_new_from_json_width_is_zero() {
        let json = r#"{"width": 0, "height": 3, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"width should be positive, but got 0"));
    }

    #[test]
    fn test_mapping_new_from_json_height_is_not_a_number() {
        let json = r#"{"width": 1, "height": "4", "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"height should be an integer, but got \"4\""));
    }

    #[test]
    fn test_mapping_new_from_json_height_is_negative() {
        let json = r#"{"width": 1, "height": -2, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"height should be positive, but got -2"));
    }

    #[test]
    fn test_mapping_new_from_json_height_is_zero() {
        let json = r#"{"width": 1, "height": 0, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"height should be positive, but got 0"));
    }

    #[test]
    fn test_mapping_new_from_json_mapping_is_missing() {
        let json = r#"{"width": 3, "height": 4}"#;
        let result = Mapping::new_from_json(json);
        assert!(result.is_ok());
        let mapping = result.unwrap();

        assert_eq!(mapping.width, 3);
        assert_eq!(mapping.height, 4);
        assert_eq!(mapping.mapping.len(), 0);
    }

    #[test]
    fn test_mapping_new_from_json_mapping_is_not_an_array() {
        let json = r#"{"width": 3, "height": 4, "mapping": {}}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *"mapping should be a list, but got {}"));
    }

    #[test]
    fn test_mapping_new_from_json_empty_mapping() {
        let json = r#"{"width": 3, "height": 4, "mapping": []}"#;
        let result = Mapping::new_from_json(json);
        assert!(result.is_ok());
        let mapping = result.unwrap();

        assert_eq!(mapping.width, 3);
        assert_eq!(mapping.height, 4);
        assert_eq!(mapping.mapping.len(), 0);
    }

    #[test]
    fn test_mapping_new_from_json_mapping_entry_is_not_object() {
        let json = r#"{"width": 3, "height": 4, "mapping": [4.2]}"#;
        let result = Mapping::new_from_json(json);
        assert!(
            matches!(result, Err(s) if s == *"each mapping entry should be an object, but got 4.2")
        );
    }

    #[test]
    fn test_mapping_new_from_json_x_is_missing() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 1, "y": 2, "qubit": 3},
            {"y": 5, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        assert!(
            matches!(result, Err(s) if s == *r#"x is not found in mapping entry {"qubit":4,"y":5}"#)
        );
    }

    #[test]
    fn test_mapping_new_from_json_x_is_not_a_number() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": {}, "y": 5, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        println!("{:?}", result);
        assert!(matches!(result, Err(s) if s == *r#"x should be an integer, but got {}"#));
    }

    #[test]
    fn test_mapping_new_from_json_x_is_negative() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": -1, "y": 5, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        println!("{:?}", result);
        assert!(matches!(result, Err(s) if s == *r#"x(-1) is out of bounds"#));
    }

    #[test]
    fn test_mapping_new_from_json_x_is_too_large() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 3, "y": 5, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        println!("{:?}", result);
        assert!(matches!(result, Err(s) if s == *r#"x(3) is out of bounds"#));
    }

    #[test]
    fn test_mapping_new_from_json_y_is_missing() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 1, "y": 2, "qubit": 3},
            {"x": 0, "qubit": 2}
        ]}"#;
        let result = Mapping::new_from_json(json);
        assert!(
            matches!(result, Err(s) if s == *r#"y is not found in mapping entry {"qubit":2,"x":0}"#)
        );
    }

    #[test]
    fn test_mapping_new_from_json_y_is_not_a_number() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 1, "y": 5.2, "qubit": 0}
        ]}"#;
        let result = Mapping::new_from_json(json);
        println!("{:?}", result);
        assert!(matches!(result, Err(s) if s == *r#"y should be an integer, but got 5.2"#));
    }

    #[test]
    fn test_mapping_new_from_json_y_is_negative() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 0, "y": -3, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        println!("{:?}", result);
        assert!(matches!(result, Err(s) if s == *r#"y(-3) is out of bounds"#));
    }

    #[test]
    fn test_mapping_new_from_json_y_is_too_large() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 0, "y": 5, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *r#"y(5) is out of bounds"#));
    }

    #[test]
    fn test_mapping_new_from_json_qubit_is_already_assinged() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 0, "y": 1, "qubit": 4},
            {"x": 2, "y": 3, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *r#"qubit 4 is already assigned"#));
    }

    #[test]
    fn test_mapping_new_from_json_qubit_is_already_occupied() {
        let json = r#"{"width": 3, "height": 4, "mapping": [
            {"x": 0, "y": 1, "qubit": 1},
            {"x": 0, "y": 1, "qubit": 4}
        ]}"#;
        let result = Mapping::new_from_json(json);
        assert!(matches!(result, Err(s) if s == *r#"(0, 1) is already occupied"#));
    }
    #[test]
    fn test_mapping_new_from_json() {
        let json = r#"{
            "width": 2,
            "height": 3,
            "mapping": [
                {"x": 1, "y": 2, "qubit": 1},
                {"x": 1, "y": 0, "qubit": 2},
                {"x": 0, "y": 2, "qubit": 3}
            ]
        }"#;
        let result = Mapping::new_from_json(json);
        assert!(result.is_ok());
        let mapping = result.unwrap();
        assert_eq!(mapping.width, 2);
        assert_eq!(mapping.height, 3);
        assert_eq!(
            mapping.mapping,
            vec![
                (1, 2, Qubit::new(1)),
                (1, 0, Qubit::new(2)),
                (0, 2, Qubit::new(3))
            ]
        );
    }
}
