pub struct RngForTesting {
    pub data: Vec<u64>,
    pub counter: u64,
}

impl rand::RngCore for RngForTesting {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        let ret = self.data[self.counter as usize % self.data.len()];
        self.counter += 1;
        ret
    }

    fn fill_bytes(&mut self, _dest: &mut [u8]) {
        unimplemented!()
    }

    fn try_fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), rand::Error> {
        unimplemented!()
    }
}

impl RngForTesting {
    pub fn new(data: &[u64]) -> Self {
        RngForTesting {
            data: data.to_vec(),
            counter: 0,
        }
    }

    pub fn new_with_zero() -> Self {
        RngForTesting {
            data: vec![0],
            counter: 0,
        }
    }

    // The program would crash when `next_64` or `next_u32` is called for the returned generator.
    pub fn new_unusable() -> Self {
        RngForTesting {
            data: vec![],
            counter: 0,
        }
    }
}
