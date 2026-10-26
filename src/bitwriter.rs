//! MSB-first bit writer for frame syntax.

/// A small MSB-first bit writer.
#[derive(Debug, Clone)]
pub(crate) struct BitWriter {
    out: Vec<u8>,
    /// Pending bits, right-aligned in the low `nbits` bits.
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            acc: 0,
            nbits: 0,
        }
    }

    /// Write the low `n` bits of `value`, MSB first.  `n <= 32`.
    pub(crate) fn write(&mut self, n: u32, value: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.acc = (self.acc << n) | (value & mask) as u64;
        self.nbits += n;
        while self.nbits >= 8 {
            let shift = self.nbits - 8;
            self.out.push((self.acc >> shift) as u8);
            self.nbits -= 8;
            self.acc &= (1u64 << self.nbits) - 1;
        }
    }

    /// Write a single flag bit.
    pub(crate) fn write_bool(&mut self, value: bool) {
        self.write(1, u32::from(value));
    }

    /// Write a signed value using `n`-bit two's complement (`put_sbits`).
    pub(crate) fn write_signed(&mut self, n: u32, value: i32) {
        self.write(n, value as u32);
    }

    /// Pad the current byte with zero bits.
    pub(crate) fn align(&mut self) {
        if self.nbits > 0 {
            let shift = 8 - self.nbits;
            self.out.push((self.acc << shift) as u8);
            self.acc = 0;
            self.nbits = 0;
        }
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::BitWriter;

    #[test]
    fn msb_first_round_trip() {
        let mut w = BitWriter::with_capacity(5);
        w.write(3, 0b101);
        w.write(5, 0b10011);
        w.write(8, 0xA5);
        w.write(16, 0xBEEF);
        w.write_signed(3, -2);
        let bytes = w.into_bytes();
        assert_eq!(bytes.len(), 5);

        let mut bits = Vec::new();
        for b in &bytes {
            for i in (0..8).rev() {
                bits.push((b >> i) & 1);
            }
        }
        let take = |n: usize, off: usize| -> u32 {
            bits[off..off + n]
                .iter()
                .fold(0, |a, &b| (a << 1) | b as u32)
        };
        assert_eq!(take(3, 0), 0b101);
        assert_eq!(take(5, 3), 0b10011);
        assert_eq!(take(8, 8), 0xA5);
        assert_eq!(take(16, 16), 0xBEEF);
        assert_eq!(take(3, 32), 0b110);
    }

    #[test]
    fn aligning_is_idempotent() {
        let mut w = BitWriter::with_capacity(4);
        w.write(1, 1);
        w.align();
        assert_eq!(w.into_bytes(), [0x80]);
    }

    #[test]
    fn booleans_and_signed_fields_are_bit_exact() {
        let mut w = BitWriter::with_capacity(4);
        w.write_bool(true);
        w.write_bool(false);
        w.write_signed(4, -1);
        w.write_signed(4, 7);
        w.write_signed(4, -8);
        w.write(0, 0xFFFF);
        assert_eq!(w.into_bytes(), [0b1011_1101, 0b1110_0000]);
    }
}
