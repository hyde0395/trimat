/// i8 슬라이스(값 ∈ {-1,0,1})를 두 비트플레인으로 인코딩.
/// 반환: (nonzero_plane, sign_plane) — 각 ceil(n/8) 바이트
pub fn encode(vals: &[i8]) -> (Vec<u8>, Vec<u8>) {
    let bytes = bit_len(vals.len());
    let mut nonzero = vec![0u8; bytes];
    let mut sign    = vec![0u8; bytes];
    for (i, &v) in vals.iter().enumerate() {
        if v != 0 { nonzero[i / 8] |= 1 << (i % 8); }
        if v  < 0 { sign[i / 8]    |= 1 << (i % 8); }
    }
    (nonzero, sign)
}

/// 비트플레인 두 개를 i8 슬라이스로 복원.
pub fn decode(nonzero: &[u8], sign: &[u8], n: usize) -> Vec<i8> {
    (0..n)
        .map(|i| {
            let nz = (nonzero[i / 8] >> (i % 8)) & 1;
            let sg = (sign[i / 8]    >> (i % 8)) & 1;
            if nz == 0 { 0 } else if sg == 0 { 1 } else { -1 }
        })
        .collect()
}

pub fn bit_len(n: usize) -> usize {
    (n + 7) / 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let vals: Vec<i8> = vec![1, 0, -1, 1, -1, 0, 0, 1, -1];
        let (nz, sg) = encode(&vals);
        assert_eq!(decode(&nz, &sg, vals.len()), vals);
    }

    #[test]
    fn test_all_zeros() {
        let vals = vec![0i8; 16];
        let (nz, sg) = encode(&vals);
        assert!(nz.iter().all(|&b| b == 0));
        assert!(sg.iter().all(|&b| b == 0));
        assert_eq!(decode(&nz, &sg, 16), vals);
    }

    #[test]
    fn test_all_pos() {
        let vals = vec![1i8; 8];
        let (nz, sg) = encode(&vals);
        assert_eq!(nz, vec![0xFF]);
        assert_eq!(sg, vec![0x00]);
    }

    #[test]
    fn test_all_neg() {
        let vals = vec![-1i8; 8];
        let (nz, sg) = encode(&vals);
        assert_eq!(nz, vec![0xFF]);
        assert_eq!(sg, vec![0xFF]);
    }

    #[test]
    fn test_bitplane_size() {
        let (nz, _) = encode(&vec![1i8; 9]);
        assert_eq!(nz.len(), 2);
    }
}
