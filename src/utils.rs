use anyhow::{bail, Result};

pub fn matmul(l: &[Vec<f64>], r: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let (li, lj) = (l.len(), l[0].len());
    let (ri, rj) = (r.len(), r[0].len());

    if lj != ri {
        bail!(
            "Matrices of mismatched shapes, lhs had {} columns, rhs had {} rows",
            lj,
            ri
        )
    }

    let mut out = vec![vec![0f64; rj]; li];

    for rowidx in 0..li {
        let rowvec = &l[rowidx];
        for colidx in 0..rj {
            let colvec: Vec<f64> = r.iter().map(|x| x[colidx]).collect();
            out[rowidx][colidx] = rowvec.iter().zip(colvec).map(|(a, b)| a * b).sum::<f64>();
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test for single case of matmul (matrix x col vector)
    #[test]
    fn test_matmul() {
        let l: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let r: Vec<Vec<f64>> = vec![vec![3.0], vec![4.0], vec![5.0]];
        let mm = matmul(&l, &r);
        assert!(mm.is_ok());
        assert_eq!(mm.unwrap(), vec![vec![26.0], vec![38.0]]);
    }
}
