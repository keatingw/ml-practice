use anyhow::{bail, Result};

/// Matrix multiplication using slices of vectors to represent matrices
/// Shapes of l and r must be aligned
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

/// Matrix-vector multiplication, assuming vector is a column vector
pub fn matvecmul(mat: &[Vec<f64>], vec: &[f64]) -> Result<Vec<f64>> {
    let r: Vec<Vec<f64>> = vec.iter().map(|x| vec![*x]).collect();
    let mm = matmul(mat, &r)?;
    Ok(mm.iter().map(|x| x[0]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test for single case of matmul (matrix x col vector)
    #[test]
    fn test_matmul() {
        let l: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let r: Vec<Vec<f64>> = vec![vec![3.0], vec![4.0], vec![5.0]];
        let mm = matmul(&l, &r).unwrap_or_default();
        assert_eq!(mm, vec![vec![26.0], vec![38.0]]);
    }

    /// Test for single case of matvecmul (matrix x col vector)
    #[test]
    fn test_matvecmul() {
        let l: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let colvec: Vec<f64> = vec![3.0, 4.0, 5.0];
        let mvm = matvecmul(&l, &colvec).unwrap_or_default();
        assert_eq!(mvm, vec![26.0, 38.0]);
    }
}
