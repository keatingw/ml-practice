use pyo3::{exceptions::PyValueError, prelude::*};

use super::utils;

/// Linear regression struct
#[pyclass]
pub struct LinRegGDRust {
    #[pyo3(get)]
    pub intercept: bool,

    #[pyo3(get)]
    pub lr: f64,

    #[pyo3(get)]
    pub num_iter: usize,

    #[pyo3(get)]
    pub weights: Option<Vec<f64>>,
}

#[pymethods]
impl LinRegGDRust {
    /// Create LinRegGDRust object
    #[new]
    #[pyo3(signature=(intercept=true, lr=0.01, num_iter=100))]
    pub fn new(intercept: bool, lr: f64, num_iter: usize) -> PyResult<Self> {
        Ok(LinRegGDRust {
            intercept,
            lr,
            num_iter,
            weights: None,
        })
    }

    /// Fit weights using gradient descent on entire batch
    pub fn fit(&mut self, data: Vec<Vec<f64>>, target: Vec<f64>) -> PyResult<()> {
        let mut data = data.clone();
        if self.intercept {
            LinRegGDRust::add_intercept(&mut data)
        }
        let dims = data[0].len();
        let num_rows = data.len();
        self.weights = Some(vec![0.0; dims]);
        let w = self.weights.as_mut().unwrap();
        for _ in 0..self.num_iter {
            let preds = utils::matvecmul(&data, w)?;
            let resid: Vec<f64> = preds
                .iter()
                .zip(&target)
                .map(|(pred, real)| pred - real)
                .collect();
            let mut grads = vec![0.0; dims];
            for (row_idx, row) in data.iter().enumerate() {
                for var_idx in 0..dims {
                    grads[var_idx] += row[var_idx] * resid[row_idx];
                }
            }
            w.iter_mut()
                .zip(grads)
                .for_each(|(weight, grad)| *weight -= grad * self.lr / (num_rows as f64))
        }
        Ok(())
    }

    /// Predict outputs on new data
    pub fn predict(&self, data: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let mut data = data.clone();
        if self.intercept {
            LinRegGDRust::add_intercept(&mut data)
        }
        if let Some(ref w) = self.weights {
            Ok(utils::matvecmul(&data, w)?)
        } else {
            Err(PyValueError::new_err("Weights not yet set on model object"))
        }
    }
}

impl LinRegGDRust {
    fn add_intercept(data: &mut [Vec<f64>]) -> () {
        data.iter_mut().for_each(|x| x.insert(0, 1.0));
    }
}
