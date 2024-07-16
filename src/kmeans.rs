use pyo3::prelude::*;
use rand::prelude::*;

#[pyclass]
pub struct KMeansRust {
    #[pyo3(get)]
    pub centers: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub allocations: Vec<usize>,
    #[pyo3(get)]
    pub max_iter: usize,
}

#[pymethods]
impl KMeansRust {
    #[new]
    pub fn fit_predict(num_centers: usize, max_iter: usize, data: Vec<Vec<f64>>) -> PyResult<Self> {
        let mut rng = thread_rng();
        let init_center_idx = rand::seq::index::sample(&mut rng, data.len(), num_centers);
        let mut centers: Vec<Vec<f64>> = init_center_idx
            .into_iter()
            .map(|x| data[x].clone())
            .collect();
        let mut n = 0;
        let mut allocation: Vec<usize> = vec![0; data.len()];
        let mut prev_allocation: Vec<usize> = vec![0; data.len()];
        while n < max_iter {
            for (rownum, row) in data.iter().enumerate() {
                let center_dists: Vec<f64> = centers
                    .iter()
                    .map(|c| {
                        c.iter()
                            .zip(row)
                            .fold(0f64, |acc, e| acc + (e.0 - e.1).powi(2))
                            .powf(0.5)
                    })
                    .collect();
                allocation[rownum] = center_dists
                    .iter()
                    .enumerate()
                    .min_by(|l, r| l.1.partial_cmp(r.1).unwrap())
                    .unwrap()
                    .0;
            }
            for i in 0..num_centers {
                let allocated_to_center = allocation
                    .iter()
                    .zip(&data)
                    .filter_map(|(alloc, x)| if *alloc == i { Some(x.clone()) } else { None })
                    .collect::<Vec<Vec<f64>>>();
                centers[i] = allocated_to_center
                    .iter()
                    .fold(vec![0f64; data[0].len()], |acc, e| {
                        acc.iter()
                            .zip(e)
                            .map(|(l, r)| l + (r / (allocated_to_center.len() as f64)))
                            .collect()
                    })
            }
            if allocation == prev_allocation {
                break;
            }
            prev_allocation = allocation.clone();
            n += 1;
        }
        Ok(KMeansRust {
            centers,
            allocations: allocation,
            max_iter,
        })
    }
}
