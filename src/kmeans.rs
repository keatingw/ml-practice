use pyo3::prelude::*;
use rand::prelude::*;
use rand::seq::index::sample;

/// KMeans fitting class implemented in Rust
#[pyclass]
pub struct KMeansRust {
    #[pyo3(get)]
    pub num_centers: usize,

    #[pyo3(get)]
    pub max_iter: usize,

    #[pyo3(get)]
    pub seed: u64,

    #[pyo3(get)]
    pub centers: Option<Vec<Vec<f64>>>,

    #[pyo3(get)]
    pub allocations: Option<Vec<usize>>,
}

#[pymethods]
impl KMeansRust {
    /// Create KMeansRust object and fit to the data
    #[new]
    pub fn new(num_centers: usize, max_iter: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(KMeansRust {
            num_centers,
            max_iter,
            seed: seed.unwrap_or_default(),
            centers: None,
            allocations: None,
        })
    }

    /// Fits kmeans to data given
    pub fn fit(&mut self, data: Vec<Vec<f64>>) -> PyResult<()> {
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Set initial centers to random set of points
        let init_center_idx = sample(&mut rng, data.len(), self.num_centers);
        let mut centers: Vec<Vec<f64>> = init_center_idx
            .into_iter()
            .map(|x| data[x].clone())
            .collect();

        // start looping to assign points to their clusters
        let mut allocations: Vec<usize> = vec![0; data.len()];
        let mut prev_allocation: Vec<usize> = vec![0; data.len()];

        for _ in 0..self.max_iter {
            // For each row, assign to its nearest center based on euclidean distance
            for (rownum, row) in data.iter().enumerate() {
                allocations[rownum] = centers
                    .iter()
                    .map(|c| squared_euclidean_dist(c, row))
                    .enumerate()
                    .min_by(|l, r| l.1.partial_cmp(&r.1).unwrap())
                    .unwrap()
                    .0;
            }

            // For each cluster, recalculate its center after the new allocations
            for i in 0..self.num_centers {
                // Get all points in the cluster
                let allocated_to_center = allocations
                    .iter()
                    .zip(&data)
                    .filter_map(|(alloc, x)| if *alloc == i { Some(x.clone()) } else { None })
                    .collect::<Vec<Vec<f64>>>();

                // For recompute the center as the average of the cluster's points
                centers[i] = allocated_to_center
                    .iter()
                    .fold(vec![0f64; data[0].len()], |acc, e| {
                        acc.iter()
                            .zip(e)
                            .map(|(l, r)| l + (r / (allocated_to_center.len() as f64)))
                            .collect()
                    })
            }

            // Basic stopping criterion for if allocations have converged
            if allocations == prev_allocation {
                break;
            }

            // save prior state and continue iterations
            prev_allocation = allocations.clone();
        }
        self.centers = Some(centers);
        self.allocations = Some(allocations);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "KMeansRust(num_centers={}, max_iter={}, seed={})",
            self.num_centers, self.max_iter, self.seed,
        )
    }
}

/// Calculates sum of squared differences
fn squared_euclidean_dist(l: &[f64], r: &[f64]) -> f64 {
    l.iter()
        .zip(r)
        .map(|(x1, x2)| (x1 - x2).powi(2))
        .sum::<f64>()
}
