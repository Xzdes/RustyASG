// --- File: src/data/sampler.rs ---

//! Data sampling strategies for DataLoader.

use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Trait for samplers - generators of indices.
pub trait Sampler: Iterator<Item = usize> {
    /// Returns the total number of samples.
    fn len(&self) -> usize;

    /// Returns whether the sampler is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resets the sampler to its initial state.
    fn reset(&mut self);
}

/// Sequential sampler - returns indices in order.
pub struct SequentialSampler {
    len: usize,
    current: usize,
}

impl SequentialSampler {
    /// Creates a sequential sampler for a dataset of the given size.
    pub fn new(len: usize) -> Self {
        Self { len, current: 0 }
    }
}

impl Iterator for SequentialSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.len {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

impl Sampler for SequentialSampler {
    fn len(&self) -> usize {
        self.len
    }

    fn reset(&mut self) {
        self.current = 0;
    }
}

/// Random sampler - returns indices in random order.
pub struct RandomSampler {
    indices: Vec<usize>,
    current: usize,
    seed: Option<u64>,
}

impl RandomSampler {
    /// Creates a random sampler for a dataset of the given size.
    pub fn new(len: usize) -> Self {
        let mut sampler = Self {
            indices: (0..len).collect(),
            current: 0,
            seed: None,
        };
        sampler.shuffle();
        sampler
    }

    /// Creates a random sampler with a fixed seed for reproducibility.
    pub fn with_seed(len: usize, seed: u64) -> Self {
        let mut sampler = Self {
            indices: (0..len).collect(),
            current: 0,
            seed: Some(seed),
        };
        sampler.shuffle();
        sampler
    }

    fn shuffle(&mut self) {
        if let Some(s) = self.seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            self.indices.shuffle(&mut rng);
        } else {
            let mut rng = rand::rng();
            self.indices.shuffle(&mut rng);
        }
    }
}

impl Iterator for RandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

impl Sampler for RandomSampler {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn reset(&mut self) {
        self.current = 0;
        self.shuffle();
    }
}

/// Batch sampler - groups indices into batches.
pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Creates a batch sampler.
    ///
    /// # Arguments
    ///
    /// * `sampler` - Inner sampler that generates indices.
    /// * `batch_size` - Batch size.
    /// * `drop_last` - Whether to drop the last incomplete batch.
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }

    /// Returns the number of batches.
    pub fn num_batches(&self) -> usize {
        let n = self.sampler.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// Resets the sampler.
    pub fn reset(&mut self) {
        self.sampler.reset();
    }
}

impl<S: Sampler> Iterator for BatchSampler<S> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            if let Some(idx) = self.sampler.next() {
                batch.push(idx);
            } else {
                break;
            }
        }

        if batch.is_empty() {
            return None;
        }

        if batch.len() < self.batch_size && self.drop_last {
            return None;
        }

        Some(batch)
    }
}

/// Weighted Random Sampler - sampling with weights.
pub struct WeightedRandomSampler {
    weights: Vec<f32>,
    cumulative_weights: Vec<f32>,
    num_samples: usize,
    current: usize,
    replacement: bool,
    used_indices: Vec<bool>,
    seed: Option<u64>,
}

impl WeightedRandomSampler {
    /// Creates a weighted sampler.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weights for each sample.
    /// * `num_samples` - Number of samples to generate.
    /// * `replacement` - Whether to sample with replacement.
    pub fn new(weights: Vec<f32>, num_samples: usize, replacement: bool) -> Self {
        let total: f32 = weights.iter().sum();
        let normalized: Vec<f32> = weights.iter().map(|w| w / total).collect();

        let mut cumulative = Vec::with_capacity(normalized.len());
        let mut sum = 0.0;
        for w in &normalized {
            sum += w;
            cumulative.push(sum);
        }

        Self {
            weights: normalized,
            cumulative_weights: cumulative,
            num_samples,
            current: 0,
            replacement,
            used_indices: vec![false; weights.len()],
            seed: None,
        }
    }

    /// Sets a seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn sample_one_replacement(&self, r: f32) -> usize {
        for (idx, &cw) in self.cumulative_weights.iter().enumerate() {
            if r < cw {
                return idx;
            }
        }
        self.cumulative_weights.len() - 1
    }

    fn sample_one_no_replacement(&mut self, r: f32) -> Option<usize> {
        // Without replacement - the weights must be recomputed.
        let available: Vec<(usize, f32)> = self
            .weights
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.used_indices[*i])
            .map(|(i, &w)| (i, w))
            .collect();

        if available.is_empty() {
            return None;
        }

        let total: f32 = available.iter().map(|(_, w)| w).sum();
        let scaled_r = r * total;

        let mut sum = 0.0;
        let mut last_idx = available[0].0;
        for &(idx, weight) in &available {
            sum += weight;
            last_idx = idx;
            if scaled_r < sum {
                self.used_indices[idx] = true;
                return Some(idx);
            }
        }

        // Fallback
        self.used_indices[last_idx] = true;
        Some(last_idx)
    }
}

impl Iterator for WeightedRandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        use rand::Rng;

        if self.current >= self.num_samples {
            return None;
        }

        let r: f32 = if let Some(s) = self.seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s + self.current as u64);
            rng.random()
        } else {
            rand::rng().random()
        };

        let result = if self.replacement {
            Some(self.sample_one_replacement(r))
        } else {
            self.sample_one_no_replacement(r)
        };

        if result.is_some() {
            self.current += 1;
        }
        result
    }
}

impl Sampler for WeightedRandomSampler {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn reset(&mut self) {
        self.current = 0;
        self.used_indices.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let mut sampler = SequentialSampler::new(5);
        let indices: Vec<_> = sampler.by_ref().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);

        sampler.reset();
        let indices: Vec<_> = sampler.by_ref().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler() {
        let mut sampler = RandomSampler::with_seed(5, 42);
        let indices: Vec<_> = sampler.by_ref().collect();
        assert_eq!(indices.len(), 5);

        // Verify that all indices are unique.
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_batch_sampler() {
        let sampler = SequentialSampler::new(10);
        let mut batch_sampler = BatchSampler::new(sampler, 3, false);

        let batches: Vec<_> = batch_sampler.by_ref().collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[3], vec![9]); // Last incomplete batch
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let sampler = SequentialSampler::new(10);
        let mut batch_sampler = BatchSampler::new(sampler, 3, true);

        let batches: Vec<_> = batch_sampler.by_ref().collect();
        assert_eq!(batches.len(), 3); // Last incomplete batch dropped
    }
}
