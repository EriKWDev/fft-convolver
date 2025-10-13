use num::Zero;
use realfft::{ComplexToReal, FftError, RealFftPlanner, RealToComplex};
use rustfft::{num_complex::Complex, FftNum};
use std::sync::Arc;

pub struct Fft<F: FftNum> {
    fft_forward: Arc<dyn RealToComplex<F>>,
    fft_inverse: Arc<dyn ComplexToReal<F>>,
    scratch: Vec<Complex<F>>,
}

impl<F: FftNum> std::fmt::Debug for Fft<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl<F: FftNum> Default for Fft<F> {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        Self {
            fft_forward: planner.plan_fft_forward(0),
            fft_inverse: planner.plan_fft_inverse(0),
            scratch: vec![],
        }
    }
}

impl<F: FftNum> Fft<F> {
    pub fn init(&mut self, length: usize) {
        profiling::function_scope!();

        let mut planner = RealFftPlanner::new();
        self.fft_forward = planner.plan_fft_forward(length);
        self.fft_inverse = planner.plan_fft_inverse(length);
        self.ensure_scratch();
    }

    fn ensure_scratch(&mut self) {
        let n = self
            .fft_forward
            .get_scratch_len()
            .max(self.fft_inverse.get_scratch_len());

        if self.scratch.len() != n {
            self.scratch.clear();
            self.scratch.extend((0..n).map(|_| Complex::zero()));
        }
    }

    pub fn forward(&mut self, input: &mut [F], output: &mut [Complex<F>]) -> Result<(), FftError> {
        profiling::function_scope!();

        self.ensure_scratch();
        self.fft_forward
            .process_with_scratch(input, output, &mut self.scratch)?;
        Ok(())
    }

    pub fn inverse(&mut self, input: &mut [Complex<F>], output: &mut [F]) -> Result<(), FftError> {
        profiling::function_scope!();

        self.ensure_scratch();
        self.fft_inverse
            .process_with_scratch(input, output, &mut self.scratch)?;

        // FFT Normalization
        let len = output.len();
        let len = F::from_usize(len).expect("usize can be converted to FftNum");

        output.iter_mut().for_each(|bin| {
            *bin = *bin / len;
        });

        Ok(())
    }
}
