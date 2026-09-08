//! Top-level detection (`Detection.cpp:772-998`): pyramid → multi-resolution
//! detection → identification → de-duplication.

pub mod candidate;
pub mod from_edges;
pub mod multires;

use crate::bank::Bank;
use crate::identification;
use crate::image::GrayImage;
use crate::marker::Marker;
use crate::params::Params;
use crate::pyramid::ImagePyramid;
use crate::robust::Pcg32;
use crate::timing::StageTimer;

/// How the pipeline is executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecMode {
    /// Fully sequential; a single PCG32 stream consumed in the same order as the
    /// upstream `CCTAG_SERIALIZE` build. Used for parity tests.
    Parity,
    /// Parallel (rayon) with per-candidate RNG streams. Deterministic for any
    /// thread count.
    Fast,
}

/// Reusable detector: parameters, bank, execution mode and work buffers.
pub struct Detector {
    pub params: Params,
    pub bank: Bank,
    pub mode: ExecMode,
    pyramid: ImagePyramid,
    multires: multires::MultiresWorkspace,
    /// RNG draws consumed by the last detection (parity diagnostics).
    pub last_rng_draws: u64,
}

impl Detector {
    pub fn new(params: Params, bank: Bank, mode: ExecMode) -> Self {
        Detector {
            params,
            bank,
            mode,
            pyramid: ImagePyramid::new(),
            multires: Default::default(),
            last_rng_draws: 0,
        }
    }

    /// Default parameters and built-in bank for `n_crowns` (3 or 4).
    pub fn with_crowns(n_crowns: usize, mode: ExecMode) -> Result<Self, crate::bank::BankError> {
        Ok(Detector::new(
            Params::new(n_crowns),
            Bank::builtin(n_crowns)?,
            mode,
        ))
    }

    /// `cctagDetection(markers, pipeId, frame, gray, params, bank, ...)`.
    /// Returns *all* markers, including the ones whose status is not reliable.
    pub fn detect(&mut self, gray: &GrayImage) -> Vec<Marker> {
        self.detect_timed(gray, None)
    }

    /// Detection without the final de-duplication / sort (markers in the order
    /// `cctagMultiresDetection` produces them: level 3 → 0). Used by parity tests.
    pub fn detect_raw(&mut self, gray: &GrayImage) -> Vec<Marker> {
        let params = &self.params;
        let mut rng = Pcg32::upstream();
        self.pyramid.build(gray, params);
        let mut markers = multires::multires_detection(
            &self.pyramid,
            params,
            self.mode,
            &mut rng,
            &mut self.multires,
            None,
        );
        self.last_rng_draws = rng.draws;
        if params.do_identification {
            identification::identify_all(
                &mut markers,
                &self.pyramid.level(0).src,
                &self.bank,
                params,
                self.mode,
            );
        }
        markers
    }

    pub fn detect_timed(
        &mut self,
        gray: &GrayImage,
        mut timer: Option<&mut StageTimer>,
    ) -> Vec<Marker> {
        let params = &self.params;
        // std::srand(1) upstream is vestigial; the real RNG is the pcg32 below.
        let mut rng = Pcg32::upstream();

        let t0 = std::time::Instant::now();
        self.pyramid.build(gray, params);
        if let Some(t) = timer.as_deref_mut() {
            t.record("pyramid", t0.elapsed());
        }

        let t1 = std::time::Instant::now();
        let mut markers = multires::multires_detection(
            &self.pyramid,
            params,
            self.mode,
            &mut rng,
            &mut self.multires,
            timer.as_deref_mut(),
        );
        self.last_rng_draws = rng.draws;
        if let Some(t) = timer.as_deref_mut() {
            t.record("multires", t1.elapsed());
        }

        if params.do_identification {
            let t2 = std::time::Instant::now();
            identification::identify_all(
                &mut markers,
                &self.pyramid.level(0).src,
                &self.bank,
                params,
                self.mode,
            );
            if let Some(t) = timer {
                t.record("identification", t2.elapsed());
            }
        }

        // Delete overlapping markers while keeping the best ones (two passes).
        let mut prelim: Vec<Marker> = Vec::new();
        for m in markers {
            multires::update(&mut prelim, m);
        }
        let mut fin: Vec<Marker> = Vec::new();
        for m in prelim {
            multires::update(&mut fin, m);
        }
        // std::list::sort is stable.
        fin.sort_by_key(|m| m.id);
        fin
    }
}
