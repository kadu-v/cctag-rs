//! Detection parameters. Mirrors `Params.hpp` / `Params.cpp` (defaults from the
//! `kDefault*` constants). Field names keep the upstream spelling (minus the
//! leading underscore) so the boost XML `<CCTagsParams>` file maps 1:1.

/// Weighting mode for `outlier_removal` (upstream `kWeight = INV_GRAD_WEIGHT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    None,
    InvGrad,
    /// Declared upstream but never implemented (would index an empty weight vector).
    InvSqrtGrad,
    /// Upstream computes `(255/g)*g` here (bug); unused by the pipeline.
    InvSquareGrad,
}

/// Compile-time weighting used by `completeFlowComponent` (Params.hpp `kWeight`).
pub const K_WEIGHT: Weight = Weight::InvGrad;

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub canny_thr_low: f32,
    pub canny_thr_high: f32,
    pub dist_search: usize,
    /// Dead on the CPU path (Bresenham.cpp re-reads the origin gradient, making the block a no-op).
    pub thr_gradient_mag_in_vote: i32,
    /// Must be 0; upstream throws otherwise (Vote.cpp:94).
    pub angle_voting: f32,
    pub ratio_voting: f32,
    pub average_vote_min: f32,
    pub thr_median_distance_ellipse: f32,
    pub maximum_nb_seeds: usize,
    pub maximum_nb_candidates_loop_two: usize,
    pub n_crowns: usize,
    pub n_circles: usize,
    pub min_points_segment_candidate: usize,
    pub min_votes_to_select_candidate: usize,
    pub thresh_robust_estimation_of_outer_ellipse: f32,
    pub ellipse_growing_elliptic_hull_width: f32,
    pub window_size_on_inner_elliptic_segment: usize,
    pub number_of_multires_layers: usize,
    pub number_of_processed_multires_layers: usize,
    pub n_samples_outer_ellipse: usize,
    pub num_cuts_in_ident_step: usize,
    pub num_samples_outer_edge_points_refinement: usize,
    /// Unused by the CPU identification (kept for XML round-trip).
    pub cuts_selection_trials: usize,
    pub sample_cut_length: usize,
    /// Must be odd.
    pub imaged_center_n_grid_sample: usize,
    pub imaged_center_neighbour_size: f32,
    pub min_ident_proba: f32,
    /// Unused by the CPU identification (kept for XML round-trip).
    pub use_lm_dif: bool,
    pub search_for_another_segment: bool,
    pub write_output: bool,
    pub do_identification: bool,
    /// CUDA only; unused.
    pub max_edges: u32,
}

impl Params {
    /// `Parameters(nCrowns)` with all upstream defaults.
    pub fn new(n_crowns: usize) -> Self {
        Params {
            canny_thr_low: 0.01,
            canny_thr_high: 0.04,
            dist_search: 30,
            thr_gradient_mag_in_vote: 2500,
            angle_voting: 0.0,
            ratio_voting: 4.0,
            average_vote_min: 0.0,
            thr_median_distance_ellipse: 3.0,
            maximum_nb_seeds: 500,
            maximum_nb_candidates_loop_two: 40,
            n_crowns,
            n_circles: 2 * n_crowns,
            min_points_segment_candidate: 10,
            min_votes_to_select_candidate: 3,
            thresh_robust_estimation_of_outer_ellipse: 30.0,
            ellipse_growing_elliptic_hull_width: 2.3,
            window_size_on_inner_elliptic_segment: 20,
            number_of_multires_layers: 4,
            number_of_processed_multires_layers: 4,
            n_samples_outer_ellipse: 150,
            num_cuts_in_ident_step: 22,
            num_samples_outer_edge_points_refinement: 20,
            cuts_selection_trials: 500,
            sample_cut_length: 100,
            imaged_center_n_grid_sample: 5,
            imaged_center_neighbour_size: 0.20,
            min_ident_proba: 1e-6,
            use_lm_dif: true,
            search_for_another_segment: true,
            write_output: false,
            do_identification: true,
            max_edges: 20000,
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        Params::new(3)
    }
}
