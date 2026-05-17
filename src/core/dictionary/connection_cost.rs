//! Bigram connection cost table between (prev_pos, cur_pos), indexed by
//! `PartOfSpeech::idx()`. Lower = more natural; added to segmentation cost
//! so higher = penalty. Normalized to [0, 8] to match engine boost ×10 scale.
//!
//! AUTO-GENERATED from IPADIC matrix.def by src/bin/generate_matrix.rs.
//! To regenerate:
//!   cargo run --bin generate-matrix --release > src/core/dictionary/connection_cost.rs

use super::PartOfSpeech;

pub const CONNECTION_COST: [[f64; PartOfSpeech::COUNT]; PartOfSpeech::COUNT] = [
    //       N      V    Adj    Adv   Part    Aux   Conj   Intj   Pref    Suf  Other 
    [ 5.189, 5.351, 5.644, 6.424, 1.667, 3.469, 6.977, 6.342, 6.516, 0.000, 4.184  ],  // N ->
    [ 6.985, 4.429, 5.075, 6.374, 3.851, 1.789, 7.206, 5.751, 6.281, 6.497, 5.315  ],  // V ->
    [ 5.818, 5.672, 5.577, 6.618, 3.867, 3.837, 4.763, 5.437, 5.065, 4.827, 5.253  ],  // Adj ->
    [ 5.650, 4.782, 4.782, 4.727, 4.954, 5.229, 5.519, 4.895, 4.953, 5.647, 5.031  ],  // Adv ->
    [ 4.679, 4.135, 4.874, 4.517, 4.768, 6.287, 5.106, 6.452, 5.281, 7.354, 3.885  ],  // Part ->
    [ 5.737, 6.142, 5.698, 5.695, 3.632, 2.531, 5.273, 5.205, 5.472, 7.476, 4.917  ],  // Aux ->
    [ 4.734, 5.136, 5.578, 3.830, 5.042, 5.711, 5.428, 5.577, 4.439, 4.129, 4.703  ],  // Conj ->
    [ 7.387, 5.913, 5.795, 5.474, 5.429, 5.776, 4.335, 3.136, 5.093, 7.404, 3.981  ],  // Intj ->
    [ 2.986, 4.922, 4.096, 6.294, 7.226, 5.609, 5.119, 5.002, 4.080, 3.181, 5.662  ],  // Pref ->
    [ 6.304, 5.568, 6.079, 6.825, 1.267, 3.240, 7.476, 6.991, 8.000, 1.312, 3.952  ],  // Suf ->
    [ 4.681, 6.075, 5.880, 3.735, 5.527, 6.205, 3.569, 3.674, 4.823, 5.724, 4.717  ],  // Other ->
];
