//! Marker bank (`CCTagMarkersBank`): radius-ratio tables per marker id.

use crate::bank_tables::{ID_FOUR_CROWNS, ID_THREE_CROWNS};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum BankError {
    #[error("unable to open the bank file: {0}")]
    Io(#[from] std::io::Error),
    #[error("bank file line {line}: cannot parse token {token:?}")]
    Parse { line: usize, token: String },
    #[error("no built-in bank for {0} crowns (only 3 or 4)")]
    UnsupportedCrowns(usize),
}

/// Radius ratios (outer radius / ring radius, descending) for every marker id.
/// Index into `markers` is the 0-based marker id reported by the detector.
#[derive(Clone, Debug, PartialEq)]
pub struct Bank {
    markers: Vec<Vec<f32>>,
}

impl Bank {
    /// Built-in bank for 3 or 4 crowns (CCTagMarkersBank.cpp:22-52).
    ///
    /// Upstream silently yields an *empty* bank for any other value, which then
    /// makes every marker `id_not_reliable`; we surface that as an error instead.
    pub fn builtin(n_crowns: usize) -> Result<Bank, BankError> {
        match n_crowns {
            3 => Ok(Bank {
                markers: ID_THREE_CROWNS.iter().map(|r| r.to_vec()).collect(),
            }),
            4 => Ok(Bank {
                markers: ID_FOUR_CROWNS.iter().map(|r| r.to_vec()).collect(),
            }),
            n => Err(BankError::UnsupportedCrowns(n)),
        }
    }

    /// Bank from a text file: one marker per line, whitespace separated values,
    /// each either `a/b` (unsigned integers, evaluated as a float division) or a
    /// decimal number (CCTagMarkersBank.hpp:42-65, `read` at .cpp:60-78).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Bank, BankError> {
        let text = std::fs::read_to_string(path)?;
        Bank::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Bank, BankError> {
        let mut markers = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let mut rr = Vec::new();
            for tok in line.split_whitespace() {
                let v = if let Some((a, b)) = tok.split_once('/') {
                    match (a.parse::<u32>(), b.parse::<u32>()) {
                        (Ok(a), Ok(b)) => a as f32 / b as f32,
                        _ => {
                            return Err(BankError::Parse {
                                line: i + 1,
                                token: tok.to_string(),
                            });
                        }
                    }
                } else {
                    match tok.parse::<f64>() {
                        Ok(v) => v as f32,
                        Err(_) => {
                            return Err(BankError::Parse {
                                line: i + 1,
                                token: tok.to_string(),
                            });
                        }
                    }
                };
                rr.push(v);
            }
            if !rr.is_empty() {
                markers.push(rr);
            }
        }
        Ok(Bank { markers })
    }

    pub fn from_markers(markers: Vec<Vec<f32>>) -> Bank {
        Bank { markers }
    }

    pub fn markers(&self) -> &[Vec<f32>] {
        &self.markers
    }

    pub fn len(&self) -> usize {
        self.markers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }
}
