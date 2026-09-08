//! Lightweight stage timer for the example CLI and benchmarks.

use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct StageTimer {
    pub durations: Vec<(String, Duration)>,
    pub counters: Vec<(String, usize)>,
}

impl StageTimer {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn record(&mut self, name: &str, d: Duration) {
        self.durations.push((name.to_string(), d));
    }
    pub fn count(&mut self, name: &str, n: usize) {
        self.counters.push((name.to_string(), n));
    }
    pub fn clear(&mut self) {
        self.durations.clear();
        self.counters.clear();
    }
    /// Human-readable table.
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (n, d) in &self.durations {
            s.push_str(&format!("{n:<24} {:>10.3} ms\n", d.as_secs_f64() * 1e3));
        }
        for (n, c) in &self.counters {
            s.push_str(&format!("{n:<24} {c:>10}\n"));
        }
        s
    }
}
