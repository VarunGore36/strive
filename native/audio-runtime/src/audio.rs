use serde_json::{Value, json};
use std::{collections::VecDeque, time::Instant};

pub const RATE: usize = 16000;

#[derive(Debug)]
pub struct Window {
    pub samples: Vec<f32>,
    pub start: usize,
    pub end: usize,
    pub fresh: usize,
    pub queued: Instant,
}

/// A fixed allocation circular buffer; incoming packets never concatenate a
/// growing array. Only completed windows allocate an owned inference snapshot.
pub struct Capture {
    ring: Vec<f32>,
    write: usize,
    pub received: usize,
    pub sequence: u64,
    next_end: usize,
    hop: usize,
    capacity: usize,
    pending: VecDeque<Window>,
    pub queued: u64,
    pub scored: u64,
    pub dropped: u64,
    drops_unreported: u64,
    overflow_events: u64,
    max_depth: usize,
}

impl Capture {
    pub fn new(window: usize, hop: usize, capacity: usize) -> Self {
        assert!(window > 0 && hop > 0 && hop <= window && capacity > 0);
        Self {
            ring: vec![0.; window],
            write: 0,
            received: 0,
            sequence: 0,
            next_end: window,
            hop,
            capacity,
            pending: VecDeque::new(),
            queued: 0,
            scored: 0,
            dropped: 0,
            drops_unreported: 0,
            overflow_events: 0,
            max_depth: 0,
        }
    }

    pub fn ingest(&mut self, packet: &[u8], max_samples: usize) -> Result<(), &'static str> {
        if packet.len() < 6 || packet.len() > 3204 || packet.len() % 2 != 0 {
            return Err("Expected sequence header and 1–1600 s16le samples");
        }
        let seq = u32::from_le_bytes(packet[..4].try_into().unwrap()) as u64;
        if seq != self.sequence {
            return Err("Duplicate or out-of-order sequence");
        }
        if self.received + (packet.len() - 4) / 2 > max_samples {
            return Err("Maximum call duration reached");
        }
        let before = self.dropped;
        for bytes in packet[4..].chunks_exact(2) {
            self.ring[self.write] = i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.;
            self.write = (self.write + 1) % self.ring.len();
            self.received += 1;
            if self.received == self.next_end {
                let mut samples = Vec::with_capacity(self.ring.len());
                samples.extend_from_slice(&self.ring[self.write..]);
                samples.extend_from_slice(&self.ring[..self.write]);
                let fresh = if self.queued == 0 {
                    0
                } else {
                    self.ring.len() - self.hop
                };
                if self.pending.len() == self.capacity {
                    self.pending.pop_front();
                    self.dropped += 1;
                    self.drops_unreported += 1;
                }
                self.pending.push_back(Window {
                    samples,
                    start: self.received - self.ring.len(),
                    end: self.received,
                    fresh,
                    queued: Instant::now(),
                });
                self.queued += 1;
                self.max_depth = self.max_depth.max(self.pending.len());
                self.next_end += self.hop;
            }
        }
        self.sequence += 1;
        if self.dropped > before {
            self.overflow_events += 1;
        }
        Ok(())
    }

    pub fn take(&mut self) -> Option<(Window, u64)> {
        let window = self.pending.pop_front()?;
        self.scored += 1;
        let dropped = std::mem::take(&mut self.drops_unreported);
        Some((window, dropped))
    }

    pub fn stats(&self) -> Value {
        json!({"windows_queued":self.queued,"windows_scored":self.scored,
            "windows_dropped":self.dropped,"frames_ingested":self.sequence,
            "overflow_events":self.overflow_events,"max_depth":self.max_depth,
            "current_depth":self.pending.len(),"capacity":self.capacity})
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.ring.fill(0.);
        for w in &mut self.pending {
            w.samples.fill(0.);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet(seq: u32, samples: &[i16]) -> Vec<u8> {
        let mut bytes = seq.to_le_bytes().to_vec();
        for x in samples {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        bytes
    }
    #[test]
    fn geometries_exact_samples_overlap_and_fresh_audio() {
        for (window, hop) in [(32000, 16000), (32000, 8000), (64000, 8000)] {
            let mut c = Capture::new(window, hop, 100);
            let source: Vec<i16> = (0..160000).map(|i| (i % 65536) as i16).collect();
            for (seq, chunk) in source.chunks(317).enumerate() {
                c.ingest(&packet(seq as u32, chunk), source.len()).unwrap();
            }
            let mut count = 0;
            let mut fresh = 0;
            while let Some((w, d)) = c.take() {
                assert_eq!(d, 0);
                assert_eq!(w.start, count * hop);
                assert_eq!(
                    w.samples,
                    source[w.start..w.end]
                        .iter()
                        .map(|x| *x as f32 / 32768.)
                        .collect::<Vec<_>>()
                );
                fresh += w.samples.len() - w.fresh;
                count += 1;
            }
            assert_eq!(count, 1 + (source.len() - window) / hop);
            assert_eq!(fresh, window + (count - 1) * hop);
            assert_eq!(c.ring.len(), window);
        }
    }
    #[test]
    fn overload_keeps_newest_and_rejects_bad_sequences() {
        let mut c = Capture::new(320, 160, 1);
        c.ingest(&packet(0, &vec![32767; 1600]), 10000).unwrap();
        assert_eq!(c.dropped, 8);
        let (w, d) = c.take().unwrap();
        assert_eq!(w.end, 1600);
        assert_eq!(d, 8);
        assert!(c.ingest(&packet(0, &[-32768]), 10000).is_err());
        assert!(c.ingest(&[0; 3206], 10000).is_err());
        assert_eq!(c.sequence, 1);
    }
}
