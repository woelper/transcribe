use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::Axis;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use pyannote_rs::EmbeddingExtractor;
use serde::{Deserialize, Serialize};

/// Segments shorter than this can't produce a reliable voice embedding,
/// so they don't participate in clustering; they're assigned to the
/// nearest cluster afterwards.
const MIN_CLUSTER_SECS: f64 = 1.0;

/// Minimum similarity between a diarized voice and an enrolled profile
/// to assign the profile's name. Enrollment and meeting audio often
/// differ in mic and distance, so this sits below the clustering
/// threshold — but high enough not to mislabel a stranger.
const PROFILE_MATCH_THRESHOLD: f32 = 0.45;

/// Candidate matches below this similarity are NOT applied — the speaker
/// keeps their "Speaker N" number and the near-miss is reported instead,
/// usually a sign the person should be re-enrolled from audio recorded
/// under the same conditions as the meeting.
pub const WEAK_MATCH_SIMILARITY: f32 = 0.55;

/// A diarized time range attributed to one speaker (1-based id).
pub struct SpeakerSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

/// An enrolled voice: a name plus a reference embedding computed from a
/// short recording of that person (see [`voice_embedding`]).
#[derive(Clone, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub name: String,
    pub embedding: Vec<f32>,
}

/// Diarization result: who-speaks-when segments, plus names (with match
/// similarity) for the speaker ids whose voice confidently matched an
/// enrolled profile. Near-misses land in `weak_matches` instead and are
/// not applied to the transcript.
pub struct Diarization {
    pub segments: Vec<SpeakerSegment>,
    pub names: HashMap<usize, (String, f32)>,
    pub weak_matches: Vec<(String, f32)>,
    /// Mean voice embedding per speaker (`centroids[id - 1]`), usable as
    /// an enrollment fingerprint for that voice.
    pub centroids: Vec<Vec<f32>>,
}

pub struct DiarizeOptions<'a> {
    pub segmentation_model: &'a Path,
    pub embedding_model: &'a Path,
    pub max_speakers: usize,
    pub threshold: f32,
    pub profiles: &'a [SpeakerProfile],
}

/// Detect who speaks when: pyannote segmentation finds speech turns,
/// wespeaker embeddings turn each one into a voice fingerprint, and
/// agglomerative clustering groups the fingerprints into speakers.
/// Speakers whose voice matches an enrolled profile are named.
pub fn diarize(samples: &[f32], sample_rate: u32, opts: &DiarizeOptions) -> Result<Diarization> {
    // The pyannote models expect i16 PCM.
    let samples: Vec<i16> = to_i16(samples);

    let turns = speech_turns(&samples, sample_rate, opts.segmentation_model)?;

    let mut extractor = EmbeddingExtractor::new(opts.embedding_model)
        .map_err(|e| anyhow!("{e:?}"))
        .with_context(|| {
            format!(
                "failed to load embedding model {}",
                opts.embedding_model.display()
            )
        })?;

    // Fingerprint every turn. Embeddings can fail on very short turns;
    // those are dropped — the overlap lookup in speaker_for_range falls
    // back to the nearest labeled segment.
    let mut ranges: Vec<(f64, f64)> = Vec::new();
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    for (start, end) in turns {
        let lo = ((start * sample_rate as f64) as usize).min(samples.len());
        let hi = ((end * sample_rate as f64) as usize).min(samples.len());
        let Ok(embedding) = extractor.compute(&samples[lo..hi]) else {
            continue;
        };
        let mut embedding: Vec<f32> = embedding.collect();
        normalize(&mut embedding);
        ranges.push((start, end));
        embeddings.push(embedding);
    }

    let (speaker_of, mut centroids) = cluster(&ranges, &embeddings, opts.max_speakers, opts.threshold);
    let (names, weak_matches) = match_profiles(&centroids, opts.profiles);
    centroids.iter_mut().for_each(|c| normalize(c));

    Ok(Diarization {
        segments: ranges
            .iter()
            .zip(speaker_of)
            .map(|(&(start, end), speaker)| SpeakerSegment { start, end, speaker })
            .collect(),
        names,
        weak_matches,
        centroids,
    })
}

fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&x| (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

/// Compute the reference embedding for enrolling a speaker: keep only the
/// speech in the sample (VAD via the segmentation model), then fingerprint
/// it with the embedding model. Record ~10s of the person talking. The
/// sample must contain only this one person's voice.
pub fn voice_embedding(
    samples: &[f32],
    sample_rate: u32,
    segmentation_model: &Path,
    embedding_model: &Path,
) -> Result<Vec<f32>> {
    let samples = to_i16(samples);
    let turns = speech_turns(&samples, sample_rate, segmentation_model)?;
    let mut speech: Vec<i16> = Vec::new();
    for (start, end) in turns {
        let lo = ((start * sample_rate as f64) as usize).min(samples.len());
        let hi = ((end * sample_rate as f64) as usize).min(samples.len());
        speech.extend_from_slice(&samples[lo..hi]);
    }
    if (speech.len() as f64) < 3.0 * sample_rate as f64 {
        bail!("need at least ~3 seconds of speech in the sample");
    }
    // Embedding quality saturates well below a minute of speech; the cap
    // keeps enrolling from a long recording fast and bounded.
    speech.truncate(60 * sample_rate as usize);

    let mut extractor = EmbeddingExtractor::new(embedding_model)
        .map_err(|e| anyhow!("{e:?}"))
        .with_context(|| format!("failed to load embedding model {}", embedding_model.display()))?;
    let mut embedding: Vec<f32> = extractor
        .compute(&speech)
        .map_err(|e| anyhow!("failed to compute voice embedding: {e:?}"))?
        .collect();
    normalize(&mut embedding);
    Ok(embedding)
}

/// Speaker id → (enrolled name, match similarity).
type NamedSpeakers = HashMap<usize, (String, f32)>;

/// Greedily assign enrolled profile names to the speaker clusters they
/// resemble most (each profile and cluster used at most once).
/// `centroids[i]` belongs to speaker id `i + 1`. Assignments below
/// [`WEAK_MATCH_SIMILARITY`] are returned separately as near-misses
/// rather than applied — a wrong name is worse than "Speaker N".
fn match_profiles(
    centroids: &[Vec<f32>],
    profiles: &[SpeakerProfile],
) -> (NamedSpeakers, Vec<(String, f32)>) {
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for (c, centroid) in centroids.iter().enumerate() {
        for (p, profile) in profiles.iter().enumerate() {
            let similarity = cosine(centroid, &profile.embedding);
            if similarity >= PROFILE_MATCH_THRESHOLD {
                pairs.push((similarity, c, p));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut names = HashMap::new();
    let mut weak = Vec::new();
    let mut used_profiles = vec![false; profiles.len()];
    let mut used_clusters = vec![false; centroids.len()];
    for (similarity, c, p) in pairs {
        if used_profiles[p] || used_clusters[c] {
            continue;
        }
        used_profiles[p] = true;
        used_clusters[c] = true;
        if similarity >= WEAK_MATCH_SIMILARITY {
            names.insert(c + 1, (profiles[p].name.clone(), similarity));
        } else {
            weak.push((profiles[p].name.clone(), similarity));
        }
    }
    (names, weak)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = (a.iter().map(|x| x * x).sum::<f32>() * b.iter().map(|x| x * x).sum::<f32>()).sqrt();
    if norm > 0.0 { dot / norm } else { 0.0 }
}

/// Run the pyannote segmentation-3.0 model and return speech turns as
/// (start, end) seconds. The model emits a per-frame class (silence,
/// one of up to 3 local speakers, or overlaps thereof), so unlike plain
/// voice activity detection this also splits when the speaker changes
/// mid-stream with no silence gap. Local classes aren't consistent
/// across windows — global identity comes from embedding clustering.
fn speech_turns(samples: &[i16], sample_rate: u32, model: &Path) -> Result<Vec<(f64, f64)>> {
    // Frame geometry of segmentation-3.0 at 16 kHz.
    const FRAME_SIZE: usize = 270;
    const FRAME_START: usize = 721;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(model)
        .with_context(|| format!("failed to load segmentation model {}", model.display()))?;

    let window_size = sample_rate as usize * 10;
    let mut padded = samples.to_vec();
    padded.resize(samples.len().next_multiple_of(window_size), 0);

    let mut turns = Vec::new();
    let mut offset = FRAME_START;
    let mut speaking = 0usize; // local speaker holding the floor, 0 = nobody
    let mut turn_start = 0.0f64;
    let audio_end = samples.len() as f64 / sample_rate as f64;

    for window in padded.chunks(window_size) {
        // Class indices are local to a window (the model orders speakers
        // by appearance), so a turn must not continue across the boundary:
        // the same index may belong to a different person in the next
        // window. Close it; clustering rejoins same-voice pieces.
        if speaking != 0 {
            let t = (offset as f64 / sample_rate as f64).min(audio_end);
            if t > turn_start {
                turns.push((turn_start, t));
            }
            speaking = 0;
        }
        let array = ndarray::Array1::from_iter(window.iter().map(|&x| x as f32));
        let array = array.view().insert_axis(Axis(0)).insert_axis(Axis(1));
        let inputs = ort::inputs![TensorRef::from_array_view(array.into_dyn())?];
        let outputs = session.run(inputs)?;
        let (shape, data) = outputs
            .get("output")
            .context("segmentation output tensor not found")?
            .try_extract_tensor::<f32>()?;

        // Output is [1, frames, classes]. The powerset classes of
        // segmentation-3.0 are: silence, one of 3 local speakers, or a
        // pair of them speaking at once.
        let n_classes = shape[shape.len() - 1] as usize;
        let class_speakers: &[&[usize]] = if n_classes == 7 {
            &[&[], &[1], &[2], &[3], &[1, 2], &[1, 3], &[2, 3]]
        } else {
            // Unknown layout: treat every non-first class as its own speaker.
            &[&[], &[1], &[2], &[3], &[4], &[5], &[6], &[7], &[8], &[9]]
        };

        for frame in data.chunks(n_classes) {
            let class = frame
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i);
            let active = class_speakers[class.min(class_speakers.len() - 1)];
            // A backchannel overlapping whoever holds the floor doesn't
            // end their turn; only losing the floor entirely does.
            let now_speaking = if active.contains(&speaking) {
                speaking
            } else {
                // Floor changes hands: of the now-active local speakers,
                // pick the one the model scores highest (single-speaker
                // class index == local speaker id).
                active
                    .iter()
                    .copied()
                    .max_by(|&a, &b| frame[a.min(n_classes - 1)].total_cmp(&frame[b.min(n_classes - 1)]))
                    .unwrap_or(0)
            };
            if now_speaking != speaking {
                let t = (offset as f64 / sample_rate as f64).min(audio_end);
                if speaking != 0 && t > turn_start {
                    turns.push((turn_start, t));
                }
                turn_start = t;
                speaking = now_speaking;
            }
            offset += FRAME_SIZE;
        }
    }
    if speaking != 0 && audio_end > turn_start {
        turns.push((turn_start, audio_end));
    }
    Ok(turns)
}

/// Average-linkage agglomerative clustering over voice embeddings:
/// every (long enough) turn starts as its own cluster, then the two
/// most similar clusters merge until no pair is more similar than the
/// threshold (and at most max_speakers clusters remain). Returns the
/// 1-based speaker id per turn, numbered by order of first appearance,
/// plus each speaker's mean embedding (`centroids[id - 1]`).
///
/// Linkage is the exact mean pairwise cosine between clusters — for
/// unit embeddings that's dot(sum_a, sum_b) / (n_a * n_b). Unlike
/// centroid linkage this resists chaining: one crosstalk turn sitting
/// between two voices can't pull their clusters together.
fn cluster(
    ranges: &[(f64, f64)],
    embeddings: &[Vec<f32>],
    max_speakers: usize,
    threshold: f32,
) -> (Vec<usize>, Vec<Vec<f32>>) {
    struct Cluster {
        sum: Vec<f32>,
        count: usize,
        members: Vec<usize>,
    }
    let linkage = |a: &Cluster, b: &Cluster| -> f32 {
        let dot: f32 = a.sum.iter().zip(&b.sum).map(|(x, y)| x * y).sum();
        dot / (a.count * b.count) as f32
    };

    let mut clusters: Vec<Cluster> = Vec::new();
    let mut short_turns: Vec<usize> = Vec::new();
    for (i, embedding) in embeddings.iter().enumerate() {
        let (start, end) = ranges[i];
        if end - start >= MIN_CLUSTER_SECS {
            clusters.push(Cluster {
                sum: embedding.clone(),
                count: 1,
                members: vec![i],
            });
        } else {
            short_turns.push(i);
        }
    }
    // Degenerate input (all turns short): cluster everything.
    if clusters.is_empty() {
        for i in short_turns.drain(..) {
            clusters.push(Cluster {
                sum: embeddings[i].clone(),
                count: 1,
                members: vec![i],
            });
        }
    }

    while clusters.len() > 1 {
        let mut best: Option<(usize, usize, f32)> = None;
        for a in 0..clusters.len() {
            for b in a + 1..clusters.len() {
                let similarity = linkage(&clusters[a], &clusters[b]);
                if best.is_none_or(|(_, _, s)| similarity > s) {
                    best = Some((a, b, similarity));
                }
            }
        }
        let (a, b, similarity) = best.unwrap();
        if similarity < threshold {
            break;
        }
        let Cluster {
            sum, count, mut members,
        } = clusters.swap_remove(b);
        let merged = &mut clusters[a];
        for (c, x) in merged.sum.iter_mut().zip(&sum) {
            *c += x;
        }
        merged.count += count;
        merged.members.append(&mut members);
    }

    // More clusters than allowed speakers? Keep the ones with the most
    // speaking time; the rest are noise (crosstalk, window-boundary
    // fragments) whose turns fold into whichever speaker they resemble.
    // Folding turn-by-turn avoids chaining two real voices together the
    // way forced pairwise merging would.
    if clusters.len() > max_speakers {
        let duration = |c: &Cluster| -> f64 {
            c.members.iter().map(|&i| ranges[i].1 - ranges[i].0).sum()
        };
        clusters.sort_by(|a, b| duration(b).total_cmp(&duration(a)));
        for dropped in clusters.split_off(max_speakers) {
            short_turns.extend(dropped.members);
        }
    }

    // Attach leftover turns to the closest speaker (without growing it).
    for i in short_turns {
        let similarity = |c: &Cluster| -> f32 {
            let dot: f32 = c.sum.iter().zip(&embeddings[i]).map(|(x, y)| x * y).sum();
            dot / c.count as f32
        };
        let nearest = clusters
            .iter_mut()
            .max_by(|a, b| similarity(a).total_cmp(&similarity(b)))
            .unwrap();
        nearest.members.push(i);
    }

    // Number speakers by when they first hold a substantial turn.
    let mut speaker_of = vec![0; embeddings.len()];
    let mut clusters: Vec<(usize, Cluster)> = clusters
        .into_iter()
        .map(|c| {
            let first = c
                .members
                .iter()
                .copied()
                .filter(|&i| ranges[i].1 - ranges[i].0 >= MIN_CLUSTER_SECS)
                .min()
                .or_else(|| c.members.iter().copied().min())
                .unwrap();
            (first, c)
        })
        .collect();
    clusters.sort_by_key(|&(first, _)| first);
    let mut centroids = Vec::with_capacity(clusters.len());
    for (speaker, (_, cluster)) in clusters.into_iter().enumerate() {
        for &i in &cluster.members {
            speaker_of[i] = speaker + 1;
        }
        let mut centroid = cluster.sum;
        centroid.iter_mut().for_each(|x| *x /= cluster.count as f32);
        centroids.push(centroid);
    }
    (speaker_of, centroids)
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Speaker whose diarized range overlaps [start, end] the most,
/// falling back to the segment nearest to the range's midpoint.
pub fn speaker_for_range(segments: &[SpeakerSegment], start: f64, end: f64) -> Option<usize> {
    let best_overlap = segments
        .iter()
        .filter_map(|s| {
            let overlap = s.end.min(end) - s.start.max(start);
            (overlap > 0.0).then_some((overlap, s.speaker))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((_, speaker)) = best_overlap {
        return Some(speaker);
    }

    let mid = (start + end) / 2.0;
    let distance = |s: &SpeakerSegment| {
        if mid < s.start {
            s.start - mid
        } else if mid > s.end {
            mid - s.end
        } else {
            0.0
        }
    };
    segments
        .iter()
        .min_by(|a, b| distance(a).total_cmp(&distance(b)))
        .map(|s| s.speaker)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_assign_greedily_and_uniquely() {
        let centroids = vec![vec![1.0, 0.0], vec![0.9, 0.4], vec![0.0, 1.0]];
        let profiles = vec![
            SpeakerProfile { name: "Alice".into(), embedding: vec![1.0, 0.05] },
            SpeakerProfile { name: "Bob".into(), embedding: vec![0.05, 1.0] },
        ];
        let (names, weak) = match_profiles(&centroids, &profiles);
        // Alice matches cluster 1 best; cluster 2 also resembles her but
        // each profile is used once. Bob matches cluster 3.
        assert_eq!(names.get(&1).map(|(n, _)| n.as_str()), Some("Alice"));
        assert_eq!(names.get(&2), None);
        assert_eq!(names.get(&3).map(|(n, _)| n.as_str()), Some("Bob"));
        assert!(weak.is_empty());
    }

    #[test]
    fn dissimilar_voices_stay_unnamed() {
        let centroids = vec![vec![1.0, 0.0]];
        let profiles = vec![SpeakerProfile { name: "Alice".into(), embedding: vec![0.0, 1.0] }];
        let (names, weak) = match_profiles(&centroids, &profiles);
        assert!(names.is_empty());
        assert!(weak.is_empty());
    }

    #[test]
    fn borderline_match_is_reported_but_not_applied() {
        // cosine ≈ 0.5: above the candidate threshold, below confident.
        let centroids = vec![vec![1.0, 0.0]];
        let profiles = vec![SpeakerProfile { name: "Alice".into(), embedding: vec![0.5, 0.866] }];
        let (names, weak) = match_profiles(&centroids, &profiles);
        assert!(names.is_empty());
        assert_eq!(weak.len(), 1);
        assert_eq!(weak[0].0, "Alice");
    }
}
