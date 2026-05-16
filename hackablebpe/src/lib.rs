// hackablebpe is inspired by Andrej Karpathy's rustbpe:
// https://github.com/karpathy/rustbpe

use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};

type Pair = (u32, u32);
type PairKey = u64;
type MergeValue = u64;

pub const BACKEND: &str = "hackablebpe_bytelevel";
pub const FORMAT: &str = "hackablebpe";
pub const VERSION: u32 = 1;
pub const DEFAULT_BUFFER_SIZE: usize = 8192;
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+";
const MEMORY_CHECK_INTERVAL_MERGES: usize = 256;
const AUTO_MEMORY_LIMIT_HEADROOM_NUMERATOR: u64 = 9;
const AUTO_MEMORY_LIMIT_HEADROOM_DENOMINATOR: u64 = 10;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SerializedMerge {
    pub left: u32,
    pub right: u32,
    pub id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTokenizer {
    pub format: String,
    pub version: u32,
    pub backend: String,
    pub special_tokens: Vec<String>,
    pub requested_vocab_size: usize,
    pub min_frequency: u64,
    pub pattern: String,
    pub merges: Vec<SerializedMerge>,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub input_paths: Vec<PathBuf>,
    pub output_path: PathBuf,
    pub vocab_size: usize,
    pub jsonl_text_field: String,
    pub min_frequency: u64,
    pub special_tokens: Vec<String>,
    pub buffer_size: usize,
    pub max_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TrainStats {
    pub documents: u64,
    pub raw_bytes: u64,
    pub pretokens: u64,
    pub unique_pretokens: usize,
    pub merges: usize,
    pub vocab_size: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeJob {
    pair: PairKey,
    count: u64,
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct Word {
    ids: Vec<u32>,
    pair_counts: Vec<(PairKey, u32)>,
}

#[pyclass(name = "Tokenizer")]
#[derive(Debug, Clone)]
pub struct RuntimeTokenizer {
    special_tokens: Vec<String>,
    special_to_id: AHashMap<String, u32>,
    special_patterns: Vec<(String, u32)>,
    merge_ranks: AHashMap<PairKey, MergeValue>,
    id_to_bytes: Vec<Option<Vec<u8>>>,
}

impl Word {
    fn new(ids: Vec<u32>) -> Self {
        let pair_counts = count_pairs_in_ids(&ids);
        Self { ids, pair_counts }
    }

    fn pair_count(&self, pair: PairKey) -> u32 {
        self.pair_counts
            .iter()
            .find_map(|(candidate, count)| (*candidate == pair).then_some(*count))
            .unwrap_or(0)
    }

    fn apply_pair_delta(&mut self, pair: PairKey, delta: i32) {
        if delta == 0 {
            return;
        }
        if let Some(idx) = self
            .pair_counts
            .iter()
            .position(|(candidate, _)| *candidate == pair)
        {
            let current = self.pair_counts[idx].1 as i64 + i64::from(delta);
            if current <= 0 {
                self.pair_counts.swap_remove(idx);
            } else {
                self.pair_counts[idx].1 = current as u32;
            }
        } else if delta > 0 {
            self.pair_counts.push((pair, delta as u32));
        }
    }

    fn merge_pair(&mut self, pair: PairKey, new_id: u32) -> Vec<(PairKey, i32)> {
        if self.ids.len() < 2 || self.pair_count(pair) == 0 {
            return Vec::new();
        }

        let (a, b) = unpack_pair(pair);
        let old = &self.ids;
        let mut out = Vec::with_capacity(old.len());
        let mut deltas = Vec::new();
        let mut i = 0;
        while i < old.len() {
            if i + 1 < old.len() && old[i] == a && old[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < old.len() {
                    Some(old[i + 2])
                } else {
                    None
                };
                if let Some(x) = left {
                    deltas.push((pack_pair(x, a), -1));
                    deltas.push((pack_pair(x, new_id), 1));
                }
                deltas.push((pack_pair(a, b), -1));
                if let Some(y) = right {
                    deltas.push((pack_pair(b, y), -1));
                    deltas.push((pack_pair(new_id, y), 1));
                }
                out.push(new_id);
                i += 2;
            } else {
                out.push(old[i]);
                i += 1;
            }
        }

        if out.len() == self.ids.len() {
            return Vec::new();
        }
        for (delta_pair, delta) in &deltas {
            self.apply_pair_delta(*delta_pair, *delta);
        }
        self.ids = out;
        deltas
    }
}

fn gpt4_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(GPT4_PATTERN).expect("valid GPT-4 pre-tokenizer regex"))
}

pub fn min_vocab_size(special_tokens_len: usize) -> usize {
    special_tokens_len + 256
}

fn byte_id(special_tokens_len: usize, byte: u8) -> u32 {
    special_tokens_len as u32 + u32::from(byte)
}

fn pack_pair(left: u32, right: u32) -> PairKey {
    (u64::from(left) << 32) | u64::from(right)
}

fn unpack_pair(pair: PairKey) -> Pair {
    ((pair >> 32) as u32, pair as u32)
}

fn pack_merge_value(rank: usize, new_id: u32) -> MergeValue {
    ((rank as u64) << 32) | u64::from(new_id)
}

fn merge_rank(value: MergeValue) -> usize {
    (value >> 32) as usize
}

fn merge_new_id(value: MergeValue) -> u32 {
    value as u32
}

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn py_io_error(message: impl Into<String>) -> PyErr {
    PyIOError::new_err(message.into())
}

fn for_each_pretoken<'a, F>(text: &'a str, mut f: F)
where
    F: FnMut(&'a str),
{
    for mat in gpt4_regex().find_iter(text) {
        let piece = mat.as_str();
        if !piece.is_empty() {
            f(piece);
        }
    }
}

fn count_pairs_in_ids(ids: &[u32]) -> Vec<(PairKey, u32)> {
    if ids.len() < 2 {
        return Vec::new();
    }
    let mut counts = AHashMap::with_capacity(ids.len() - 1);
    for pair in ids.windows(2).map(|window| pack_pair(window[0], window[1])) {
        *counts.entry(pair).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

fn validate_special_tokens(special_tokens: &[String]) -> Result<(), String> {
    let mut seen = AHashSet::new();
    for token in special_tokens {
        if token.is_empty() {
            return Err("special tokens must not be empty".to_string());
        }
        if !seen.insert(token) {
            return Err(format!("duplicate special token {token:?}"));
        }
    }
    Ok(())
}

fn build_id_to_bytes(
    special_tokens_len: usize,
    merges: &[SerializedMerge],
) -> Result<Vec<Option<Vec<u8>>>, String> {
    let mut id_to_bytes = Vec::with_capacity(min_vocab_size(special_tokens_len) + merges.len());
    for _ in 0..special_tokens_len {
        id_to_bytes.push(None);
    }
    for byte in 0..=255 {
        id_to_bytes.push(Some(vec![byte as u8]));
    }
    for (rank, merge) in merges.iter().enumerate() {
        let expected_id = min_vocab_size(special_tokens_len) + rank;
        if merge.id as usize != expected_id {
            return Err(format!(
                "merge id {} is not contiguous; expected {}",
                merge.id, expected_id
            ));
        }
        let left = id_to_bytes
            .get(merge.left as usize)
            .and_then(|entry| entry.as_ref())
            .ok_or_else(|| format!("merge references unknown left id {}", merge.left))?;
        let right = id_to_bytes
            .get(merge.right as usize)
            .and_then(|entry| entry.as_ref())
            .ok_or_else(|| format!("merge references unknown right id {}", merge.right))?;
        let mut bytes = Vec::with_capacity(left.len() + right.len());
        bytes.extend_from_slice(left);
        bytes.extend_from_slice(right);
        id_to_bytes.push(Some(bytes));
    }
    Ok(id_to_bytes)
}

impl RuntimeTokenizer {
    fn from_serialized(serialized: SerializedTokenizer) -> Result<Self, String> {
        if serialized.format != FORMAT {
            return Err(format!(
                "unsupported tokenizer format {}; expected {FORMAT}",
                serialized.format
            ));
        }
        if serialized.version != VERSION {
            return Err(format!(
                "unsupported tokenizer version {}; expected {VERSION}",
                serialized.version
            ));
        }
        validate_special_tokens(&serialized.special_tokens)?;

        let mut special_to_id = AHashMap::with_capacity(serialized.special_tokens.len());
        for (id, token) in serialized.special_tokens.iter().enumerate() {
            special_to_id.insert(token.clone(), id as u32);
        }
        let mut special_patterns = serialized
            .special_tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect::<Vec<_>>();
        special_patterns.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

        let id_to_bytes = build_id_to_bytes(serialized.special_tokens.len(), &serialized.merges)?;
        let mut merge_ranks = AHashMap::with_capacity(serialized.merges.len());
        for (rank, merge) in serialized.merges.iter().enumerate() {
            let pair = pack_pair(merge.left, merge.right);
            if merge_ranks
                .insert(pair, pack_merge_value(rank, merge.id))
                .is_some()
            {
                return Err(format!(
                    "duplicate merge pair ({}, {})",
                    merge.left, merge.right
                ));
            }
        }

        Ok(Self {
            special_tokens: serialized.special_tokens,
            special_to_id,
            special_patterns,
            merge_ranks,
            id_to_bytes,
        })
    }

    fn apply_merges(&self, ids: &mut Vec<u32>) {
        if ids.len() < 2 || self.merge_ranks.is_empty() {
            return;
        }

        let mut merged = Vec::with_capacity(ids.len());
        loop {
            let mut best: Option<(usize, Pair, u32)> = None;
            for pair in ids.windows(2).map(|window| (window[0], window[1])) {
                if let Some(value) = self.merge_ranks.get(&pack_pair(pair.0, pair.1)).copied() {
                    let rank = merge_rank(value);
                    let new_id = merge_new_id(value);
                    match best {
                        Some((best_rank, _, _)) if rank >= best_rank => {}
                        _ => best = Some((rank, pair, new_id)),
                    }
                }
            }
            let Some((_, best_pair, new_id)) = best else {
                break;
            };

            merged.clear();
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == best_pair.0 && ids[i + 1] == best_pair.1 {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(ids[i]);
                    i += 1;
                }
            }
            if merged.len() == ids.len() {
                break;
            }
            std::mem::swap(ids, &mut merged);
            if ids.len() < 2 {
                break;
            }
        }
    }

    fn encode_piece(&self, piece: &str) -> Vec<u32> {
        let mut ids = piece
            .as_bytes()
            .iter()
            .map(|byte| byte_id(self.special_tokens.len(), *byte))
            .collect::<Vec<_>>();
        self.apply_merges(&mut ids);
        ids
    }

    fn append_regular_text(&self, text: &str, ids: &mut Vec<u32>) {
        for_each_pretoken(text, |piece| ids.extend(self.encode_piece(piece)));
    }

    fn next_special_match(&self, text: &str, cursor: usize) -> Option<(usize, usize, u32)> {
        self.special_patterns
            .iter()
            .filter_map(|(pattern, id)| {
                text[cursor..]
                    .find(pattern)
                    .map(|offset| (cursor + offset, pattern.len(), *id))
            })
            .min_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| right.1.cmp(&left.1))
                    .then_with(|| left.2.cmp(&right.2))
            })
    }

    fn encode_to_ids(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut ids = Vec::new();
        let mut cursor = 0;
        while cursor < text.len() {
            if let Some((start, special_len, special_id)) = self.next_special_match(text, cursor) {
                self.append_regular_text(&text[cursor..start], &mut ids);
                ids.push(special_id);
                cursor = start + special_len;
            } else {
                self.append_regular_text(&text[cursor..], &mut ids);
                break;
            }
        }
        ids
    }

    fn decode_ids(&self, ids: &[u32]) -> Result<String, String> {
        let mut bytes = Vec::new();
        for id in ids {
            let Some(entry) = self.id_to_bytes.get(*id as usize) else {
                return Err(format!("unknown token id {id}"));
            };
            if let Some(token_bytes) = entry {
                bytes.extend_from_slice(token_bytes);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn parse_u64_file(path: &str) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    let text = text.trim();
    if text == "max" {
        return None;
    }
    text.parse().ok()
}

fn cgroup_memory_limit_bytes() -> Option<u64> {
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        let limit = parse_u64_file(path)?;
        if limit > 0 && limit < (1_u64 << 60) {
            return Some(limit);
        }
    }
    None
}

fn current_rss_bytes() -> Option<u64> {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                return Some(kb * 1024);
            }
        }
    }

    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let kb = text.trim().parse::<u64>().ok()?;
    Some(kb * 1024)
}

fn effective_memory_limit_bytes(configured: Option<u64>) -> Option<u64> {
    configured.or_else(|| {
        cgroup_memory_limit_bytes().map(|limit| {
            limit * AUTO_MEMORY_LIMIT_HEADROOM_NUMERATOR / AUTO_MEMORY_LIMIT_HEADROOM_DENOMINATOR
        })
    })
}

fn check_memory_limit(limit: Option<u64>, context: &str) -> Result<(), String> {
    let Some(limit) = limit else {
        return Ok(());
    };
    let Some(rss) = current_rss_bytes() else {
        return Ok(());
    };
    if rss <= limit {
        return Ok(());
    }
    Err(format!(
        "memory limit exceeded during {context}: current RSS is {:.1} MiB, configured limit is {:.1} MiB. \
         Tokenizer training memory grows with corpus diversity; reduce input size, train on a sample, or raise the memory limit.",
        rss as f64 / 1024.0 / 1024.0,
        limit as f64 / 1024.0 / 1024.0,
    ))
}

fn count_batch(batch: &[String]) -> (AHashMap<CompactString, u64>, u64) {
    batch
        .par_iter()
        .map(|text| {
            let mut local = AHashMap::new();
            let mut pretokens = 0;
            for_each_pretoken(text, |piece| {
                *local.entry(CompactString::from(piece)).or_insert(0) += 1;
                pretokens += 1;
            });
            (local, pretokens)
        })
        .reduce(
            || (AHashMap::new(), 0),
            |(mut left, left_pretokens), (right, right_pretokens)| {
                for (piece, count) in right {
                    *left.entry(piece).or_insert(0) += count;
                }
                (left, left_pretokens + right_pretokens)
            },
        )
}

fn merge_batch_counts(
    counts: &mut AHashMap<CompactString, u64>,
    batch: &[String],
    total_pretokens: &mut u64,
) {
    let (local, pretokens) = count_batch(batch);
    for (piece, count) in local {
        *counts.entry(piece).or_insert(0) += count;
    }
    *total_pretokens += pretokens;
}

fn read_jsonl_text(line: &str, field: &str) -> Result<Option<String>, String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|err| format!("malformed JSONL row: {err}"))?;
    Ok(value
        .get(field)
        .and_then(|field| field.as_str())
        .filter(|text| !text.trim().is_empty())
        .map(ToOwned::to_owned))
}

fn collect_pretoken_counts(
    paths: &[PathBuf],
    jsonl_text_field: &str,
    buffer_size: usize,
    memory_limit: Option<u64>,
) -> Result<(AHashMap<CompactString, u64>, u64, u64, u64), String> {
    let mut counts = AHashMap::new();
    let mut documents = 0;
    let mut raw_bytes = 0;
    let mut pretokens = 0;
    let mut batch = Vec::with_capacity(buffer_size.max(1));

    for path in paths {
        let metadata = fs::metadata(path)
            .map_err(|err| format!("failed to stat input {}: {err}", path.display()))?;
        raw_bytes += metadata.len();
        if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            let file = File::open(path)
                .map_err(|err| format!("failed to open input {}: {err}", path.display()))?;
            let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
            for (line_idx, line) in reader.lines().enumerate() {
                let line = line.map_err(|err| {
                    format!(
                        "failed to read input {} line {}: {err}",
                        path.display(),
                        line_idx + 1
                    )
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                match read_jsonl_text(&line, jsonl_text_field) {
                    Ok(Some(text)) => {
                        batch.push(text);
                        documents += 1;
                    }
                    Ok(None) => {}
                    Err(err) => {
                        return Err(format!("{} at {}:{}", err, path.display(), line_idx + 1));
                    }
                }
                if batch.len() >= buffer_size.max(1) {
                    merge_batch_counts(&mut counts, &batch, &mut pretokens);
                    check_memory_limit(
                        memory_limit,
                        &format!(
                            "pretoken counting after {documents} documents and {} unique pretokens",
                            counts.len()
                        ),
                    )?;
                    batch.clear();
                }
            }
        } else {
            let text = fs::read_to_string(path)
                .map_err(|err| format!("failed to read input {}: {err}", path.display()))?;
            if !text.trim().is_empty() {
                batch.push(text);
                documents += 1;
            }
            if batch.len() >= buffer_size.max(1) {
                merge_batch_counts(&mut counts, &batch, &mut pretokens);
                check_memory_limit(
                    memory_limit,
                    &format!(
                        "pretoken counting after {documents} documents and {} unique pretokens",
                        counts.len()
                    ),
                )?;
                batch.clear();
            }
        }
    }

    if !batch.is_empty() {
        merge_batch_counts(&mut counts, &batch, &mut pretokens);
        check_memory_limit(
            memory_limit,
            &format!(
                "pretoken counting after {documents} documents and {} unique pretokens",
                counts.len()
            ),
        )?;
    }
    Ok((counts, documents, raw_bytes, pretokens))
}

fn build_pair_index(
    words: &[Word],
    weights: &[u64],
) -> (AHashMap<PairKey, u64>, AHashMap<PairKey, AHashSet<usize>>) {
    let pair_counts = words
        .par_iter()
        .zip(weights.par_iter())
        .map(|(word, weight)| {
            let mut local = AHashMap::with_capacity(word.pair_counts.len());
            for (pair, local_count) in &word.pair_counts {
                local.insert(*pair, u64::from(*local_count) * *weight);
            }
            local
        })
        .reduce(AHashMap::new, |mut left, right| {
            for (pair, count) in right {
                *left.entry(pair).or_insert(0) += count;
            }
            left
        });

    let mut positions = AHashMap::new();
    for (idx, word) in words.iter().enumerate() {
        for (pair, _) in &word.pair_counts {
            positions
                .entry(*pair)
                .or_insert_with(AHashSet::new)
                .insert(idx);
        }
    }
    (pair_counts, positions)
}

fn apply_pair_count_delta(pair_counts: &mut AHashMap<PairKey, u64>, pair: PairKey, delta: i64) {
    if delta == 0 {
        return;
    }
    if delta > 0 {
        *pair_counts.entry(pair).or_insert(0) += delta as u64;
        return;
    }
    let amount = (-delta) as u64;
    if let Some(count) = pair_counts.get_mut(&pair) {
        if *count <= amount {
            pair_counts.remove(&pair);
        } else {
            *count -= amount;
        }
    }
}

fn remove_position(positions: &mut AHashMap<PairKey, AHashSet<usize>>, pair: PairKey, idx: usize) {
    let should_remove = if let Some(set) = positions.get_mut(&pair) {
        set.remove(&idx);
        set.is_empty()
    } else {
        false
    };
    if should_remove {
        positions.remove(&pair);
    }
}

fn next_merge_job(
    heap: &mut OctonaryHeap<MergeJob>,
    pair_counts: &AHashMap<PairKey, u64>,
    min_frequency: u64,
) -> Option<MergeJob> {
    while let Some(job) = heap.pop() {
        let current_count = pair_counts.get(&job.pair).copied().unwrap_or(0);
        if current_count < min_frequency {
            continue;
        }
        if current_count == job.count {
            return Some(job);
        }
        heap.push(MergeJob {
            pair: job.pair,
            count: current_count,
        });
    }
    None
}

fn train_from_words(
    mut words: Vec<Word>,
    weights: Vec<u64>,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens_len: usize,
    memory_limit: Option<u64>,
) -> Result<Vec<SerializedMerge>, String> {
    let min_vocab = min_vocab_size(special_tokens_len);
    let (mut pair_counts, mut positions) = build_pair_index(&words, &weights);
    check_memory_limit(memory_limit, "initial pair index build")?;
    let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
    for (pair, count) in &pair_counts {
        if *count >= min_frequency {
            heap.push(MergeJob {
                pair: *pair,
                count: *count,
            });
        }
    }

    let mut merges = Vec::with_capacity(vocab_size.saturating_sub(min_vocab));
    let mut current_vocab_size = min_vocab;
    while current_vocab_size < vocab_size {
        let Some(job) = next_merge_job(&mut heap, &pair_counts, min_frequency) else {
            break;
        };
        let new_id = current_vocab_size as u32;
        let affected = positions.get(&job.pair).cloned().unwrap_or_default();
        if affected.is_empty() {
            pair_counts.remove(&job.pair);
            continue;
        }

        let mut changed = false;
        let mut changed_pairs = AHashSet::new();
        for idx in affected {
            if words[idx].pair_count(job.pair) == 0 {
                remove_position(&mut positions, job.pair, idx);
                continue;
            }
            let deltas = words[idx].merge_pair(job.pair, new_id);
            if deltas.is_empty() {
                remove_position(&mut positions, job.pair, idx);
                continue;
            }
            changed = true;
            let weight = weights[idx] as i64;
            for (pair, delta) in deltas {
                apply_pair_count_delta(&mut pair_counts, pair, i64::from(delta) * weight);
                if words[idx].pair_count(pair) > 0 {
                    positions
                        .entry(pair)
                        .or_insert_with(AHashSet::new)
                        .insert(idx);
                } else {
                    remove_position(&mut positions, pair, idx);
                }
                changed_pairs.insert(pair);
            }
        }

        if !changed {
            pair_counts.remove(&job.pair);
            continue;
        }

        let (left, right) = unpack_pair(job.pair);
        merges.push(SerializedMerge {
            left,
            right,
            id: new_id,
        });
        current_vocab_size += 1;
        if merges.len() % MEMORY_CHECK_INTERVAL_MERGES == 0 {
            check_memory_limit(
                memory_limit,
                &format!(
                    "merge training after {} of {} requested merges",
                    merges.len(),
                    vocab_size.saturating_sub(min_vocab)
                ),
            )?;
        }

        for pair in changed_pairs {
            if let Some(count) = pair_counts.get(&pair).copied() {
                if count >= min_frequency {
                    heap.push(MergeJob { pair, count });
                }
            }
        }
    }
    Ok(merges)
}

fn words_from_counts(
    counts: AHashMap<CompactString, u64>,
    special_tokens_len: usize,
) -> (Vec<Word>, Vec<u64>) {
    let pairs = counts
        .into_iter()
        .filter(|(piece, count)| !piece.is_empty() && *count > 0)
        .collect::<Vec<_>>();
    pairs
        .into_par_iter()
        .map(|(piece, count)| {
            let ids = piece
                .as_bytes()
                .iter()
                .map(|byte| byte_id(special_tokens_len, *byte))
                .collect::<Vec<_>>();
            (Word::new(ids), count)
        })
        .unzip()
}

pub fn train(config: TrainConfig) -> Result<TrainStats, String> {
    validate_special_tokens(&config.special_tokens)?;
    let min_vocab = min_vocab_size(config.special_tokens.len());
    if config.vocab_size < min_vocab {
        return Err(format!(
            "vocab_size must be at least {min_vocab} for byte-level BPE with {} special token(s)",
            config.special_tokens.len()
        ));
    }

    let start = Instant::now();
    let min_frequency = config.min_frequency.max(1);
    let memory_limit = effective_memory_limit_bytes(config.max_memory_bytes);
    let (counts, documents, raw_bytes, pretokens) = collect_pretoken_counts(
        &config.input_paths,
        &config.jsonl_text_field,
        config.buffer_size.max(1),
        memory_limit,
    )?;
    let unique_pretokens = counts.len();
    let (words, weights) = words_from_counts(counts, config.special_tokens.len());
    check_memory_limit(memory_limit, "pretoken materialization")?;
    let merges = train_from_words(
        words,
        weights,
        config.vocab_size,
        min_frequency,
        config.special_tokens.len(),
        memory_limit,
    )?;
    let actual_vocab_size = min_vocab + merges.len();
    let tokenizer = SerializedTokenizer {
        format: FORMAT.to_string(),
        version: VERSION,
        backend: BACKEND.to_string(),
        special_tokens: config.special_tokens,
        requested_vocab_size: config.vocab_size,
        min_frequency,
        pattern: GPT4_PATTERN.to_string(),
        merges,
    };
    let payload = serde_json::to_string_pretty(&tokenizer)
        .map_err(|err| format!("failed to serialize tokenizer: {err}"))?;
    if let Some(parent) = config.output_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create tokenizer directory {}: {err}",
                parent.display()
            )
        })?;
    }
    fs::write(&config.output_path, payload).map_err(|err| {
        format!(
            "failed to write tokenizer {}: {err}",
            config.output_path.display()
        )
    })?;

    Ok(TrainStats {
        documents,
        raw_bytes,
        pretokens,
        unique_pretokens,
        merges: actual_vocab_size - min_vocab,
        vocab_size: actual_vocab_size,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    })
}

fn read_tokenizer(path: &str) -> PyResult<RuntimeTokenizer> {
    let payload = fs::read_to_string(path)
        .map_err(|err| py_io_error(format!("failed to read tokenizer {path}: {err}")))?;
    let serialized: SerializedTokenizer = serde_json::from_str(&payload)
        .map_err(|err| py_value_error(format!("failed to parse tokenizer {path}: {err}")))?;
    RuntimeTokenizer::from_serialized(serialized).map_err(py_value_error)
}

#[pymethods]
impl RuntimeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_to_ids(text)
    }

    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|text| self.encode_to_ids(text))
            .collect()
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.decode_ids(&ids).map_err(py_value_error)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.special_to_id.get(token).copied()
    }

    fn get_vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }
}

#[pyfunction]
fn load_from_file(path: &str) -> PyResult<RuntimeTokenizer> {
    read_tokenizer(path)
}

#[pymodule]
fn _hackablebpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RuntimeTokenizer>()?;
    m.add_function(wrap_pyfunction!(load_from_file, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_merge_pair_updates_counts() {
        let mut word = Word::new(vec![1, 2, 3, 1, 2]);
        let mut deltas = AHashMap::new();
        for (pair, delta) in word.merge_pair(pack_pair(1, 2), 99) {
            *deltas.entry(pair).or_insert(0) += delta;
        }
        assert_eq!(word.ids, vec![99, 3, 99]);
        assert_eq!(word.pair_count(pack_pair(99, 3)), 1);
        assert_eq!(word.pair_count(pack_pair(3, 99)), 1);
        assert_eq!(word.pair_count(pack_pair(1, 2)), 0);
        assert_eq!(deltas.get(&pack_pair(1, 2)), Some(&-2));
    }

    #[test]
    fn learns_expected_simple_merge() {
        let mut counts = AHashMap::new();
        counts.insert(CompactString::from("aaaa"), 1);
        let (words, weights) = words_from_counts(counts, 2);
        let merges = train_from_words(words, weights, 2 + 256 + 1, 1, 2, None).unwrap();
        let a = byte_id(2, b'a');
        assert_eq!((merges[0].left, merges[0].right), (a, a));
    }
}
