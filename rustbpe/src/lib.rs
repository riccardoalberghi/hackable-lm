use std::cmp::Ordering;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use ahash::{AHashMap, AHashSet};
use dary_heap::OctonaryHeap;
use memchr::memchr;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};

type Pair = (u32, u32);
type PairKey = u64;
type MergeValue = u64;

const FORMAT: &str = "hackable_lm_rustbpe";
const VERSION: u32 = 1;
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+";
const HEAP_MERGE_MIN_LEN: usize = 24;
const MAX_ENCODE_CACHE_ENTRIES: usize = 65_536;
const MAX_CACHED_PIECE_BYTES: usize = 96;

// Inspired by Andrej Karpathy's rustbpe (https://github.com/karpathy/rustbpe):
// start from byte IDs, learn frequent pair merges, and apply merge ranks at
// encode time. This crate extends that shape with local byte-run
// pre-tokenization, JSON serialization, special-token handling, batch encoding,
// and the Python API needed by hackable-lm.

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct SerializedMerge {
    left: u32,
    right: u32,
    id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedTokenizer {
    format: String,
    version: u32,
    special_tokens: Vec<String>,
    merges: Vec<SerializedMerge>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeJob {
    pair: PairKey,
    count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncodeJob {
    rank: usize,
    pos: usize,
    left: u32,
    right: u32,
    new_id: u32,
}

impl Ord for EncodeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for EncodeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

#[pyclass]
#[derive(Debug, Clone)]
pub struct Encoding {
    #[pyo3(get)]
    ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncodePart {
    Piece(usize),
    Special(u32),
}

#[derive(Debug, Clone, Default)]
struct DocumentPieces<'a> {
    parts: Vec<EncodePart>,
    pieces: Vec<&'a [u8]>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Tokenizer {
    special_tokens: Vec<String>,
    special_to_id: AHashMap<String, u32>,
    special_patterns: Vec<(String, u32)>,
    merges: Vec<SerializedMerge>,
    merge_ranks: AHashMap<PairKey, MergeValue>,
    id_to_bytes: Vec<Option<Vec<u8>>>,
    encode_cache: Arc<Mutex<AHashMap<Vec<u8>, Vec<u32>>>>,
}

#[derive(Debug, Clone)]
struct Word {
    ids: Vec<u32>,
    pair_counts: AHashMap<PairKey, u32>,
}

impl Word {
    fn new(ids: Vec<u32>) -> Self {
        let pair_counts = count_pairs_in_ids(&ids);
        Self { ids, pair_counts }
    }

    fn pair_count(&self, pair: PairKey) -> u32 {
        self.pair_counts.get(&pair).copied().unwrap_or(0)
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

        let mut aggregated = AHashMap::new();
        for (delta_pair, delta) in deltas {
            *aggregated.entry(delta_pair).or_insert(0) += delta;
        }
        for (&delta_pair, &delta) in aggregated.iter() {
            if delta == 0 {
                continue;
            }
            if delta < 0 {
                let amount = (-delta) as u32;
                let should_remove = if let Some(count) = self.pair_counts.get_mut(&delta_pair) {
                    if *count <= amount {
                        true
                    } else {
                        *count -= amount;
                        false
                    }
                } else {
                    false
                };
                if should_remove {
                    self.pair_counts.remove(&delta_pair);
                }
            } else {
                *self.pair_counts.entry(delta_pair).or_insert(0) += delta as u32;
            }
        }
        self.ids = out;
        aggregated
            .into_iter()
            .filter(|(_, delta)| *delta != 0)
            .collect()
    }
}

fn py_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn py_io_error(message: impl Into<String>) -> PyErr {
    PyIOError::new_err(message.into())
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

fn byte_offset(special_tokens_len: usize) -> u32 {
    special_tokens_len as u32
}

fn byte_id(special_tokens_len: usize, byte: u8) -> u32 {
    byte_offset(special_tokens_len) + u32::from(byte)
}

fn min_vocab_size(special_tokens_len: usize) -> usize {
    special_tokens_len + 256
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

fn gpt4_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(GPT4_PATTERN).expect("valid GPT-4 pre-tokenizer regex"))
}

fn is_ascii_letter(byte: u8) -> bool {
    byte.is_ascii_alphabetic()
}

fn is_ascii_number(byte: u8) -> bool {
    byte.is_ascii_digit()
}

fn is_ascii_letter_or_number(byte: u8) -> bool {
    is_ascii_letter(byte) || is_ascii_number(byte)
}

fn is_crlf(byte: u8) -> bool {
    byte == b'\r' || byte == b'\n'
}

fn is_ascii_non_word_prefix(byte: u8) -> bool {
    !is_crlf(byte) && !is_ascii_letter_or_number(byte)
}

fn is_ascii_punctuation_piece(byte: u8) -> bool {
    !byte.is_ascii_whitespace() && !is_ascii_letter_or_number(byte)
}

fn contraction_len(bytes: &[u8], i: usize) -> Option<usize> {
    if bytes.get(i).copied() != Some(b'\'') {
        return None;
    }
    let b1 = bytes.get(i + 1).copied()?.to_ascii_lowercase();
    if matches!(b1, b's' | b'd' | b'm' | b't') {
        return Some(2);
    }
    let b2 = bytes.get(i + 2).copied()?.to_ascii_lowercase();
    match (b1, b2) {
        (b'l', b'l') | (b'v', b'e') | (b'r', b'e') => Some(3),
        _ => None,
    }
}

fn for_each_regex_pretoken<'a, F>(text: &'a str, mut f: F)
where
    F: FnMut(&'a [u8]),
{
    for mat in gpt4_regex().find_iter(text) {
        let piece = mat.as_str().as_bytes();
        if !piece.is_empty() {
            f(piece);
        }
    }
}

fn for_each_ascii_pretoken<'a, F>(text: &'a str, mut f: F)
where
    F: FnMut(&'a [u8]),
{
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(len) = contraction_len(bytes, i) {
            f(&bytes[i..i + len]);
            i += len;
            continue;
        }

        let start = i;
        if is_ascii_letter(bytes[i]) {
            i += 1;
            while i < bytes.len() && is_ascii_letter(bytes[i]) {
                i += 1;
            }
            f(&bytes[start..i]);
            continue;
        }
        if is_ascii_non_word_prefix(bytes[i])
            && i + 1 < bytes.len()
            && is_ascii_letter(bytes[i + 1])
        {
            i += 2;
            while i < bytes.len() && is_ascii_letter(bytes[i]) {
                i += 1;
            }
            f(&bytes[start..i]);
            continue;
        }
        if is_ascii_number(bytes[i]) {
            i += 1;
            while i < bytes.len() && i - start < 3 && is_ascii_number(bytes[i]) {
                i += 1;
            }
            f(&bytes[start..i]);
            continue;
        }

        let punct_start = if bytes[i] == b' '
            && i + 1 < bytes.len()
            && is_ascii_punctuation_piece(bytes[i + 1])
        {
            i += 1;
            i
        } else {
            i
        };
        if punct_start < bytes.len() && is_ascii_punctuation_piece(bytes[punct_start]) {
            i = punct_start + 1;
            while i < bytes.len() && is_ascii_punctuation_piece(bytes[i]) {
                i += 1;
            }
            while i < bytes.len() && is_crlf(bytes[i]) {
                i += 1;
            }
            f(&bytes[start..i]);
            continue;
        }

        if bytes[i].is_ascii_whitespace() {
            let mut j = i;
            let mut last_newline_end = None;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                if is_crlf(bytes[j]) {
                    last_newline_end = Some(j + 1);
                }
                j += 1;
            }
            if let Some(end) = last_newline_end {
                f(&bytes[i..end]);
                i = end;
            } else {
                f(&bytes[i..j]);
                i = j;
            }
            continue;
        }

        i += 1;
        f(&bytes[start..i]);
    }
}

fn for_each_pretoken<'a, F>(text: &'a str, f: F)
where
    F: FnMut(&'a [u8]),
{
    if text.is_ascii() {
        for_each_ascii_pretoken(text, f);
    } else {
        for_each_regex_pretoken(text, f);
    }
}

fn piece_to_ids(piece: &[u8], special_tokens_len: usize) -> Vec<u32> {
    piece
        .iter()
        .map(|byte| byte_id(special_tokens_len, *byte))
        .collect()
}

fn push_pretokenized_sequences(
    text: &str,
    special_tokens_len: usize,
    sequences: &mut Vec<Vec<u32>>,
) {
    for_each_pretoken(text, |piece| {
        if !piece.is_empty() {
            sequences.push(piece_to_ids(piece, special_tokens_len));
        }
    });
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

impl Tokenizer {
    fn from_parts(
        special_tokens: Vec<String>,
        merges: Vec<SerializedMerge>,
    ) -> Result<Self, String> {
        validate_special_tokens(&special_tokens)?;
        let id_to_bytes = build_id_to_bytes(special_tokens.len(), &merges)?;
        let mut special_to_id = AHashMap::with_capacity(special_tokens.len());
        for (id, token) in special_tokens.iter().enumerate() {
            special_to_id.insert(token.clone(), id as u32);
        }
        let mut special_patterns = special_tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect::<Vec<_>>();
        special_patterns.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

        let mut merge_ranks = AHashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
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
            special_tokens,
            special_to_id,
            special_patterns,
            merges,
            merge_ranks,
            id_to_bytes,
            encode_cache: Arc::new(Mutex::new(AHashMap::new())),
        })
    }

    fn base(special_tokens: Vec<String>) -> Result<Self, String> {
        Self::from_parts(special_tokens, Vec::new())
    }

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
        Self::from_parts(serialized.special_tokens, serialized.merges)
    }

    fn to_serialized(&self) -> SerializedTokenizer {
        SerializedTokenizer {
            format: FORMAT.to_string(),
            version: VERSION,
            special_tokens: self.special_tokens.clone(),
            merges: self.merges.clone(),
        }
    }

    fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() {
            return Vec::new();
        }
        let mut ids = bytes
            .iter()
            .map(|byte| byte_id(self.special_tokens.len(), *byte))
            .collect::<Vec<_>>();
        self.apply_merges(&mut ids);
        ids
    }

    fn encode_bytes_cached(
        &self,
        bytes: &[u8],
        cache: &mut AHashMap<Vec<u8>, Vec<u32>>,
    ) -> Vec<u32> {
        if bytes.len() <= MAX_CACHED_PIECE_BYTES {
            if let Some(ids) = cache.get(bytes) {
                return ids.clone();
            }
            let ids = self.encode_bytes(bytes);
            if cache.len() < MAX_ENCODE_CACHE_ENTRIES {
                cache.insert(bytes.to_vec(), ids.clone());
            }
            ids
        } else {
            self.encode_bytes(bytes)
        }
    }

    fn encode_regular_text_into(
        &self,
        text: &str,
        ids: &mut Vec<u32>,
        mut cache: Option<&mut AHashMap<Vec<u8>, Vec<u32>>>,
    ) {
        for_each_pretoken(text, |piece| {
            if let Some(cache) = cache.as_deref_mut() {
                ids.extend(self.encode_bytes_cached(piece, cache));
            } else {
                ids.extend(self.encode_bytes(piece));
            }
        });
    }

    fn next_special_match(&self, text: &str, cursor: usize) -> Option<(usize, usize, u32)> {
        match self.special_patterns.as_slice() {
            [] => None,
            [(pattern, id)] => {
                let haystack = text[cursor..].as_bytes();
                let needle = pattern.as_bytes();
                let first = needle[0];
                let mut offset = 0;
                while let Some(pos) = memchr(first, &haystack[offset..]) {
                    let start = offset + pos;
                    if haystack[start..].starts_with(needle) {
                        return Some((cursor + start, needle.len(), *id));
                    }
                    offset = start + 1;
                }
                None
            }
            patterns => patterns
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
                }),
        }
    }

    fn apply_merges(&self, ids: &mut Vec<u32>) {
        if ids.len() < 2 || self.merge_ranks.is_empty() {
            return;
        }
        if ids.len() >= HEAP_MERGE_MIN_LEN {
            self.apply_merges_heap(ids);
        } else {
            self.apply_merges_scan(ids);
        }
    }

    fn apply_merges_scan(&self, ids: &mut Vec<u32>) {
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

    fn push_encode_job(
        &self,
        heap: &mut OctonaryHeap<EncodeJob>,
        values: &[u32],
        next: &[Option<usize>],
        pos: usize,
    ) {
        let Some(right_pos) = next[pos] else {
            return;
        };
        let pair = (values[pos], values[right_pos]);
        if let Some(value) = self.merge_ranks.get(&pack_pair(pair.0, pair.1)).copied() {
            heap.push(EncodeJob {
                rank: merge_rank(value),
                pos,
                left: pair.0,
                right: pair.1,
                new_id: merge_new_id(value),
            });
        }
    }

    fn apply_merges_heap(&self, ids: &mut Vec<u32>) {
        let n = ids.len();
        let mut values = ids.clone();
        let mut prev = (0..n)
            .map(|idx| if idx > 0 { Some(idx - 1) } else { None })
            .collect::<Vec<_>>();
        let mut next = (0..n)
            .map(|idx| (idx + 1 < n).then_some(idx + 1))
            .collect::<Vec<_>>();
        let mut active = vec![true; n];
        let mut heap = OctonaryHeap::with_capacity(n.saturating_sub(1));
        for pos in 0..n.saturating_sub(1) {
            self.push_encode_job(&mut heap, &values, &next, pos);
        }

        while let Some(job) = heap.pop() {
            if !active[job.pos] || values[job.pos] != job.left {
                continue;
            }
            let Some(right_pos) = next[job.pos] else {
                continue;
            };
            if !active[right_pos] || values[right_pos] != job.right {
                continue;
            }

            values[job.pos] = job.new_id;
            active[right_pos] = false;
            let right_next = next[right_pos];
            next[job.pos] = right_next;
            if let Some(after) = right_next {
                prev[after] = Some(job.pos);
            }

            if let Some(left_pos) = prev[job.pos] {
                self.push_encode_job(&mut heap, &values, &next, left_pos);
            }
            self.push_encode_job(&mut heap, &values, &next, job.pos);
        }

        ids.clear();
        let mut pos = Some(0);
        while let Some(idx) = pos {
            if active[idx] {
                ids.push(values[idx]);
            }
            pos = next[idx];
        }
    }

    fn encode_to_ids(&self, text: &str) -> Vec<u32> {
        self.encode_to_ids_inner(text, None)
    }

    fn encode_to_ids_cached(
        &self,
        text: &str,
        cache: &mut AHashMap<Vec<u8>, Vec<u32>>,
    ) -> Vec<u32> {
        self.encode_to_ids_inner(text, Some(cache))
    }

    fn encode_to_ids_inner(
        &self,
        text: &str,
        mut cache: Option<&mut AHashMap<Vec<u8>, Vec<u32>>>,
    ) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut ids = Vec::new();
        let mut cursor = 0;
        while cursor < text.len() {
            let matched = self.next_special_match(text, cursor);
            if let Some((start, special_len, special_id)) = matched {
                self.encode_regular_text_into(&text[cursor..start], &mut ids, cache.as_deref_mut());
                ids.push(special_id);
                cursor = start + special_len;
            } else {
                self.encode_regular_text_into(&text[cursor..], &mut ids, cache.as_deref_mut());
                break;
            }
        }
        ids
    }

    fn append_regular_text_parts<'a>(
        &self,
        text: &'a str,
        parts: &mut Vec<EncodePart>,
        pieces: &mut Vec<&'a [u8]>,
    ) {
        for_each_pretoken(text, |piece| {
            let idx = pieces.len();
            pieces.push(piece);
            parts.push(EncodePart::Piece(idx));
        });
    }

    fn collect_encode_parts<'a>(&self, text: &'a str) -> DocumentPieces<'a> {
        let mut doc = DocumentPieces::default();
        if text.is_empty() {
            return doc;
        }

        let mut cursor = 0;
        while cursor < text.len() {
            let matched = self.next_special_match(text, cursor);
            if let Some((start, special_len, special_id)) = matched {
                self.append_regular_text_parts(
                    &text[cursor..start],
                    &mut doc.parts,
                    &mut doc.pieces,
                );
                doc.parts.push(EncodePart::Special(special_id));
                cursor = start + special_len;
            } else {
                self.append_regular_text_parts(&text[cursor..], &mut doc.parts, &mut doc.pieces);
                break;
            }
        }
        doc
    }

    fn encode_batch_shared_cache(&self, texts: Vec<String>) -> Vec<Encoding> {
        if texts.is_empty() {
            return Vec::new();
        }

        let docs = texts
            .par_iter()
            .map(|text| self.collect_encode_parts(text))
            .collect::<Vec<_>>();

        let total_pieces = docs.iter().map(|doc| doc.pieces.len()).sum();
        let mut unique_by_piece = AHashMap::with_capacity(total_pieces);
        let mut unique_pieces = Vec::new();
        let mut doc_piece_maps = Vec::with_capacity(docs.len());
        for doc in &docs {
            let mut local_map = Vec::with_capacity(doc.pieces.len());
            for &piece in &doc.pieces {
                let unique_idx = if let Some(idx) = unique_by_piece.get(piece).copied() {
                    idx
                } else {
                    let idx = unique_pieces.len();
                    let owned = piece.to_vec();
                    unique_by_piece.insert(owned.clone(), idx);
                    unique_pieces.push(owned);
                    idx
                };
                local_map.push(unique_idx);
            }
            doc_piece_maps.push(local_map);
        }

        let mut encoded_pieces = vec![Vec::new(); unique_pieces.len()];
        let mut missing = Vec::new();
        if let Ok(cache) = self.encode_cache.lock() {
            for (idx, piece) in unique_pieces.iter().enumerate() {
                if piece.len() <= MAX_CACHED_PIECE_BYTES {
                    if let Some(ids) = cache.get(piece) {
                        encoded_pieces[idx] = ids.clone();
                        continue;
                    }
                }
                missing.push(idx);
            }
        } else {
            missing.extend(0..unique_pieces.len());
        }

        let missing_encoded = missing
            .par_iter()
            .map(|idx| (*idx, self.encode_bytes(&unique_pieces[*idx])))
            .collect::<Vec<_>>();

        for (idx, ids) in &missing_encoded {
            encoded_pieces[*idx] = ids.clone();
        }

        if let Ok(mut cache) = self.encode_cache.lock() {
            for (idx, ids) in missing_encoded {
                let piece = &unique_pieces[idx];
                if piece.len() <= MAX_CACHED_PIECE_BYTES && cache.len() < MAX_ENCODE_CACHE_ENTRIES {
                    cache.insert(piece.clone(), ids);
                }
            }
        }

        let ids_by_piece = encoded_pieces;
        docs.into_par_iter()
            .zip(doc_piece_maps.into_par_iter())
            .map(|(doc, local_piece_map)| {
                let mut ids = Vec::with_capacity(doc.parts.len());
                for part in doc.parts {
                    match part {
                        EncodePart::Piece(local_idx) => {
                            let unique_idx = local_piece_map[local_idx];
                            ids.extend_from_slice(&ids_by_piece[unique_idx]);
                        }
                        EncodePart::Special(id) => ids.push(id),
                    }
                }
                Encoding { ids }
            })
            .collect()
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

    fn save_to_path(&self, path: &str) -> PyResult<()> {
        let serialized = self.to_serialized();
        let payload = serde_json::to_string_pretty(&serialized)
            .map_err(|err| py_value_error(format!("failed to serialize tokenizer: {err}")))?;
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                py_io_error(format!(
                    "failed to create tokenizer directory {}: {err}",
                    parent.display()
                ))
            })?;
        }
        fs::write(path, payload).map_err(|err| {
            py_io_error(format!(
                "failed to write tokenizer {}: {err}",
                path.display()
            ))
        })
    }
}

fn count_pairs_in_ids(ids: &[u32]) -> AHashMap<PairKey, u32> {
    let mut counts = AHashMap::new();
    for pair in ids.windows(2).map(|window| pack_pair(window[0], window[1])) {
        *counts.entry(pair).or_insert(0) += 1;
    }
    counts
}

fn deduplicate_sequences(sequences: Vec<Vec<u32>>) -> (Vec<Vec<u32>>, Vec<u64>) {
    let mut ids_by_sequence: AHashMap<Vec<u32>, usize> = AHashMap::new();
    let mut unique = Vec::new();
    let mut weights = Vec::new();
    for sequence in sequences {
        if sequence.is_empty() {
            continue;
        }
        if let Some(idx) = ids_by_sequence.get(&sequence).copied() {
            weights[idx] += 1;
        } else {
            let idx = unique.len();
            ids_by_sequence.insert(sequence.clone(), idx);
            unique.push(sequence);
            weights.push(1);
        }
    }
    (unique, weights)
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
            for (pair, local_count) in word.pair_counts.iter() {
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
        for pair in word.pair_counts.keys().copied() {
            positions
                .entry(pair)
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

fn train_from_sequences(
    sequences: Vec<Vec<u32>>,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
) -> Result<Tokenizer, String> {
    validate_special_tokens(&special_tokens)?;
    let min_vocab = min_vocab_size(special_tokens.len());
    if vocab_size < min_vocab {
        return Err(format!(
            "vocab_size must be at least {min_vocab} for byte-level BPE with {} special token(s)",
            special_tokens.len()
        ));
    }
    let min_frequency = min_frequency.max(1);
    let (sequences, weights) = deduplicate_sequences(sequences);
    let mut words = sequences.into_par_iter().map(Word::new).collect::<Vec<_>>();
    let (mut pair_counts, mut positions) = build_pair_index(&words, &weights);
    let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
    for (pair, count) in pair_counts.iter() {
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
        let mut changed_pairs = AHashSet::new();
        let mut changed = false;
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
            let weight = weights[idx];
            for (pair, delta) in deltas {
                apply_pair_count_delta(&mut pair_counts, pair, i64::from(delta) * weight as i64);
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
            break;
        }
        let (left, right) = unpack_pair(job.pair);
        merges.push(SerializedMerge {
            left,
            right,
            id: new_id,
        });
        current_vocab_size += 1;

        for pair in changed_pairs {
            if let Some(count) = pair_counts.get(&pair).copied() {
                if count >= min_frequency {
                    heap.push(MergeJob { pair, count });
                }
            }
        }
    }
    Tokenizer::from_parts(special_tokens, merges)
}

fn texts_to_sequences<I>(texts: I, special_tokens_len: usize) -> Vec<Vec<u32>>
where
    I: IntoIterator<Item = String>,
{
    texts
        .into_iter()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|text| {
            let mut local = Vec::new();
            push_pretokenized_sequences(text, special_tokens_len, &mut local);
            local
        })
        .reduce(Vec::new, |mut left, right| {
            left.extend(right);
            left
        })
}

fn read_tokenizer(path: &str) -> PyResult<Tokenizer> {
    let payload = fs::read_to_string(path)
        .map_err(|err| py_io_error(format!("failed to read tokenizer {path}: {err}")))?;
    let serialized: SerializedTokenizer = serde_json::from_str(&payload)
        .map_err(|err| py_value_error(format!("failed to parse tokenizer {path}: {err}")))?;
    Tokenizer::from_serialized(serialized).map_err(py_value_error)
}

#[pymethods]
impl Tokenizer {
    #[new]
    #[pyo3(signature = (special_tokens=None))]
    fn py_new(special_tokens: Option<Vec<String>>) -> PyResult<Self> {
        Self::base(special_tokens.unwrap_or_default()).map_err(py_value_error)
    }

    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        read_tokenizer(path)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.save_to_path(path)
    }

    fn encode(&self, text: &str) -> Encoding {
        let mut cache = AHashMap::new();
        Encoding {
            ids: self.encode_to_ids_cached(text, &mut cache),
        }
    }

    fn encode_batch(&self, texts: Vec<String>) -> Vec<Encoding> {
        self.encode_batch_shared_cache(texts)
    }

    fn encode_batch_uncached(&self, texts: Vec<String>) -> Vec<Encoding> {
        texts
            .iter()
            .map(|text| Encoding {
                ids: self.encode_to_ids(text),
            })
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
fn train_from_texts(
    texts: Vec<String>,
    output_path: &str,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
) -> PyResult<()> {
    let sequences = texts_to_sequences(texts, special_tokens.len());
    let tokenizer = train_from_sequences(sequences, vocab_size, min_frequency, special_tokens)
        .map_err(py_value_error)?;
    tokenizer.save_to_path(output_path)
}

#[pyfunction]
fn train_from_iterator(
    texts: &Bound<'_, PyAny>,
    output_path: &str,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
) -> PyResult<()> {
    let raw_texts = texts
        .try_iter()?
        .map(|item| item?.extract())
        .collect::<PyResult<Vec<String>>>()?;
    let sequences = texts_to_sequences(raw_texts, special_tokens.len());
    let tokenizer = train_from_sequences(sequences, vocab_size, min_frequency, special_tokens)
        .map_err(py_value_error)?;
    tokenizer.save_to_path(output_path)
}

#[pyfunction]
fn load_from_file(path: &str) -> PyResult<Tokenizer> {
    read_tokenizer(path)
}

#[pymodule]
fn _hackable_lm_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Encoding>()?;
    m.add_class::<Tokenizer>()?;
    m.add_function(wrap_pyfunction!(train_from_texts, m)?)?;
    m.add_function(wrap_pyfunction!(train_from_iterator, m)?)?;
    m.add_function(wrap_pyfunction!(load_from_file, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
