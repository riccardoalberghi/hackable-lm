use super::*;

fn trained(texts: &[&str], vocab_size: usize) -> Tokenizer {
    let sequences = texts_to_sequences(
        texts
            .iter()
            .map(|text| text.to_string())
            .collect::<Vec<_>>(),
        1,
    );
    train_from_sequences(sequences, vocab_size, 1, vec!["<|endoftext|>".to_string()]).unwrap()
}

fn merge_sequence(sequence: &[u32], pair: Pair, new_id: u32) -> Vec<u32> {
    let mut merged = Vec::with_capacity(sequence.len());
    let mut i = 0;
    while i < sequence.len() {
        if i + 1 < sequence.len() && sequence[i] == pair.0 && sequence[i + 1] == pair.1 {
            merged.push(new_id);
            i += 2;
        } else {
            merged.push(sequence[i]);
            i += 1;
        }
    }
    merged
}

fn regex_pretokens(text: &str) -> Vec<Vec<u8>> {
    let mut pieces = Vec::new();
    for_each_regex_pretoken(text, |piece| pieces.push(piece.to_vec()));
    pieces
}

fn ascii_pretokens(text: &str) -> Vec<Vec<u8>> {
    let mut pieces = Vec::new();
    for_each_ascii_pretoken(text, |piece| pieces.push(piece.to_vec()));
    pieces
}

#[test]
fn ascii_pretokenizer_matches_regex() {
    let mut cases = vec![
        "",
        "hello world",
        "Hello, world!\n",
        "can't they'll I'VE",
        "1234567",
        "  leading and  internal",
        "\tindent\n\nnext",
        " .abc !?\r\nmore",
        "emoji-free ASCII only",
        "<|endoftext|>abc",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();

    let alphabet = [
        b'a', b'Z', b'1', b'2', b' ', b'\n', b'\r', b'.', b'!', b'\'', b'\t',
    ];
    for len in 0..=4 {
        let mut indexes = vec![0usize; len];
        loop {
            cases.push(
                indexes
                    .iter()
                    .map(|idx| alphabet[*idx])
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(char::from)
                    .collect::<String>(),
            );
            let mut pos = 0;
            while pos < len {
                indexes[pos] += 1;
                if indexes[pos] < alphabet.len() {
                    break;
                }
                indexes[pos] = 0;
                pos += 1;
            }
            if pos == len {
                break;
            }
        }
    }

    for case in cases {
        assert_eq!(ascii_pretokens(&case), regex_pretokens(&case), "{case:?}");
    }
}

fn naive_best_pair(sequences: &[Vec<u32>], min_frequency: u64) -> Option<PairKey> {
    let mut counts = AHashMap::new();
    for sequence in sequences {
        for pair in sequence
            .windows(2)
            .map(|window| pack_pair(window[0], window[1]))
        {
            *counts.entry(pair).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count >= min_frequency)
        .max_by(|(left_pair, left_count), (right_pair, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_pair.cmp(left_pair))
        })
        .map(|(pair, _)| pair)
}

fn naive_merges(
    mut sequences: Vec<Vec<u32>>,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens_len: usize,
) -> Vec<SerializedMerge> {
    let mut merges = Vec::new();
    let mut current_vocab_size = min_vocab_size(special_tokens_len);
    while current_vocab_size < vocab_size {
        let Some(pair) = naive_best_pair(&sequences, min_frequency) else {
            break;
        };
        let new_id = current_vocab_size as u32;
        let mut changed = false;
        for sequence in sequences.iter_mut() {
            let merged = merge_sequence(sequence, unpack_pair(pair), new_id);
            if merged.len() != sequence.len() {
                changed = true;
            }
            *sequence = merged;
        }
        if !changed {
            break;
        }
        let (left, right) = unpack_pair(pair);
        merges.push(SerializedMerge {
            left,
            right,
            id: new_id,
        });
        current_vocab_size += 1;
    }
    merges
}

#[test]
fn round_trips_utf8() {
    let tok = trained(&["hello hello", "cafe \u{00e9} \u{1f680}"], 280);
    let text = "hello cafe \u{00e9} \u{1f680}";
    let ids = tok.encode_to_ids(text);
    assert_eq!(tok.decode_ids(&ids).unwrap(), text);
}

#[test]
fn learns_expected_simple_merge() {
    let tok = trained(&["aaaa"], 258);
    let first_merge = tok.merges.first().unwrap();
    let a = byte_id(tok.special_tokens.len(), b'a');
    assert_eq!((first_merge.left, first_merge.right), (a, a));
}

#[test]
fn special_tokens_are_encoded_and_skipped_on_decode() {
    let tok = trained(&["hello"], 260);
    let ids = tok.encode_to_ids("a<|endoftext|>b");
    assert!(ids.contains(&0));
    assert_eq!(tok.decode_ids(&ids).unwrap(), "ab");
}

#[test]
fn word_merge_pair_updates_deltas_incrementally() {
    let mut word = Word::new(vec![1, 2, 3, 1, 2]);
    let deltas = word
        .merge_pair(pack_pair(1, 2), 99)
        .into_iter()
        .collect::<AHashMap<_, _>>();
    assert_eq!(word.ids, vec![99, 3, 99]);
    assert_eq!(word.pair_count(pack_pair(99, 3)), 1);
    assert_eq!(word.pair_count(pack_pair(3, 99)), 1);
    assert_eq!(word.pair_count(pack_pair(1, 2)), 0);
    assert_eq!(deltas.get(&pack_pair(1, 2)), Some(&-2));
    assert_eq!(deltas.get(&pack_pair(2, 3)), Some(&-1));
    assert_eq!(deltas.get(&pack_pair(3, 1)), Some(&-1));
    assert_eq!(deltas.get(&pack_pair(99, 3)), Some(&1));
    assert_eq!(deltas.get(&pack_pair(3, 99)), Some(&1));
}

#[test]
fn heap_encoder_matches_scan_encoder() {
    let tok = trained(
        &[
            "antidisestablishmentarianism antidisestablishmentarianism",
            "counterrevolutionarycounterrevolutionary",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ],
        360,
    );
    let mut scan = "antidisestablishmentarianismcounterrevolutionaryaaaaaaaa"
        .as_bytes()
        .iter()
        .map(|byte| byte_id(tok.special_tokens.len(), *byte))
        .collect::<Vec<_>>();
    let mut heap = scan.clone();
    tok.apply_merges_scan(&mut scan);
    tok.apply_merges_heap(&mut heap);
    assert_eq!(heap, scan);
}

#[test]
fn cached_batch_encoding_matches_uncached() {
    let tok = trained(
        &[
            "hello hello repeated repeated repeated",
            "world world repeated repeated repeated",
        ],
        300,
    );
    let texts = vec![
        "hello world repeated repeated".to_string(),
        "world hello repeated repeated".to_string(),
        "hello<|endoftext|>world repeated".to_string(),
    ];
    let cached = tok
        .encode_batch(texts.clone())
        .into_iter()
        .map(|encoding| encoding.ids)
        .collect::<Vec<_>>();
    let uncached = tok
        .encode_batch_uncached(texts)
        .into_iter()
        .map(|encoding| encoding.ids)
        .collect::<Vec<_>>();
    assert_eq!(cached, uncached);
}

#[test]
fn deterministic_tie_breaking_prefers_lower_pair() {
    let special_len = 1;
    let sequences = vec![vec![
        byte_id(special_len, b'a'),
        byte_id(special_len, b'b'),
        byte_id(special_len, b'c'),
        byte_id(special_len, b'd'),
    ]];
    let tok = train_from_sequences(
        sequences,
        min_vocab_size(special_len) + 1,
        1,
        vec!["<|endoftext|>".to_string()],
    )
    .unwrap();
    let first = tok.merges.first().unwrap();
    assert_eq!(
        (first.left, first.right),
        (byte_id(special_len, b'a'), byte_id(special_len, b'b'))
    );
}

#[test]
fn incremental_training_matches_naive_recounting() {
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let sequences = texts_to_sequences(
        vec![
            "abababab".to_string(),
            "abc abc abc".to_string(),
            "cafe \u{00e9} cafe \u{00e9}".to_string(),
            "abababab".to_string(),
        ],
        special_tokens.len(),
    );
    let tok = train_from_sequences(sequences.clone(), 280, 1, special_tokens.clone()).unwrap();
    let expected = naive_merges(sequences, 280, 1, special_tokens.len());
    assert_eq!(tok.merges, expected);
}
