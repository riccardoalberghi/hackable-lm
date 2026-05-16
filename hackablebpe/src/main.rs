use std::env;
use std::path::PathBuf;
use std::process;

use hackablebpe::{train, TrainConfig, DEFAULT_BUFFER_SIZE};

const EOS_TOKEN: &str = "<|endoftext|>";
const BOS_TOKEN: &str = "<|beginofsequence|>";

fn usage() -> &'static str {
    "usage: hackablebpe_train train --output PATH --vocab-size N [--jsonl-text-field text] [--min-frequency N] [--buffer-size N] [--max-memory-gib N] [--special-token TOKEN ...] INPUT..."
}

fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_gib(value: &str, flag: &str) -> Result<u64, String> {
    let gib = value
        .parse::<f64>()
        .map_err(|_| format!("{flag} must be a positive number, got {value:?}"))?;
    if !gib.is_finite() || gib <= 0.0 {
        return Err(format!("{flag} must be a positive number, got {value:?}"));
    }
    Ok((gib * 1024.0 * 1024.0 * 1024.0) as u64)
}

fn parse_train(args: &[String]) -> Result<TrainConfig, String> {
    let mut output_path = None;
    let mut vocab_size = None;
    let mut jsonl_text_field = "text".to_string();
    let mut min_frequency = 2;
    let mut buffer_size = DEFAULT_BUFFER_SIZE;
    let mut max_memory_bytes = match env::var("HACKABLEBPE_MAX_MEMORY_GIB") {
        Ok(value) => Some(parse_gib(&value, "HACKABLEBPE_MAX_MEMORY_GIB")?),
        Err(_) => None,
    };
    let mut special_tokens = Vec::new();
    let mut input_paths = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => output_path = Some(PathBuf::from(take_value(args, &mut i, "--output")?)),
            "--vocab-size" => {
                let value = take_value(args, &mut i, "--vocab-size")?;
                vocab_size = Some(
                    value
                        .parse()
                        .map_err(|_| format!("--vocab-size must be an integer, got {value:?}"))?,
                );
            }
            "--jsonl-text-field" => {
                jsonl_text_field = take_value(args, &mut i, "--jsonl-text-field")?;
            }
            "--min-frequency" => {
                let value = take_value(args, &mut i, "--min-frequency")?;
                min_frequency = value
                    .parse()
                    .map_err(|_| format!("--min-frequency must be an integer, got {value:?}"))?;
            }
            "--buffer-size" => {
                let value = take_value(args, &mut i, "--buffer-size")?;
                buffer_size = value
                    .parse()
                    .map_err(|_| format!("--buffer-size must be an integer, got {value:?}"))?;
            }
            "--max-memory-gib" => {
                let value = take_value(args, &mut i, "--max-memory-gib")?;
                max_memory_bytes = Some(parse_gib(&value, "--max-memory-gib")?);
            }
            "--special-token" => {
                special_tokens.push(take_value(args, &mut i, "--special-token")?);
            }
            "--" => {
                input_paths.extend(args[i + 1..].iter().map(PathBuf::from));
                break;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            path => input_paths.push(PathBuf::from(path)),
        }
        i += 1;
    }

    if special_tokens.is_empty() {
        special_tokens.push(EOS_TOKEN.to_string());
        special_tokens.push(BOS_TOKEN.to_string());
    }
    if input_paths.is_empty() {
        return Err("at least one input path is required".to_string());
    }

    Ok(TrainConfig {
        input_paths,
        output_path: output_path.ok_or_else(|| "--output is required".to_string())?,
        vocab_size: vocab_size.ok_or_else(|| "--vocab-size is required".to_string())?,
        jsonl_text_field,
        min_frequency,
        special_tokens,
        buffer_size,
        max_memory_bytes,
    })
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) != Some("train") {
        eprintln!("{}", usage());
        process::exit(2);
    }

    let config = match parse_train(&args[1..]) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("error: {err}\n{}", usage());
            process::exit(2);
        }
    };
    match train(config) {
        Ok(stats) => {
            let mib = stats.raw_bytes as f64 / (1024.0 * 1024.0);
            let mib_sec = if stats.elapsed_seconds > 0.0 {
                mib / stats.elapsed_seconds
            } else {
                0.0
            };
            eprintln!(
                "trained tokenizer: docs={} raw_mib={:.1} pretokens={} unique_pretokens={} merges={} vocab_size={} elapsed={:.2}s throughput={:.1} MiB/s",
                stats.documents,
                mib,
                stats.pretokens,
                stats.unique_pretokens,
                stats.merges,
                stats.vocab_size,
                stats.elapsed_seconds,
                mib_sec,
            );
        }
        Err(err) => {
            eprintln!("error: {err}");
            process::exit(1);
        }
    }
}
