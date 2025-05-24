use ahash::{HashMap, HashMapExt};
use rayon::prelude::*;
use std::{
    collections::HashSet,
    fmt::Debug,
    fs::File,
    hash::Hash,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};

const HASHER: ahash::RandomState = ahash::RandomState::with_seeds(0, 0, 0, 0);

#[derive(Debug)]
pub struct BPETokenizer {
    // Mapping of token pairs to their merge order (lower value indicates higher priority)
    merges: HashMap<(u16, u16), u16>,
    // Vocabulary: token bytes to token id
    vocab: HashMap<Box<[u8]>, u16>,
    // Inverse vocabulary: token id to token bytes
    inv_vocab: Box<[Box<[u8]>]>,
}

pub struct BPESet {
    tokenizer: BPETokenizer,
    set: HashSet<Box<[u16]>>,
}

impl BPESet {
    pub fn new<I: IntoIterator<Item = B>, B: AsRef<[u8]>>(
        train_data: I,
        min_bpe_freq: usize,
        max_token_id: u16,
    ) -> BPESet {
        let tokenizer = BPETokenizer::build(train_data, min_bpe_freq, max_token_id);
        let set = HashSet::new();
        BPESet { tokenizer, set }
    }
    pub fn with_vocab(path: &Path) -> std::io::Result<BPESet> {
        let tokenizer = BPETokenizer::load(path)?;
        let set = HashSet::new();
        Ok(BPESet { tokenizer, set })
    }
    pub fn save_vocab(&self, path: &Path) -> std::io::Result<()> {
        self.tokenizer.save(path)?;
        Ok(())
    }
    pub fn insert(&mut self, bytes: &[u8]) {
        let tokens = self.tokenizer.encode(bytes);
        self.set.insert(tokens.into_boxed_slice());
    }
    pub fn contains(&self, bytes: &[u8]) -> bool {
        let tokens = self.tokenizer.encode(bytes);
        self.set.contains(&*tokens)
    }
    pub fn memory_usage(&self) -> usize {
        self.tokenizer.memory_usage()
            + self.set.capacity() * size_of::<Box<[u16]>>()
            + self.set.iter().map(|item| item.len() * 2).sum::<usize>()
    }
}

fn run<I: Send + Sync + 'static, K: Send + Sync + 'static>(
    iter: &[I],
    init: impl Fn() -> K,
    closure: impl Fn(&mut K, &I) + Send + 'static + Clone,
    merge: impl Fn(K, K) -> K,
) -> K {
    let mut threads = Vec::new();
    let amount = iter.len();
    // safe as long as code don't panic
    let iter: &'static [I] = unsafe { std::slice::from_raw_parts(iter.as_ptr(), amount) };
    let cpus = num_cpus::get();
    let mut chunks = iter.chunks((amount + cpus) / cpus);
    let (tx, rx) = std::sync::mpsc::channel::<K>();
    for _ in 0..cpus {
        let mut buffer = init();
        let chunk = chunks.next().unwrap();
        let tx = tx.clone();
        let closure = closure.clone();
        let handle = std::thread::spawn(move || {
            for item in chunk {
                closure(&mut buffer, item);
            }
            tx.send(buffer).unwrap();
        });
        threads.push(handle);
    }
    (0..cpus).map(|_| rx.recv().unwrap()).fold(init(), merge)
}

impl BPETokenizer {
    /// Builds the byte-level BPE tokenizer by training on the provided corpus.
    /// It iteratively merges the most frequent adjacent pair for maximum compression.
    pub fn build<S: AsRef<[u8]>>(items: impl IntoIterator<Item = S>, min_freq: usize, max_token: u16) -> Self {
        println!("this can take a while...");
        let mut start = Instant::now();
        // Initialize vocabulary with all 256 possible bytes.
        let mut vocab: HashMap<Box<[u8]>, u16> = HashMap::with_hasher(HASHER);
        let mut inv_vocab_vec: Vec<Box<[u8]>> = Vec::with_capacity(65536);
        for i in 0u16..=255 {
            let token = Box::new([i as u8]) as Box<[u8]>;
            vocab.insert(token.clone(), i);
            inv_vocab_vec.push(token);
        }
        let mut next_token_id: u16 = 256; // next available token id

        // Convert training items into corpus sequences (Vec<Vec<u16>>)
        let mut samples: Vec<Vec<u16>> = items
            .into_iter()
            .map(|item| item.as_ref().iter().map(|&b| b as u16).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        println!("building vocab");
        // shuffle to improve multi thread distribution

        let mut merges: HashMap<(u16, u16), u16> = HashMap::new();
        let mut merge_len = HashMap::<(u16, u16), usize>::new();

        loop {
            start = Instant::now();
            if next_token_id % 500 == 0 {
                println!("sorting sample");
                // make retain a little faster by move short item to end of vec
                samples.par_sort_unstable_by(|left, right| right.len().cmp(&left.len()).then_with(|| right.cmp(left)));
                println!("truncating sample len < 2");
                samples.retain(|seq| seq.len() >= 2);
                println!("{next_token_id} tokens proceeds {} sample remain", samples.len());
            }

            let pair_freq = run(
                &samples,
                HashMap::<(u16, u16), usize>::new,
                |pair_freq, seq| {
                    if seq.len() < 2 {
                        return;
                    }
                    for window in seq.windows(2) {
                        let pair = (window[0], window[1]);
                        *pair_freq.entry(pair).or_default() += 1;
                    }
                },
                |mut left, right| {
                    for (k, v) in right {
                        *left.entry(k).or_default() += v;
                    }
                    left
                },
            );

            if pair_freq.is_empty() {
                break; // no more pairs to merge
            }

            // Find the most frequent pair.
            let (&best_pair, &freq) = pair_freq
                .par_iter()
                .max_by_key(|&(pair, &count)| (count, merge_len.get(pair).copied().unwrap_or_default()))
                .unwrap();
            // In this implementation we always merge the best pair,
            // but you could decide to skip rare pairs if desired.
            if freq < min_freq {
                break;
            }

            // Create new token by merging the best pair.
            let (left, right) = best_pair;
            let new_token = inv_vocab_vec[left as usize]
                .iter()
                .chain(&inv_vocab_vec[right as usize])
                .copied()
                .collect::<Vec<_>>();
            let new_token_box = new_token.into_boxed_slice();

            merge_len.insert(best_pair, new_token_box.len());
            // Add new token to vocabulary and inverse vocabulary.
            vocab.insert(new_token_box.clone(), next_token_id);
            inv_vocab_vec.push(new_token_box);

            // Record merge rule: mapping from best_pair to the new token id.
            merges.insert(best_pair, next_token_id);

            // Update corpus sequences: merge all occurrences of best_pair.
            samples.par_iter_mut().for_each(|seq| {
                let mut i = 0;
                let mut new_seq = Vec::with_capacity(seq.len().next_power_of_two());
                while i < seq.len() {
                    if i + 1 < seq.len() && (seq[i], seq[i + 1]) == best_pair {
                        new_seq.push(next_token_id);
                        i += 2; // skip the next token as well
                    } else {
                        new_seq.push(seq[i]);
                        i += 1;
                    }
                }
                *seq = new_seq;
            });

            if next_token_id == max_token {
                break; // reached token limit
            }
            next_token_id += 1;
        }

        let inv_vocab = inv_vocab_vec.into_boxed_slice();
        println!("vocab size: {}", inv_vocab.len());
        BPETokenizer {
            merges,
            vocab,
            inv_vocab,
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // merged
        writer.write_all(self.merges.len().to_le_bytes().as_ref())?;
        for ((a, b), c) in &self.merges {
            writer.write_all(a.to_le_bytes().as_ref())?;
            writer.write_all(b.to_le_bytes().as_ref())?;
            writer.write_all(c.to_le_bytes().as_ref())?;
        }

        // vocab
        writer.write_all(self.vocab.len().to_le_bytes().as_ref())?;
        for (bytes, v) in &self.vocab {
            writer.write_all(bytes.len().to_le_bytes().as_ref())?;
            writer.write_all(bytes)?;
            writer.write_all(v.to_le_bytes().as_ref())?;
        }
        // inv_vocab
        writer.write_all(self.inv_vocab.len().to_le_bytes().as_ref())?;
        for bytes in &self.inv_vocab {
            writer.write_all(bytes.len().to_le_bytes().as_ref())?;
            writer.write_all(bytes)?;
        }
        writer.flush()?;
        writer.into_inner()?.sync_all()?;
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut merges = HashMap::new();
        let mut vocab = HashMap::new();
        let mut inv_vocab = Vec::new();
        let mut reader = BufReader::new(file);
        let mut buf2 = [0; 2];
        let mut buf8 = [0; 8];
        reader.read_exact(&mut buf8)?;
        let merge_size = usize::from_le_bytes(buf8);
        for _ in 0..merge_size {
            reader.read_exact(&mut buf2)?;
            let a = u16::from_le_bytes(buf2);
            reader.read_exact(&mut buf2)?;
            let b = u16::from_le_bytes(buf2);
            reader.read_exact(&mut buf2)?;
            let c = u16::from_le_bytes(buf2);
            merges.insert((a, b), c);
        }

        reader.read_exact(&mut buf8)?;
        let vocab_size = usize::from_le_bytes(buf8);
        let mut bytes_buf = Vec::with_capacity(4096);
        for _ in 0..vocab_size {
            reader.read_exact(&mut buf8)?;
            let len = usize::from_le_bytes(buf8);
            bytes_buf.resize(len, 0);
            reader.read_exact(&mut bytes_buf)?;
            let k = bytes_buf.clone().into_boxed_slice();
            reader.read_exact(&mut buf2)?;
            let v = u16::from_le_bytes(buf2);
            vocab.insert(k, v);
        }

        reader.read_exact(&mut buf8)?;
        let inv_vocab_size = usize::from_le_bytes(buf8);
        for _ in 0..inv_vocab_size {
            reader.read_exact(&mut buf8)?;
            let len = usize::from_le_bytes(buf8);
            bytes_buf.resize(len, 0);
            reader.read_exact(&mut bytes_buf)?;
            inv_vocab.push(bytes_buf.clone().into_boxed_slice());
        }

        Ok(Self {
            merges,
            vocab,
            inv_vocab: inv_vocab.into_boxed_slice(),
        })
    }

    /// Encodes the input byte slice into a sequence of token IDs using the learned merge rules.
    pub fn encode(&self, input: &[u8]) -> Vec<u16> {
        // Start with base tokens (each byte)
        let mut tokens: Vec<u16> = input.iter().map(|&b| b as u16).collect();

        // Iteratively merge tokens based on learned rules.
        loop {
            let mut best_candidate_index: Option<usize> = None;
            let mut best_rank: u16 = u16::MAX;

            // Evaluate all adjacent pairs to find mergeable ones.
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&merge_token) = self.merges.get(&pair) {
                    // Lower token id indicates an earlier merge (higher priority).
                    if merge_token < best_rank {
                        best_rank = merge_token;
                        best_candidate_index = Some(i);
                    }
                }
            }

            if let Some(i) = best_candidate_index {
                // Merge the best candidate pair.
                let pair = (tokens[i], tokens[i + 1]);
                let new_token = self.merges.get(&pair).unwrap();
                tokens[i] = *new_token;
                tokens.remove(i + 1);
            } else {
                break; // no more mergeable pairs.
            }
        }

        tokens
    }

    pub fn decode(&self, token_ids: &[u16]) -> Vec<u8> {
        token_ids
            .iter()
            .filter_map(|&id| self.inv_vocab.get(id as usize))
            .flatten()
            .copied()
            .collect()
    }

    pub fn memory_usage(&self) -> usize {
        let vocab_use = self.vocab.keys().map(|k| k.len() + 8).sum::<usize>()
            + (self.vocab.capacity()
                * size_of::<(Box<[u8]>, usize /* u16 get padded to alignment of key anyway */)>());
        let inv_vocab_use =
            self.inv_vocab.iter().map(|x| x.len()).sum::<usize>() + (self.inv_vocab.len() * size_of::<Box<[u8]>>());
        let merges_use = self.merges.len() * size_of::<((u16, u16), u16, u16)>()
            + self.merges.capacity() * size_of::<((u16, u16), u16, u16)>();
        vocab_use + inv_vocab_use + merges_use
    }
}
