#![feature(specialization)]
#![feature(iter_intersperse)]
#![feature(ptr_internals)]
#![feature(vec_into_raw_parts)]
#![feature(likely_unlikely)]
#![allow(incomplete_features)]
mod bpe;

use crate::bpe::BPESet;
use ahash::HashSet;
use name_lookup::read_all;

fn main() {
    let (contents, sample) = read_all();
    let hashset = contents.iter().collect::<HashSet<_>>();

    println!("names: {} records", contents.len());
    let base = {
        let base = hashset.iter().map(|x| x.len()).sum::<usize>() + (hashset.capacity() * size_of::<Box<str>>());
        println!("hashset memory usage: {base} bytes");
        drop(hashset);
        base
    };
    {
        //let mut set = BPESet::new(contents.iter().map(|x| x.as_bytes()), 10, 4095);
        //set.save_vocab("vocab.bpe".as_ref()).unwrap();
        let mut set = BPESet::with_vocab("vocab.bpe".as_ref()).unwrap();
        contents.iter().for_each(|x| set.insert(x.as_bytes()));
        let mem_use = set.memory_usage();
        println!("bpe_set hashset memory usage: {mem_use} bytes");

        println!("bpe reduced {}", base as f64 / mem_use as f64)
    }
}
