#![feature(test)]
extern crate test;

use name_lookup::{bpe::BPESet, read_all};
use test::Bencher;

#[bench]
fn bench_bpe_set(b: &mut Bencher) {
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", (num_cpus::get() / 2).to_string());
    }
    let (contents, sample) = read_all();
    let mut set = BPESet::with_vocab("vocab.bpe".as_ref()).unwrap();
    for x in contents {
        set.insert(x.as_bytes());
    }

    b.iter(|| {
        for s in &sample {
            let expect = set.contains(s.as_bytes());
            let expect = std::hint::black_box(expect);
            assert!(expect);
        }
    });
}
