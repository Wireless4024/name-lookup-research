#![feature(test)]
extern crate test;

use ahash::HashSet;
use name_lookup::read_all;
use test::Bencher;

#[bench]
fn bench_hashset(b: &mut Bencher) {
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", (num_cpus::get() / 2).to_string());
    }
    let (contents, sample) = read_all();
    let set = contents.into_iter().collect::<HashSet<_>>();

    b.iter(|| {
        for s in &sample {
            let expect = set.contains(s);
            let expect = std::hint::black_box(expect);
            assert!(expect);
        }
    });
}
