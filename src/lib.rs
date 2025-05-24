#![feature(specialization)]
#![feature(iter_intersperse)]
#![feature(ptr_internals)]
#![feature(vec_into_raw_parts)]
#![feature(likely_unlikely)]
#![allow(incomplete_features)]
pub mod bpe;

use rand::{
    prelude::{IndexedRandom, StdRng},
    SeedableRng,
};
use rayon::iter::*;
use std::path::Path;

pub fn read_all() -> (Vec<Box<str>>, Vec<Box<str>>) {
    // download name from here https://pypi.org/project/names-dataset/#full-dataset
    // and look for google drive link
    let data = "/mnt/hdd/dataset/name_dataset/data/";
    let full = std::fs::read_dir(data)
        .unwrap()
        .collect::<Result<Vec<_>, std::io::Error>>()
        .unwrap()
        .into_par_iter()
        .flat_map(|x| read(&x.path()))
        .collect::<Vec<_>>();
    let sample = full
        .choose_multiple(&mut StdRng::seed_from_u64(123456789), 100)
        .cloned()
        .collect();
    (full, sample)
}

fn read(path: &Path) -> Vec<Box<str>> {
    let mut out = Vec::new();
    let content = std::fs::read_to_string(path).unwrap();
    for line in content.lines().take(100000) {
        let Some((take, v)) = line.split_once(",") else {
            continue;
        };
        if take.is_empty() {
            continue;
        }
        out.push(take.trim_ascii().to_string().into_boxed_str());
    }
    out
}
