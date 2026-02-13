use std::time::Duration;

use rand::seq::IteratorRandom;
use rand_distr::{Distribution, Exp};

pub mod waitable_lock;

pub fn human_message_duration(chars: usize, chars_per_sec: f32) -> Duration {
    let expected_message_time = chars as f32 / chars_per_sec;
    //let distr = Normal::new(expected_message_time, 5.0).expect("std is finite");
    let distr = Exp::new(1.0 / expected_message_time).unwrap_or(Exp::new(1.0 / 3.0).unwrap());
    let mut rng = rand::rng();
    Duration::from_secs_f32(distr.sample(&mut rng).max(5.0))
}

pub fn random_rejection_message() -> &'static str {
    const REJECTIONS: &str = include_str!("utils/rejections.txt");
    REJECTIONS
        .lines()
        .choose(&mut rand::rng())
        .unwrap_or("Dead inside..")
}
