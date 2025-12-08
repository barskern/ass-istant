use std::env;
use std::sync::LazyLock;
use std::time::Duration;

use directories::ProjectDirs;
use rand::Rng;
use rand_distr::Exp;

pub mod waitable_lock;

pub static PROJECT_DIRS: LazyLock<ProjectDirs> =
    LazyLock::new(|| ProjectDirs::from("", "", env!("CARGO_PKG_NAME")).unwrap());

pub fn human_message_duration(chars: usize, chars_per_sec: f32) -> Duration {
    let expected_message_time = chars as f32 / chars_per_sec;
    //let distr = Normal::new(expected_message_time, 5.0).expect("std is finite");
    let distr = Exp::new(1.0 / expected_message_time).unwrap_or(Exp::new(1.0 / 3.0).unwrap());
    let mut rng = rand::rng();
    Duration::from_secs_f32(rng.sample(distr).max(5.0))
}
