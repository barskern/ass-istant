alias c := check
alias r := run
alias b := build
alias t := test

default:
    just --list

build:
    cargo build --workspace --all-features --all-targets

clean:
    cargo clean --verbose

clippy:
   cargo clippy --workspace --all-features --all-targets -- -D warnings

check-fmt:
    cargo +nightly fmt --all -- --check

fmt:
    cargo +nightly fmt --all

test:
    cargo test --workspace --all-features --all-targets

check:
    @just check-fmt
    @just clippy
    @just test

all:
    @just clean
    @just check
    @just build

run:
    cargo run --bin ass-istant
