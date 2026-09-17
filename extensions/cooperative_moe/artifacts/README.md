# Validated native artifact

`prepare_profile.py` and the adapter pin `cooperative_moe.so` to

`9a9c44f0e423e3cfe595f195925e520af8e1b5bc56814fcaefccda87a4e983ae`.

A clean nvcc rebuild in the recipe image is **not** bit-identical: GNU build-id
and CUDA `-lineinfo` filename metadata change across runs. This fork therefore
builds the binary once, gates it, and commits that pinned artifact; the operator
path installs it rather than compiling it.

The binary ships checked in to this directory (Git ignores `*.so` at the repo
root; this path is the exception); verify it with `sha256sum -c SHA256SUMS`. Do
not repin a different hash without repeating the 54-case GPU gate
(see `docs/cooperative-moe-build-repin.md`).
