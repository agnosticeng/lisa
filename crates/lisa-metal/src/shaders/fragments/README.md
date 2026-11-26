# Shader fragments

Runs of source shared verbatim by several kernel headers. A header names a
fragment with a whole line `//@include fragments/<name>.inc`; `build.rs`
replaces that line with the fragment's bytes when it packs `kernels.bin`, so
every kernel compiles exactly the text it would have without the indirection.
Fragments are never packed on their own and never include other fragments.

Each fragment is named after the upstream header it comes from, or `shared`
when the run carries no `#line` origin. Fragments carry the licence and
attribution of the headers that include them; see the repository NOTICE.

Provenance pins record hashes of the expanded sources and are checked by
`shader_provenance_tests.rs`.
