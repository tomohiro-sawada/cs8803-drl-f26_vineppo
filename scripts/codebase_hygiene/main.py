"""
Run checks on the codebase conventions.

These scripts are kept separate from `tests` (even though they are *like* tests of codebase conventions) so to not violate Ousterhout's test isomorphism principle.


AGENT INSTRUCTIONS:

- Implement test to make sure no Python file has forbidden character (especially dash `-`) in their file name. Should be trivial to do if you use use Rust-based clients such as `fd`.

"""
