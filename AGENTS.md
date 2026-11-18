I value robustness, minimizing total lines of code, avoiding bloat, avoiding duplication.
I hate fallbacks, useless if-else statements (things like "elif code is broken then repair it" should not exist, it should fail loudly), and I love asserts, which allow to catch early bugs. Fallbacks hide bugs and bloat the code.

Instructions for coding agents are marked with `AGENT INSTRUCTIONS` in the doc strings.

# Conventions 

- When you add new entries to `prose` or update a log, include the timestamp (generated using `bash.timestamp` function) and the git commit hash (via `git rev-parse --short HEAD`) in your response. Crucial for traceability.
