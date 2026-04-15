# Base agent bash environment.
# Run-specific agent.bashrc files (in each run directory) source this
# and add run-specific env vars. This file defines shared tool aliases.

# Make bin/ tools available
if [[ -n "${SCAFFOLD_ROOT:-}" ]]; then
    export PATH="${SCAFFOLD_ROOT}/bin:${PATH}"
fi

# Alias `test` so the strategy agent can call it naturally
test_examples() {
    run-tests "$@"
}
alias test='test_examples'
