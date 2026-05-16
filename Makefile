EXT_RUSTFLAGS :=
ifeq ($(shell uname),Darwin)
EXT_RUSTFLAGS := RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"
endif
RUSTUP_INIT_URL := https://sh.rustup.rs
RUST_ENV := PATH="$(HOME)/.cargo/bin:$$PATH"

.PHONY: setup install-rust
setup: install-rust
	$(RUST_ENV) cargo build --release --manifest-path hackablebpe/Cargo.toml --bin hackablebpe_train
	$(RUST_ENV) $(EXT_RUSTFLAGS) cargo build --release --manifest-path hackablebpe/Cargo.toml --lib --features extension-module
	uv sync --locked
	@echo "✓ Setup complete"

install-rust:
	@if ! command -v cargo >/dev/null 2>&1 && ! "$(HOME)/.cargo/bin/cargo" --version >/dev/null 2>&1; then \
		echo "Installing Rust toolchain with rustup"; \
		curl --proto '=https' --tlsv1.2 -sSf "$(RUSTUP_INIT_URL)" | sh -s -- -y --profile minimal; \
	fi

.PHONY: sync
sync:
	uv sync --locked

.PHONY: test
test:
	uv run pytest

.PHONY: lint
lint:
	uv run ruff check

.PHONY: clean
clean:
	rm -rf .venv
	uv cache clean
