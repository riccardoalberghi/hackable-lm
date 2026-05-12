.PHONY: setup
setup:
	@echo "Checking for Rust toolchain..."
	@which rustc > /dev/null 2>&1 && (echo "  ✓ Rust already installed") || ( \
		echo "  Installing Rust via rustup..." && \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet && \
		. "$$HOME/.cargo/env" && \
		echo "  ✓ Rust installed" )
	uv sync --locked
	@echo "✓ Setup complete"

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