# Multi-stage Dockerfile for ruvector-postgres extension
# Builds the extension and creates a PostgreSQL image with it installed

# Build stage
FROM rust:1.83-slim-bookworm AS builder

# Install build dependencies including PostgreSQL 17 from PGDG
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    clang \
    libclang-dev \
    flex \
    bison \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Add PostgreSQL official apt repository
RUN curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Install PostgreSQL 17 development packages
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-17 \
    postgresql-17 \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-pgrx
RUN cargo install cargo-pgrx --version 0.12.9 --locked

# Set up workspace
WORKDIR /build

# Copy only Cargo files first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ruvector-postgres/Cargo.toml ./crates/ruvector-postgres/

# Copy source code
COPY crates/ruvector-postgres ./crates/ruvector-postgres/

# Initialize pgrx with system PostgreSQL
RUN cd crates/ruvector-postgres && \
    cargo pgrx init --pg17=/usr/lib/postgresql/17/bin/pg_config

# Build the extension with all features including embeddings
RUN cd crates/ruvector-postgres && \
    cargo pgrx package --features "pg17 index-all quant-all embeddings"

# Runtime stage
FROM postgres:17-bookworm

# Labels
LABEL maintainer="ruvector team"
LABEL description="PostgreSQL with ruvector extension - high-performance vector similarity search with local embeddings"
LABEL version="0.2.4"

# Copy the built extension from builder
COPY --from=builder /build/target/release/ruvector-postgres-pg17/usr/share/postgresql/17/extension/* \
    /usr/share/postgresql/17/extension/
COPY --from=builder /build/target/release/ruvector-postgres-pg17/usr/lib/postgresql/17/lib/* \
    /usr/lib/postgresql/17/lib/

# Copy SQL files and control file
COPY --from=builder /build/crates/ruvector-postgres/ruvector.control \
    /usr/share/postgresql/17/extension/
COPY --from=builder /build/crates/ruvector-postgres/sql/*.sql \
    /usr/share/postgresql/17/extension/

# Add initialization script to create extension
RUN mkdir -p /docker-entrypoint-initdb.d
RUN echo "CREATE EXTENSION IF NOT EXISTS ruvector;" > /docker-entrypoint-initdb.d/01-ruvector.sql

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres || exit 1

# Expose PostgreSQL port
EXPOSE 5432

# Use the default PostgreSQL entrypoint
CMD ["postgres"]
