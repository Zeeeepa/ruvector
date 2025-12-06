# @ruvector/studio

Custom Supabase Studio components and pages for RuVector - the AI-native vector database.

## Features

This package contains custom pages and components for managing RuVector's AI-native features:

- **Vector Indexes** - HNSW and IVFFlat index management with performance monitoring
- **Attention Mechanisms** - 39 attention functions for in-database transformer computations
- **Graph Neural Networks** - GCN, GraphSAGE, GAT models with SQL integration
- **Hyperbolic Embeddings** - Poincaré ball and Lorentz hyperboloid embeddings
- **Self-Learning** - ReasoningBank adaptive learning with trajectory tracking
- **Agent Routing** - Intelligent query routing to specialized agents

## Structure

```
npm/studio/
├── components/
│   └── interfaces/
│       └── RuVector/
│           └── RuVectorHome.tsx    # Main dashboard component
├── pages/
│   └── project/
│       └── [ref]/
│           ├── index.tsx           # Project dashboard override
│           ├── vectors/            # Vector index management
│           ├── attention/          # Attention mechanisms
│           ├── gnn/               # Graph neural networks
│           ├── hyperbolic/        # Hyperbolic embeddings
│           ├── learning/          # Self-learning system
│           └── routing/           # Agent routing
└── package.json
```

## Usage

These files are designed to be copied into a Supabase Studio build. See the Docker configuration for automated integration.

### Docker Build

```bash
# Build custom studio image
docker build -f docker/Dockerfile.studio -t ruvector-studio:custom .

# Run studio
docker run -p 3001:3000 \
  -e STUDIO_PG_META_URL=http://host.docker.internal:8080 \
  ruvector-studio:custom
```

## Development

To modify the studio pages:

1. Edit files in `npm/studio/`
2. Rebuild the Docker image with `--no-cache`
3. Deploy the updated container

## License

Apache-2.0
