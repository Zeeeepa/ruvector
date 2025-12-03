//! SONA Training System
//!
//! Templated training pipelines for specialized model adaptation.
//!
//! ## Overview
//!
//! The training module provides:
//! - **Training Templates**: Pre-configured training setups for common use cases
//! - **Agent Factory**: Create and manage multiple specialized agents
//! - **Training Pipelines**: Structured workflows for different verticals
//! - **Metrics & Results**: Comprehensive training analytics
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ruvector_sona::training::{TrainingTemplate, AgentFactory, TrainingPipeline};
//!
//! // Use a preset template
//! let template = TrainingTemplate::code_agent();
//! let pipeline = TrainingPipeline::from_template(template);
//!
//! // Train on examples
//! for example in examples {
//!     pipeline.add_example(example);
//! }
//! let results = pipeline.train()?;
//! ```

mod templates;
mod factory;
mod pipeline;
mod metrics;

pub use templates::{
    TrainingTemplate, TemplatePreset, VerticalConfig,
    AgentType, TaskDomain, TrainingMethod, DataSizeHint,
};
pub use factory::{
    AgentFactory, ManagedAgent, AgentHandle, AgentStats,
    TrainingExample as FactoryTrainingExample, SimpleExample, SharedAgentFactory,
};
pub use pipeline::{
    TrainingPipeline, PipelineStage, TrainingExample,
    BatchConfig, TrainingCallback,
};
pub use metrics::{
    TrainingMetrics, TrainingResult, EpochStats,
    QualityMetrics, PerformanceMetrics,
};
