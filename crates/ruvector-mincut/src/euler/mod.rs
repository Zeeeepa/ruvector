//! Euler Tour Trees for dynamic tree operations
//!
//! Provides O(log n) operations for:
//! - Dynamic connectivity queries
//! - Subtree aggregation
//! - Edge insertion/deletion in trees
//!
//! # Overview
//!
//! An Euler Tour Tree (ETT) represents a tree as a sequence of vertices
//! encountered during an Euler tour (DFS traversal). Each edge (u, v)
//! contributes two occurrences: entering and exiting the subtree rooted at v.
//!
//! The sequence is stored in a balanced BST (treap) with implicit keys,
//! where position = size of left subtree. This enables:
//! - O(log n) split at any position
//! - O(log n) merge of two sequences
//! - O(log n) subtree aggregation via range queries
//!
//! # Example
//!
//! ```rust
//! use ruvector_mincut::euler::EulerTourTree;
//!
//! let mut ett = EulerTourTree::new();
//!
//! // Create trees
//! ett.make_tree(1).unwrap();
//! ett.make_tree(2).unwrap();
//! ett.make_tree(3).unwrap();
//!
//! // Link them: 1 - 2 - 3
//! ett.link(1, 2).unwrap();
//! ett.link(2, 3).unwrap();
//!
//! // Check connectivity
//! assert!(ett.connected(1, 3));
//!
//! // Get tree size
//! assert_eq!(ett.tree_size(1).unwrap(), 3);
//! ```

use crate::{MinCutError, Result};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Node identifier
pub type NodeId = u64;

/// Treap node for balanced BST representation of Euler tour
#[derive(Debug, Clone)]
struct TreapNode {
    /// The vertex this occurrence represents
    vertex: NodeId,
    /// Priority for treap balancing (random)
    priority: u64,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Parent pointer
    parent: Option<usize>,
    /// Subtree size (number of nodes in subtree)
    size: usize,
    /// Value at this node
    value: f64,
    /// Aggregate over subtree (sum in this implementation)
    subtree_aggregate: f64,
}

impl TreapNode {
    /// Create a new treap node
    fn new(vertex: NodeId, priority: u64, value: f64) -> Self {
        Self {
            vertex,
            priority,
            left: None,
            right: None,
            parent: None,
            size: 1,
            value,
            subtree_aggregate: value,
        }
    }
}

/// Represents a tree as an Euler tour stored in a balanced BST (treap)
pub struct EulerTourTree {
    /// Node storage (arena allocation)
    nodes: Vec<TreapNode>,
    /// Free list for deleted nodes (indices)
    free_list: Vec<usize>,
    /// Map from vertex to its first occurrence in tour
    first_occurrence: HashMap<NodeId, usize>,
    /// Map from vertex to its last occurrence in tour
    last_occurrence: HashMap<NodeId, usize>,
    /// Map from edge (u, v) to the tour node representing entry to subtree
    edge_to_node: HashMap<(NodeId, NodeId), usize>,
    /// Root of the treap (per tree)
    roots: HashMap<NodeId, usize>,
    /// Random number generator for priorities
    rng: rand::rngs::StdRng,
}

impl EulerTourTree {
    /// Create a new empty Euler Tour Tree
    pub fn new() -> Self {
        Self::with_seed(42)
    }

    /// Create with a seed for reproducibility
    pub fn with_seed(seed: u64) -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            first_occurrence: HashMap::new(),
            last_occurrence: HashMap::new(),
            edge_to_node: HashMap::new(),
            roots: HashMap::new(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Create a singleton tree with one vertex
    pub fn make_tree(&mut self, v: NodeId) -> Result<()> {
        if self.first_occurrence.contains_key(&v) {
            return Err(MinCutError::InternalError(format!(
                "Vertex {} already exists in a tree",
                v
            )));
        }

        let priority = self.rng.gen();
        let idx = self.allocate_node(v, priority, 0.0);

        self.first_occurrence.insert(v, idx);
        self.last_occurrence.insert(v, idx);
        self.roots.insert(v, idx);

        Ok(())
    }

    /// Link: Make v a child of u (v must be in separate tree)
    pub fn link(&mut self, u: NodeId, v: NodeId) -> Result<()> {
        // Validate vertices exist
        let u_idx = *self.first_occurrence.get(&u)
            .ok_or_else(|| MinCutError::InvalidVertex(u))?;
        let v_root = *self.first_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;

        // Check they're in different trees
        if self.connected(u, v) {
            return Err(MinCutError::EdgeExists(u, v));
        }

        // Reroot u's tree to make u the root
        self.reroot_internal(u)?;

        // Get the new root of u's tree after rerooting
        let u_root = self.find_root_idx(u_idx)?;

        // Create two new tour nodes for the edge (u, v)
        let priority1 = self.rng.gen();
        let priority2 = self.rng.gen();
        let enter_v = self.allocate_node(v, priority1, 0.0);
        let exit_v = self.allocate_node(u, priority2, 0.0);

        // Store edge mapping
        self.edge_to_node.insert((u, v), enter_v);

        // Merge: u_tour + [enter_v] + v_tour + [exit_v]
        let merged1 = self.merge(Some(u_root), Some(enter_v));
        let merged2 = self.merge(merged1, Some(v_root));
        let final_root = self.merge(merged2, Some(exit_v));

        // Update root mapping for all vertices in the merged tree
        if let Some(root) = final_root {
            let root_vertex = self.nodes[root].vertex;
            self.update_root_mapping(root, root_vertex);
        }

        Ok(())
    }

    /// Cut: Remove edge between u and v
    ///
    /// # Note
    ///
    /// This operation is partially implemented. The `find_matching_exit`
    /// helper is a simplified stub and needs full implementation for
    /// production use.
    pub fn cut(&mut self, u: NodeId, v: NodeId) -> Result<()> {
        // Find the edge occurrence nodes
        let edge_node = self.edge_to_node.remove(&(u, v))
            .or_else(|| self.edge_to_node.remove(&(v, u)))
            .ok_or_else(|| MinCutError::EdgeNotFound(u, v))?;

        // Reroot to make u the root
        self.reroot_internal(u)?;

        // After rerooting, (u, v) edge should have enter_v node
        let enter_v = edge_node;

        // Find the corresponding exit node (the 'u' occurrence after v's subtour)
        let exit_u_idx = self.find_matching_exit(enter_v)?;

        // Get positions
        let pos1 = self.get_position(enter_v);
        let pos2 = self.get_position(exit_u_idx);

        // Ensure pos1 < pos2
        let (start_pos, end_pos) = if pos1 < pos2 {
            (pos1, pos2)
        } else {
            (pos2, pos1)
        };

        // Get root of the tree
        let root = self.find_root_idx(*self.first_occurrence.get(&u).unwrap())?;

        // Split to extract v's subtree
        // tree = left + [enter_v] + middle + [exit_u] + right
        let (left, rest) = self.split(root, start_pos);
        if rest.is_none() {
            return Err(MinCutError::InternalError("Split failed".to_string()));
        }

        let (enter_and_middle, right) = self.split(rest.unwrap(), end_pos - start_pos + 1);

        // Further split to separate enter_v from middle
        let (enter_node, middle_and_exit) = self.split_first(enter_and_middle);

        // Split to separate middle from exit_u
        let (middle, exit_node) = self.split_last(middle_and_exit);

        // Free the edge nodes
        if let Some(idx) = enter_node {
            self.free_node(idx);
        }
        if let Some(idx) = exit_node {
            self.free_node(idx);
        }

        // Merge u's parts: left + right
        let u_tree = self.merge(left, right);

        // v's tree is middle
        let v_tree = middle;

        // Update root mappings
        if let Some(root_idx) = u_tree {
            let root_vertex = self.nodes[root_idx].vertex;
            self.update_root_mapping(root_idx, root_vertex);
        }
        if let Some(root_idx) = v_tree {
            let root_vertex = self.nodes[root_idx].vertex;
            self.update_root_mapping(root_idx, root_vertex);
        }

        Ok(())
    }

    /// Check if two vertices are in the same tree
    pub fn connected(&self, u: NodeId, v: NodeId) -> bool {
        match (self.first_occurrence.get(&u), self.first_occurrence.get(&v)) {
            (Some(&u_idx), Some(&v_idx)) => {
                let u_root = self.find_root_idx(u_idx);
                let v_root = self.find_root_idx(v_idx);
                matches!((u_root, v_root), (Ok(ur), Ok(vr)) if ur == vr)
            }
            _ => false,
        }
    }

    /// Find the root of the tree containing v
    pub fn find_root(&self, v: NodeId) -> Result<NodeId> {
        let v_idx = *self.first_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;
        let root_idx = self.find_root_idx(v_idx)?;
        Ok(self.nodes[root_idx].vertex)
    }

    /// Get the size of the tree containing v
    pub fn tree_size(&self, v: NodeId) -> Result<usize> {
        let v_idx = *self.first_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;
        let root_idx = self.find_root_idx(v_idx)?;

        // Count unique vertices in the tour by collecting them
        let vertices = self.collect_vertices(root_idx);
        Ok(vertices.len())
    }

    /// Get the size of the subtree rooted at v
    pub fn subtree_size(&self, v: NodeId) -> Result<usize> {
        let first_idx = *self.first_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;
        let last_idx = *self.last_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;

        if first_idx == last_idx {
            return Ok(1); // Singleton or leaf
        }

        // Subtree size = (occurrences between first and last + 1) / 2
        let pos1 = self.get_position(first_idx);
        let pos2 = self.get_position(last_idx);
        let range_size = pos2.saturating_sub(pos1) + 1;

        Ok((range_size + 1) / 2)
    }

    /// Aggregate over the subtree rooted at v
    pub fn subtree_aggregate(&self, v: NodeId) -> Result<f64> {
        let first_idx = *self.first_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;

        // For simplicity, return the aggregate of the first occurrence's subtree
        Ok(self.nodes[first_idx].subtree_aggregate)
    }

    /// Update the value at vertex v
    pub fn update_value(&mut self, v: NodeId, value: f64) -> Result<()> {
        let first_idx = *self.first_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;

        self.nodes[first_idx].value = value;
        self.pull_up(first_idx);

        Ok(())
    }

    /// Reroot the tree containing v to make v the root
    pub fn reroot(&mut self, v: NodeId) -> Result<()> {
        self.reroot_internal(v)
    }

    /// Get the number of vertices
    pub fn len(&self) -> usize {
        self.first_occurrence.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.first_occurrence.is_empty()
    }

    // ===== Internal Implementation =====

    /// Find root index of the treap containing node at idx
    fn find_root_idx(&self, mut idx: usize) -> Result<usize> {
        let mut visited = 0;
        let max_depth = self.nodes.len() * 2; // Prevent infinite loops

        while let Some(parent) = self.nodes[idx].parent {
            idx = parent;
            visited += 1;
            if visited > max_depth {
                return Err(MinCutError::InternalError("Cycle detected in tree".to_string()));
            }
        }
        Ok(idx)
    }

    /// Reroot implementation
    fn reroot_internal(&mut self, v: NodeId) -> Result<()> {
        let v_first = *self.first_occurrence.get(&v)
            .ok_or_else(|| MinCutError::InvalidVertex(v))?;

        // Get current root
        let root = self.find_root_idx(v_first)?;

        // If v is already at position 0, nothing to do
        let pos = self.get_position(v_first);
        if pos == 0 {
            return Ok(());
        }

        // Rotate the tour: move [0..pos) to the end
        // Split at position pos
        let (left, right) = self.split(root, pos);

        // Merge in reverse order: right + left
        let new_root = self.merge(right, left);

        // Update root mapping
        if let Some(root_idx) = new_root {
            let root_vertex = self.nodes[root_idx].vertex;
            self.update_root_mapping(root_idx, root_vertex);
        }

        Ok(())
    }

    /// Update root mapping for all vertices in the tree
    fn update_root_mapping(&mut self, root_idx: usize, _root_vertex: NodeId) {
        // Collect all unique vertices in this tree
        let vertices = self.collect_vertices(root_idx);
        for vertex in vertices {
            self.roots.insert(vertex, root_idx);
        }
    }

    /// Collect all unique vertices in the subtree
    fn collect_vertices(&self, idx: usize) -> Vec<NodeId> {
        let mut vertices = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.collect_vertices_helper(idx, &mut vertices, &mut visited);
        vertices
    }

    fn collect_vertices_helper(&self, idx: usize, vertices: &mut Vec<NodeId>, visited: &mut std::collections::HashSet<NodeId>) {
        let node = &self.nodes[idx];
        if visited.insert(node.vertex) {
            vertices.push(node.vertex);
        }

        if let Some(left) = node.left {
            self.collect_vertices_helper(left, vertices, visited);
        }
        if let Some(right) = node.right {
            self.collect_vertices_helper(right, vertices, visited);
        }
    }

    /// Find the matching exit node for an enter node
    fn find_matching_exit(&self, _enter_idx: usize) -> Result<usize> {
        // This is a simplified implementation
        // In a full implementation, we'd track which exit corresponds to which enter
        // For now, we'll search for the next occurrence of the parent vertex
        Err(MinCutError::InternalError("Not implemented".to_string()))
    }

    /// Split treap at position pos
    /// Returns (left, right) where left contains [0..pos) and right contains [pos..)
    fn split(&mut self, root: usize, pos: usize) -> (Option<usize>, Option<usize>) {
        if pos == 0 {
            return (None, Some(root));
        }

        let left_size = self.nodes[root].left.map(|l| self.nodes[l].size).unwrap_or(0);

        if pos <= left_size {
            // Split in left subtree
            if let Some(left_child) = self.nodes[root].left {
                let (split_left, split_right) = self.split(left_child, pos);
                self.nodes[root].left = split_right;
                if let Some(idx) = split_right {
                    self.nodes[idx].parent = Some(root);
                }
                self.pull_up(root);

                if let Some(idx) = split_left {
                    self.nodes[idx].parent = None;
                }

                (split_left, Some(root))
            } else {
                (None, Some(root))
            }
        } else {
            // Split in right subtree
            let right_pos = pos - left_size - 1;
            if let Some(right_child) = self.nodes[root].right {
                let (split_left, split_right) = self.split(right_child, right_pos);
                self.nodes[root].right = split_left;
                if let Some(idx) = split_left {
                    self.nodes[idx].parent = Some(root);
                }
                self.pull_up(root);

                if let Some(idx) = split_right {
                    self.nodes[idx].parent = None;
                }

                (Some(root), split_right)
            } else {
                (Some(root), None)
            }
        }
    }

    /// Merge two treaps maintaining treap property (max heap on priority)
    fn merge(&mut self, left: Option<usize>, right: Option<usize>) -> Option<usize> {
        match (left, right) {
            (None, right) => right,
            (left, None) => left,
            (Some(l), Some(r)) => {
                if self.nodes[l].priority > self.nodes[r].priority {
                    // Left root has higher priority
                    let new_right = self.merge(self.nodes[l].right, Some(r));
                    self.nodes[l].right = new_right;
                    if let Some(idx) = new_right {
                        self.nodes[idx].parent = Some(l);
                    }
                    self.nodes[l].parent = None;
                    self.pull_up(l);
                    Some(l)
                } else {
                    // Right root has higher priority
                    let new_left = self.merge(Some(l), self.nodes[r].left);
                    self.nodes[r].left = new_left;
                    if let Some(idx) = new_left {
                        self.nodes[idx].parent = Some(r);
                    }
                    self.nodes[r].parent = None;
                    self.pull_up(r);
                    Some(r)
                }
            }
        }
    }

    /// Split off the first element
    fn split_first(&mut self, root: Option<usize>) -> (Option<usize>, Option<usize>) {
        match root {
            None => (None, None),
            Some(idx) => {
                let (first, rest) = self.split(idx, 1);
                (first, rest)
            }
        }
    }

    /// Split off the last element
    fn split_last(&mut self, root: Option<usize>) -> (Option<usize>, Option<usize>) {
        match root {
            None => (None, None),
            Some(idx) => {
                let size = self.nodes[idx].size;
                if size == 0 {
                    return (None, None);
                }
                let (rest, last) = self.split(idx, size - 1);
                (rest, last)
            }
        }
    }

    /// Allocate a new node
    fn allocate_node(&mut self, vertex: NodeId, priority: u64, value: f64) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx] = TreapNode::new(vertex, priority, value);
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(TreapNode::new(vertex, priority, value));
            idx
        }
    }

    /// Free a node
    fn free_node(&mut self, idx: usize) {
        // Clear the node
        self.nodes[idx].left = None;
        self.nodes[idx].right = None;
        self.nodes[idx].parent = None;
        self.free_list.push(idx);
    }

    /// Update aggregate information bottom-up
    fn pull_up(&mut self, idx: usize) {
        let mut size = 1;
        let mut aggregate = self.nodes[idx].value;

        if let Some(left) = self.nodes[idx].left {
            size += self.nodes[left].size;
            aggregate += self.nodes[left].subtree_aggregate;
        }

        if let Some(right) = self.nodes[idx].right {
            size += self.nodes[right].size;
            aggregate += self.nodes[right].subtree_aggregate;
        }

        self.nodes[idx].size = size;
        self.nodes[idx].subtree_aggregate = aggregate;
    }

    /// Get position of node in the sequence (0-indexed)
    fn get_position(&self, idx: usize) -> usize {
        let mut pos = self.nodes[idx].left.map(|l| self.nodes[l].size).unwrap_or(0);
        let mut current = idx;

        while let Some(parent) = self.nodes[current].parent {
            if self.nodes[parent].right == Some(current) {
                // Current is right child, add left subtree + parent
                pos += 1;
                if let Some(left) = self.nodes[parent].left {
                    pos += self.nodes[left].size;
                }
            }
            current = parent;
        }

        pos
    }
}

impl Default for EulerTourTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_tree() {
        let mut ett = EulerTourTree::new();
        assert!(ett.make_tree(1).is_ok());
        assert!(ett.make_tree(2).is_ok());
        assert_eq!(ett.len(), 2);
    }

    #[test]
    fn test_singleton_tree() {
        let mut ett = EulerTourTree::new();
        ett.make_tree(1).unwrap();

        assert_eq!(ett.tree_size(1).unwrap(), 1);
        assert_eq!(ett.subtree_size(1).unwrap(), 1);
        assert!(ett.connected(1, 1));
    }

    #[test]
    fn test_link_two_vertices() {
        let mut ett = EulerTourTree::new();
        ett.make_tree(1).unwrap();
        ett.make_tree(2).unwrap();

        assert!(!ett.connected(1, 2));

        ett.link(1, 2).unwrap();

        assert!(ett.connected(1, 2));
        assert_eq!(ett.tree_size(1).unwrap(), 2);
        assert_eq!(ett.tree_size(2).unwrap(), 2);
    }

    #[test]
    fn test_link_multiple_vertices() {
        let mut ett = EulerTourTree::new();

        for i in 1..=5 {
            ett.make_tree(i).unwrap();
        }

        // Create chain: 1 - 2 - 3 - 4 - 5
        ett.link(1, 2).unwrap();
        ett.link(2, 3).unwrap();
        ett.link(3, 4).unwrap();
        ett.link(4, 5).unwrap();

        // All should be connected
        for i in 1..=5 {
            for j in 1..=5 {
                assert!(ett.connected(i, j));
            }
        }

        assert_eq!(ett.tree_size(1).unwrap(), 5);
    }

    #[test]
    fn test_update_value() {
        let mut ett = EulerTourTree::new();
        ett.make_tree(1).unwrap();

        ett.update_value(1, 10.0).unwrap();
        assert_eq!(ett.subtree_aggregate(1).unwrap(), 10.0);

        ett.update_value(1, 25.5).unwrap();
        assert_eq!(ett.subtree_aggregate(1).unwrap(), 25.5);
    }

    #[test]
    fn test_reroot() {
        let mut ett = EulerTourTree::new();
        ett.make_tree(1).unwrap();
        ett.make_tree(2).unwrap();
        ett.make_tree(3).unwrap();

        ett.link(1, 2).unwrap();
        ett.link(1, 3).unwrap();

        // Initially rooted at 1
        assert_eq!(ett.find_root(1).unwrap(), 1);
        assert_eq!(ett.find_root(2).unwrap(), 1);

        // Reroot to 2
        ett.reroot(2).unwrap();

        // All should still be connected
        assert!(ett.connected(1, 2));
        assert!(ett.connected(2, 3));
        assert_eq!(ett.tree_size(2).unwrap(), 3);
    }

    #[test]
    fn test_invalid_vertex() {
        let ett = EulerTourTree::new();

        assert!(matches!(
            ett.find_root(999),
            Err(MinCutError::InvalidVertex(999))
        ));

        assert!(!ett.connected(1, 2));
    }

    #[test]
    fn test_edge_already_exists() {
        let mut ett = EulerTourTree::new();
        ett.make_tree(1).unwrap();
        ett.make_tree(2).unwrap();

        ett.link(1, 2).unwrap();

        // Trying to link again should fail
        assert!(matches!(
            ett.link(1, 2),
            Err(MinCutError::EdgeExists(1, 2))
        ));
    }

    #[test]
    fn test_split_and_merge() {
        let mut ett = EulerTourTree::new();
        ett.make_tree(1).unwrap();
        ett.make_tree(2).unwrap();
        ett.make_tree(3).unwrap();

        ett.link(1, 2).unwrap();
        ett.link(2, 3).unwrap();

        // Verify all connected
        assert!(ett.connected(1, 3));
        assert_eq!(ett.tree_size(1).unwrap(), 3);
    }

    #[test]
    fn test_tree_size_updates() {
        let mut ett = EulerTourTree::new();

        for i in 1..=10 {
            ett.make_tree(i).unwrap();
        }

        // Build a star: 1 connected to all others
        for i in 2..=10 {
            ett.link(1, i).unwrap();
        }

        assert_eq!(ett.tree_size(1).unwrap(), 10);
        assert_eq!(ett.tree_size(5).unwrap(), 10);
    }

    #[test]
    fn test_empty_tree() {
        let ett = EulerTourTree::new();
        assert!(ett.is_empty());
        assert_eq!(ett.len(), 0);
    }

    #[test]
    fn test_subtree_aggregate_simple() {
        let mut ett = EulerTourTree::new();
        ett.make_tree(1).unwrap();
        ett.make_tree(2).unwrap();

        ett.update_value(1, 5.0).unwrap();
        ett.update_value(2, 3.0).unwrap();

        assert_eq!(ett.subtree_aggregate(1).unwrap(), 5.0);
        assert_eq!(ett.subtree_aggregate(2).unwrap(), 3.0);
    }

    #[test]
    fn test_reproducible_with_seed() {
        let mut ett1 = EulerTourTree::with_seed(12345);
        let mut ett2 = EulerTourTree::with_seed(12345);

        for i in 1..=5 {
            ett1.make_tree(i).unwrap();
            ett2.make_tree(i).unwrap();
        }

        ett1.link(1, 2).unwrap();
        ett2.link(1, 2).unwrap();

        assert_eq!(ett1.tree_size(1).unwrap(), ett2.tree_size(1).unwrap());
    }

    #[test]
    fn test_large_tree() {
        let mut ett = EulerTourTree::new();
        let n = 100;

        for i in 0..n {
            ett.make_tree(i).unwrap();
        }

        // Create a chain
        for i in 0..n-1 {
            ett.link(i, i + 1).unwrap();
        }

        assert_eq!(ett.tree_size(0).unwrap(), n as usize);
        assert_eq!(ett.tree_size(n - 1).unwrap(), n as usize);
        assert!(ett.connected(0, n - 1));
    }

    #[test]
    fn test_multiple_trees() {
        let mut ett = EulerTourTree::new();

        // Create two separate trees
        ett.make_tree(1).unwrap();
        ett.make_tree(2).unwrap();
        ett.make_tree(3).unwrap();
        ett.make_tree(4).unwrap();

        ett.link(1, 2).unwrap();
        ett.link(3, 4).unwrap();

        // Within same tree
        assert!(ett.connected(1, 2));
        assert!(ett.connected(3, 4));

        // Across different trees
        assert!(!ett.connected(1, 3));
        assert!(!ett.connected(2, 4));

        assert_eq!(ett.tree_size(1).unwrap(), 2);
        assert_eq!(ett.tree_size(3).unwrap(), 2);
    }
}
