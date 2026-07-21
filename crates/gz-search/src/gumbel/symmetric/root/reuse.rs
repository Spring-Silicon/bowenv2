use super::*;

impl<G, C> SymmetricSelfplayRootTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub(super) fn prepare_reuse(
        &mut self,
        selected: usize,
        expected: Board<G>,
    ) -> EngineResult<()> {
        if !self.config.tree_reuse {
            return Ok(());
        }
        if self.reused.is_some() {
            return Err(internal("symmetric reused tree already prepared"));
        }
        let Some(branch) = self.nodes[0].edges[selected].branch else {
            return Ok(());
        };
        let BranchTarget::Node(root_index) = branch.target else {
            return Ok(());
        };
        let root = self
            .nodes
            .get(root_index)
            .ok_or_else(|| internal("invalid symmetric reused root index"))?;
        if !same_board(root.board, expected) {
            return Ok(());
        }

        let nodes = self.compact_subtree(root_index)?;
        let carried_root_visits = nodes[0].total_visits();
        let carried_nodes = nodes.len();
        let mut referenced_graphs = HashSet::new();
        let mut referenced_candidates = HashSet::new();
        for node in &nodes {
            referenced_graphs.extend(node.board.graphs);
            referenced_candidates.extend(node.candidates.iter().copied());
            referenced_graphs.extend(node.edges.iter().filter_map(|edge| edge.after_graph));
        }

        // Preserve the promoted root's statistics. The next run snapshots
        // these counts as its baseline: allocation uses fresh-visit deltas,
        // while Q estimates and the policy target use the carried aggregate.

        let mut created_graphs = Vec::new();
        for graph in self.created_graphs.drain(..) {
            if referenced_graphs.contains(&graph) {
                created_graphs.push(graph);
            } else {
                self.releasable.graphs.push(graph);
            }
        }
        let mut created_candidates = Vec::new();
        for candidate in self.created_candidates.drain(..) {
            if referenced_candidates.contains(&candidate) {
                created_candidates.push(candidate);
            } else {
                self.releasable.candidates.push(candidate);
            }
        }
        self.reused = Some(ReusedTree {
            nodes,
            created_graphs,
            created_candidates,
            carried_nodes,
            carried_root_visits,
        });
        Ok(())
    }

    fn compact_subtree(&self, root_index: usize) -> EngineResult<Vec<Node<G, C>>> {
        let mut remap = vec![None; self.nodes.len()];
        let mut old_indices = Vec::new();
        let mut stack = vec![root_index];
        while let Some(index) = stack.pop() {
            let node = self
                .nodes
                .get(index)
                .ok_or_else(|| internal("invalid symmetric subtree node index"))?;
            if remap[index].is_some() {
                continue;
            }
            remap[index] = Some(old_indices.len());
            old_indices.push(index);
            for edge in node.edges.iter().rev() {
                if let Some(Branch {
                    target: BranchTarget::Node(child),
                    ..
                }) = edge.branch
                {
                    stack.push(child);
                }
            }
            if let Some(Branch {
                target: BranchTarget::Node(child),
                ..
            }) = node.pass
            {
                stack.push(child);
            }
        }

        let mut nodes = Vec::with_capacity(old_indices.len());
        for old_index in old_indices {
            let mut node = self.nodes[old_index].clone();
            for edge in &mut node.edges {
                if let Some(branch) = &mut edge.branch {
                    remap_branch(branch, &remap)?;
                }
            }
            if let Some(branch) = &mut node.pass {
                remap_branch(branch, &remap)?;
            }
            nodes.push(node);
        }
        Ok(nodes)
    }

    pub(super) fn transfer_selected_graph(&mut self, selected: G) -> EngineResult<()> {
        if let Some(reused) = &mut self.reused
            && let Some(index) = reused
                .created_graphs
                .iter()
                .position(|graph| *graph == selected)
        {
            reused.created_graphs.swap_remove(index);
            return Ok(());
        }
        let index = self
            .created_graphs
            .iter()
            .position(|graph| *graph == selected)
            .ok_or_else(|| internal("selected symmetric graph is not owned"))?;
        self.created_graphs.swap_remove(index);
        Ok(())
    }
}
