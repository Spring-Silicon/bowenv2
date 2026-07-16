use super::math::softmax;
use super::types::{MctsConfig, MctsHandleBatch};
use crate::support::internal;
use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    CandidateHash, EngineResult, ModelVersion, PortableCandidateRef, PortableSearchActionRef,
    ReplayGraphContext,
};
use gz_eval::EvalAction;

pub(crate) struct MctsTree<G, C> {
    pub(crate) config: MctsConfig,
    pub(crate) nodes: Vec<MctsNode<G, C>>,
    pub(crate) eval_count: usize,
    pub(crate) portable_contexts: usize,
    pub(crate) carried_nodes: usize,
    pub(crate) carried_root_visits: u32,
}

impl<G, C> MctsTree<G, C>
where
    G: Copy,
    C: Copy,
{
    pub(crate) fn new(config: MctsConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            eval_count: 0,
            portable_contexts: 0,
            carried_nodes: 0,
            carried_root_visits: 0,
        }
    }

    pub(crate) fn compact_subtree(
        &self,
        root_index: usize,
    ) -> EngineResult<(Self, ReplayGraphContext, MctsHandleBatch<G, C>)> {
        let mut remap = vec![None; self.nodes.len()];
        let mut old_indices = Vec::new();
        let mut stack = vec![root_index];

        while let Some(index) = stack.pop() {
            if remap[index].is_some() {
                continue;
            }
            remap[index] = Some(old_indices.len());
            old_indices.push(index);
            for child in self.nodes[index].children.iter().rev().flatten() {
                stack.push(*child);
            }
        }

        let mut nodes = Vec::with_capacity(old_indices.len());
        let mut handles = MctsHandleBatch::default();
        let mut carried_root_visits = 0;
        for (new_index, &old_index) in old_indices.iter().enumerate() {
            let mut node = self.nodes[old_index].clone();
            handles.graphs.push(node.graph);
            handles.candidates.extend(node.candidates.iter().copied());
            for child in &mut node.children {
                if let Some(old_child) = *child {
                    *child = remap[old_child];
                }
            }
            if new_index == 0 {
                carried_root_visits = node.visits.iter().sum();
            }
            nodes.push(node);
        }

        let root_context = nodes
            .first()
            .map(|node| node.context)
            .ok_or_else(|| internal("empty reused subtree"))?;
        let carried_nodes = nodes.len();
        Ok((
            Self {
                config: self.config,
                nodes,
                eval_count: 0,
                portable_contexts: 0,
                carried_nodes,
                carried_root_visits,
            },
            root_context,
            handles,
        ))
    }

    pub(crate) fn backup(&mut self, path: &[MctsEdge], value: f32) {
        for edge in path {
            let node = &mut self.nodes[edge.node_index];
            node.visits[edge.action] += 1;
            node.value_sum[edge.action] += value;
            node.q[edge.action] = node.value_sum[edge.action] / node.visits[edge.action] as f32;
        }
    }

    pub(crate) fn mask_action(&mut self, node_index: usize, action: usize) {
        let node = &mut self.nodes[node_index];
        node.masked[action] = true;
        if node
            .masked
            .iter()
            .take(node.candidates.len())
            .all(|masked| *masked)
        {
            node.masked[node.candidates.len()] = false;
        }
    }

    pub(crate) fn selected_after(&self, node_index: usize, action: usize) -> EngineResult<G> {
        let node = &self.nodes[node_index];
        if node.is_stop(action) {
            return Ok(node.graph);
        }
        let child = node.children[action].ok_or_else(|| internal("missing selected child"))?;
        Ok(self.nodes[child].graph)
    }

    pub(crate) fn selected_after_context(
        &self,
        node_index: usize,
        action: usize,
    ) -> EngineResult<ReplayGraphContext> {
        let node = &self.nodes[node_index];
        if node.is_stop(action) {
            return Ok(node.context);
        }
        let child = node.children[action].ok_or_else(|| internal("missing selected child"))?;
        Ok(self.nodes[child].context)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MctsNode<G, C> {
    pub(crate) graph: G,
    pub(crate) context: ReplayGraphContext,
    pub(crate) candidates: Vec<C>,
    pub(crate) eval_actions: Vec<EvalAction>,
    pub(crate) candidate_hashes: Vec<CandidateHash>,
    pub(crate) summaries: Vec<Option<SearchCandidateSummary>>,
    pub(crate) logits: Vec<f32>,
    pub(crate) priors: Vec<f32>,
    pub(crate) value: f32,
    pub(crate) model_version: ModelVersion,
    pub(crate) children: Vec<Option<usize>>,
    pub(crate) visits: Vec<u32>,
    pub(crate) value_sum: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) masked: Vec<bool>,
}

impl<G, C> MctsNode<G, C>
where
    C: Copy,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        graph: G,
        context: ReplayGraphContext,
        candidates: Vec<C>,
        eval_actions: Vec<EvalAction>,
        candidate_hashes: Vec<CandidateHash>,
        summaries: Vec<Option<SearchCandidateSummary>>,
        output: gz_eval::EvalOutput,
        mask_stop: bool,
    ) -> Self {
        let priors = softmax(&output.policy_logits);
        let action_count = output.policy_logits.len();
        let mut masked = vec![false; action_count];
        if mask_stop && !candidates.is_empty() {
            masked[candidates.len()] = true;
        }
        Self {
            graph,
            context,
            candidates,
            eval_actions,
            candidate_hashes,
            summaries,
            logits: output.policy_logits,
            priors,
            value: output.value,
            model_version: output.model_version,
            children: vec![None; action_count],
            visits: vec![0; action_count],
            value_sum: vec![0.0; action_count],
            q: vec![0.0; action_count],
            masked,
        }
    }

    pub(crate) fn action_count(&self) -> usize {
        self.logits.len()
    }

    pub(crate) fn is_stop(&self, action: usize) -> bool {
        action == self.candidates.len()
    }

    pub(crate) fn search_action(&self, action: usize) -> EngineResult<SearchAction<C>> {
        if self.is_stop(action) {
            Ok(SearchAction::Stop)
        } else {
            self.candidates
                .get(action)
                .copied()
                .map(SearchAction::Candidate)
                .ok_or_else(|| internal("invalid selected action"))
        }
    }

    pub(crate) fn action_ref(&self, action: usize) -> EngineResult<PortableSearchActionRef> {
        if self.is_stop(action) {
            Ok(PortableSearchActionRef::stop(self.context))
        } else {
            self.candidate_hashes
                .get(action)
                .copied()
                .map(|hash| {
                    PortableSearchActionRef::candidate(PortableCandidateRef::new(
                        self.context,
                        hash,
                    ))
                })
                .ok_or_else(|| internal("invalid selected action"))
        }
    }

    pub(crate) fn action_refs(&self) -> Vec<PortableSearchActionRef> {
        let mut refs = Vec::with_capacity(self.candidate_hashes.len() + 1);
        refs.extend(self.candidate_hashes.iter().copied().map(|hash| {
            PortableSearchActionRef::candidate(PortableCandidateRef::new(self.context, hash))
        }));
        refs.push(PortableSearchActionRef::stop(self.context));
        refs
    }

    pub(crate) fn unmasked_actions(&self) -> impl Iterator<Item = usize> + '_ {
        self.masked
            .iter()
            .enumerate()
            .filter_map(|(index, masked)| (!*masked).then_some(index))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MctsEdge {
    pub(crate) node_index: usize,
    pub(crate) action: usize,
}
