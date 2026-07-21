use super::*;

impl<G, C> SymmetricSelfplayRootTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub(super) fn poll_wave(
        &mut self,
        mut wave: WaveRun<G, C>,
    ) -> EngineResult<SearchPoll<G, C, SymmetricRootResult<G, C>>> {
        loop {
            if wave.active.is_empty() && !self.launch_wave(&mut wave) {
                return self.finish_root(wave.run);
            }
            if wave.root_apply_preflight_ready() {
                self.resolve_root_apply_preflight(&mut wave)?;
            }

            let defer_completions = wave.any_pending();
            let mut progressed = false;
            for index in 0..wave.active.len() {
                match self.advance_wave_simulation(&mut wave.active[index], defer_completions)? {
                    WaveAdvance::Work(work) => {
                        self.state = State::WaveRunning(wave);
                        return Ok(SearchPoll::Work(work));
                    }
                    WaveAdvance::Progressed => progressed = true,
                    WaveAdvance::Waiting => {}
                }
            }

            if wave.any_pending() {
                self.state = State::WaveRunning(wave);
                return Ok(SearchPoll::Blocked);
            }

            if wave
                .active
                .iter()
                .all(|simulation| matches!(simulation.state, WaveSimulationState::Outcome { .. }))
            {
                for simulation in wave.active.drain(..) {
                    let WaveSimulationState::Outcome { descent, value } = simulation.state else {
                        unreachable!("checked wave outcome state");
                    };
                    self.backup_descent(&descent, value);
                    wave.run.simulations += 1;
                    wave.run.schedule_index += 1;
                }
                continue;
            }

            if !progressed {
                self.state = State::WaveRunning(wave);
                return Err(internal("symmetric wave made no progress"));
            }
        }
    }

    fn launch_wave(&self, wave: &mut WaveRun<G, C>) -> bool {
        let Some(&target) = wave.run.schedule.get(wave.run.schedule_index) else {
            return false;
        };
        self.replenish_considered(&mut wave.run);
        let width = wave.run.schedule[wave.run.schedule_index..]
            .iter()
            .take_while(|&&visit| visit == target)
            .count();
        let root = &self.nodes[0];
        let mut virtual_visits = root
            .edges
            .iter()
            .map(|edge| edge.visits)
            .collect::<Vec<_>>();

        let mut actions = Vec::with_capacity(width);
        for _ in 0..width {
            let Some(action) = self.wave_root_action(&wave.run, target, &virtual_visits) else {
                break;
            };
            if actions.contains(&action) {
                break;
            }
            actions.push(action);
            virtual_visits[action] = virtual_visits[action].saturating_add(1);
        }

        // STOP performs no engine apply. Keep it at a wave boundary so the
        // speculative first-visit preflight remains candidate-only and the
        // sequential schedule order is preserved exactly.
        if let Some(stop) = actions.iter().position(|&action| root.is_stop(action)) {
            actions.truncate(stop.max(1));
        }

        let unexpanded = actions
            .iter()
            .filter(|&&action| root.edges[action].branch.is_none())
            .count();
        if unexpanded > 0 && unexpanded < actions.len() {
            actions.truncate(1);
        }
        wave.root_apply_preflight = actions.len() > 1 && unexpanded == actions.len();
        wave.active
            .extend(actions.into_iter().map(|action| WaveSimulation {
                state: WaveSimulationState::Running(self.new_descent(action)),
            }));

        !wave.active.is_empty()
    }

    fn replenish_considered(&self, run: &mut Run) {
        let root = &self.nodes[0];
        let desired = root
            .masked
            .iter()
            .filter(|masked| !**masked)
            .count()
            .min(self.config.max_considered_actions.get());
        let current = run
            .considered
            .iter()
            .filter(|&&action| !root.masked[action])
            .count();
        if current >= desired {
            return;
        }

        let mut replacements = (0..root.action_count())
            .filter(|&action| !root.masked[action] && !run.considered.contains(&action))
            .collect::<Vec<_>>();
        replacements.sort_by(|&left, &right| {
            run.base_scores[right]
                .total_cmp(&run.base_scores[left])
                .then_with(|| left.cmp(&right))
        });
        run.considered
            .extend(replacements.into_iter().take(desired - current));
    }

    fn resolve_root_apply_preflight(&mut self, wave: &mut WaveRun<G, C>) -> EngineResult<()> {
        let all_safe = wave.active.iter().all(|simulation| {
            let WaveSimulationState::RootApplyReady(ready) = &simulation.state else {
                return false;
            };
            self.root_apply_safe(ready)
        });
        let ready = wave
            .active
            .iter_mut()
            .map(|simulation| {
                let state = std::mem::replace(&mut simulation.state, WaveSimulationState::Vacant);
                let WaveSimulationState::RootApplyReady(ready) = state else {
                    return Err(internal("incomplete symmetric root apply preflight"));
                };
                Ok(ready)
            })
            .collect::<EngineResult<Vec<_>>>()?;
        wave.root_apply_preflight = false;

        if all_safe {
            // Distinct successful root applies are independent; install them
            // in schedule order before their branches advance concurrently.
            for (simulation, ready) in wave.active.iter_mut().zip(ready) {
                self.consume_speculative_graph(ready.applied.after)?;
                simulation.state =
                    self.resume_wave_apply(ready.descent, ready.node, ready.action, ready.applied)?;
            }
            return Ok(());
        }

        // A rejection can retarget the first simulation. Discard later
        // speculation and resume through the existing sequential path.
        let mut ready = ready.into_iter();
        let first = ready
            .next()
            .ok_or_else(|| internal("empty symmetric root apply preflight"))?;
        for discarded in ready {
            self.discard_speculative_graph(discarded.applied.after)?;
        }
        wave.active.truncate(1);
        self.consume_speculative_graph(first.applied.after)?;
        wave.active[0].state =
            self.resume_wave_apply(first.descent, first.node, first.action, first.applied)?;
        Ok(())
    }

    fn root_apply_safe(&self, ready: &RootApplyReady<G, C>) -> bool {
        if ready.applied.rejected.is_some() {
            return false;
        }
        let player = self.nodes[ready.node].board.player;
        let context = self.identity.context(ready.applied.after_hash);
        !self.config.no_backtrack || !ready.descent.seen[player.index()].contains(&context)
    }

    fn consume_speculative_graph(&mut self, graph: G) -> EngineResult<()> {
        let index = self
            .speculative_graphs
            .iter()
            .position(|candidate| *candidate == graph)
            .ok_or_else(|| internal("untracked speculative symmetric graph"))?;
        self.speculative_graphs.swap_remove(index);
        Ok(())
    }

    fn discard_speculative_graph(&mut self, graph: G) -> EngineResult<()> {
        self.consume_speculative_graph(graph)?;
        self.releasable.graphs.push(graph);
        Ok(())
    }

    fn wave_root_action(&self, run: &Run, target: u32, visits: &[u32]) -> Option<usize> {
        let root = &self.nodes[0];
        let scale = (self.config.c_visit + visits.iter().copied().max().unwrap_or(0) as f32)
            * self.config.c_scale;
        let completed_q = self.completed_q(0);
        let scores = run
            .base_scores
            .iter()
            .zip(completed_q)
            .map(|(base, q)| base + scale * q)
            .collect::<Vec<_>>();
        run.considered
            .iter()
            .copied()
            .filter(|&action| {
                !root.masked[action]
                    && visits[action] == run.root_visit_baseline[action].saturating_add(target)
            })
            .max_by(|&left, &right| {
                scores[left]
                    .total_cmp(&scores[right])
                    .then_with(|| right.cmp(&left))
            })
            .or_else(|| {
                run.considered
                    .iter()
                    .copied()
                    .filter(|&action| !root.masked[action])
                    .max_by(|&left, &right| {
                        scores[left]
                            .total_cmp(&scores[right])
                            .then_with(|| right.cmp(&left))
                    })
            })
    }

    fn new_descent(&self, action: usize) -> Descent {
        let root = &self.nodes[0];
        let mut seen = self.visited.clone();
        for player in [GumbelPlayer::One, GumbelPlayer::Two] {
            if let Some(context) = root.board.contexts[player.index()] {
                seen[player.index()].insert(context);
            }
        }
        Descent {
            node: 0,
            path: Vec::new(),
            seen,
            forced: Some(action),
        }
    }

    fn advance_wave_simulation(
        &mut self,
        simulation: &mut WaveSimulation<G, C>,
        defer_completions: bool,
    ) -> EngineResult<WaveAdvance<G, C>> {
        let state = std::mem::replace(&mut simulation.state, WaveSimulationState::Vacant);
        match state {
            WaveSimulationState::Running(mut descent) => {
                let node_index = descent.node;
                let action = match descent.forced.take() {
                    Some(action) => Some(action),
                    None => self.select_nonroot(node_index),
                };
                let Some(action) = action else {
                    if let Some(branch) = self.nodes[node_index].pass {
                        descent.path.push(PathStep::Transform { flip: branch.flip });
                        simulation.state = wave_target_state(descent, branch.target);
                    } else {
                        let mut board = self.nodes[node_index].board;
                        let player = board.player;
                        board.inactive[player.index()] = true;
                        board.player = player.opponent();
                        simulation.state = WaveSimulationState::Resolve {
                            descent,
                            board,
                            attach: Attach {
                                slot: AttachSlot::Pass { node: node_index },
                                turns: 1,
                            },
                        };
                    }
                    return Ok(WaveAdvance::Progressed);
                };

                if let Some(branch) = self.nodes[node_index].edges[action].branch {
                    let player = self.nodes[node_index].board.player;
                    if let Some(context) = self.nodes[node_index].edges[action].after_context {
                        if self.config.no_backtrack
                            && descent.seen[player.index()].contains(&context)
                        {
                            self.mask_action(node_index, action);
                            simulation.state = WaveSimulationState::Running(descent);
                            return Ok(WaveAdvance::Progressed);
                        }
                        descent.seen[player.index()].insert(context);
                    }
                    descent.path.push(PathStep::Decision {
                        node: node_index,
                        action,
                        flip: branch.flip,
                    });
                    simulation.state = wave_target_state(descent, branch.target);
                    return Ok(WaveAdvance::Progressed);
                }

                if self.nodes[node_index].is_stop(action) {
                    let board = self.stopped_board(node_index);
                    simulation.state = WaveSimulationState::Resolve {
                        descent,
                        board,
                        attach: Attach {
                            slot: AttachSlot::Action {
                                node: node_index,
                                action,
                            },
                            turns: 1,
                        },
                    };
                    return Ok(WaveAdvance::Progressed);
                }

                let token = self.next_token();
                let node = &self.nodes[node_index];
                let work = SearchWork::Apply(ApplyWork {
                    token,
                    graph: node.board.current_graph(),
                    candidate: node.candidates[action],
                });
                simulation.state = WaveSimulationState::ApplyPending {
                    token,
                    descent,
                    node: node_index,
                    action,
                };
                Ok(WaveAdvance::Work(work))
            }
            WaveSimulationState::Resolve {
                descent,
                mut board,
                mut attach,
            } => {
                while !board.active(self.config.max_steps, board.player) {
                    if board.terminal(self.config.max_steps) {
                        let player = board.player;
                        let context = board.contexts[player.index()]
                            .ok_or_else(|| internal("terminal symmetric board has no context"))?;
                        let request = EvalRequest::with_position(
                            context,
                            vec![EvalAction::stop(context)],
                            self.board_position(board, player),
                        )
                        .map_err(|_| internal("invalid terminal symmetric eval request"))?;
                        let opponent = player.opponent();
                        let token = self.next_token();
                        let work = SearchWork::Eval(EvalWork {
                            token,
                            graph: board.graphs[player.index()],
                            candidates: Vec::new(),
                            request: request.clone(),
                            measure_options: self.config.measure_options,
                            opponent: Some(Box::new(EvalOpponentWork {
                                graph: board.graphs[opponent.index()],
                                position: self.board_position(board, opponent),
                            })),
                        });
                        simulation.state = WaveSimulationState::TerminalEvalPending {
                            token,
                            descent,
                            attach,
                            request,
                        };
                        return Ok(WaveAdvance::Work(work));
                    }
                    board.player = board.player.opponent();
                    attach.turns = attach.turns.saturating_add(1);
                }
                let token = self.next_token();
                let work = SearchWork::Expand(ExpandWork {
                    token,
                    graph: board.current_graph(),
                    options: self.config.candidate_options,
                });
                simulation.state = WaveSimulationState::ExpandPending {
                    token,
                    descent,
                    board,
                    attach,
                };
                Ok(WaveAdvance::Work(work))
            }
            WaveSimulationState::EvalReady {
                descent,
                board,
                attach,
                expansion,
            } => {
                let player = board.player;
                let request = EvalRequest::with_position(
                    expansion.context,
                    expansion.eval_actions.clone(),
                    self.board_position(board, player),
                )
                .map_err(|_| internal("invalid symmetric wave eval request"))?;
                let opponent = player.opponent();
                let token = self.next_token();
                let work = SearchWork::Eval(EvalWork {
                    token,
                    graph: board.graphs[player.index()],
                    candidates: expansion
                        .candidates
                        .iter()
                        .map(|candidate| candidate.handle)
                        .collect(),
                    request: request.clone(),
                    measure_options: self.config.measure_options,
                    opponent: Some(Box::new(EvalOpponentWork {
                        graph: board.graphs[opponent.index()],
                        position: self.board_position(board, opponent),
                    })),
                });
                simulation.state = WaveSimulationState::EvalPending {
                    token,
                    descent,
                    board,
                    attach,
                    expansion,
                    request,
                };
                Ok(WaveAdvance::Work(work))
            }
            WaveSimulationState::EvalComplete {
                descent,
                board,
                attach,
                expansion,
                request,
                output,
            } => {
                if defer_completions {
                    simulation.state = WaveSimulationState::EvalComplete {
                        descent,
                        board,
                        attach,
                        expansion,
                        request,
                        output,
                    };
                    return Ok(WaveAdvance::Waiting);
                }
                simulation.state =
                    self.complete_wave_eval(descent, board, attach, expansion, request, output)?;
                Ok(WaveAdvance::Progressed)
            }
            WaveSimulationState::TerminalEvalComplete {
                descent,
                attach,
                request,
                output,
            } => {
                if defer_completions {
                    simulation.state = WaveSimulationState::TerminalEvalComplete {
                        descent,
                        attach,
                        request,
                        output,
                    };
                    return Ok(WaveAdvance::Waiting);
                }
                output
                    .validate_for(&request)
                    .map_err(eval_error_to_engine_error)?;
                self.eval_count += 1;
                let mut descent = descent;
                let value =
                    self.attach_target(&mut descent, attach, BranchTarget::Terminal(output.value))?;
                simulation.state = WaveSimulationState::Outcome { descent, value };
                Ok(WaveAdvance::Progressed)
            }
            WaveSimulationState::Outcome { descent, value } => {
                simulation.state = WaveSimulationState::Outcome { descent, value };
                Ok(WaveAdvance::Waiting)
            }
            state @ WaveSimulationState::RootApplyReady(_) => {
                simulation.state = state;
                Ok(WaveAdvance::Waiting)
            }
            state @ (WaveSimulationState::ApplyPending { .. }
            | WaveSimulationState::ExpandPending { .. }
            | WaveSimulationState::EvalPending { .. }
            | WaveSimulationState::TerminalEvalPending { .. }) => {
                simulation.state = state;
                Ok(WaveAdvance::Waiting)
            }
            WaveSimulationState::Vacant => Err(internal("vacant symmetric wave simulation")),
        }
    }

    pub(super) fn resume_wave(
        &mut self,
        wave: &mut WaveRun<G, C>,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()> {
        let root_apply_preflight = wave.root_apply_preflight;
        let simulation = wave
            .active
            .iter_mut()
            .find(|simulation| simulation.state.pending_token() == Some(token))
            .ok_or_else(|| internal("unknown symmetric wave token"))?;
        let state = std::mem::replace(&mut simulation.state, WaveSimulationState::Vacant);
        simulation.state = match (state, result) {
            (
                WaveSimulationState::ApplyPending {
                    descent,
                    node,
                    action,
                    ..
                },
                SearchWorkResult::Apply(applied),
            ) => {
                if root_apply_preflight {
                    self.speculative_graphs.push(applied.after);
                    WaveSimulationState::RootApplyReady(RootApplyReady {
                        descent,
                        node,
                        action,
                        applied,
                    })
                } else {
                    self.resume_wave_apply(descent, node, action, applied)?
                }
            }
            (
                WaveSimulationState::ExpandPending {
                    descent,
                    board,
                    attach,
                    ..
                },
                SearchWorkResult::Expand(expansion),
            ) => self.resume_wave_expand(descent, board, attach, expansion)?,
            (
                WaveSimulationState::EvalPending {
                    descent,
                    board,
                    attach,
                    expansion,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => WaveSimulationState::EvalComplete {
                descent,
                board,
                attach,
                expansion,
                request,
                output,
            },
            (
                WaveSimulationState::TerminalEvalPending {
                    descent,
                    attach,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => WaveSimulationState::TerminalEvalComplete {
                descent,
                attach,
                request,
                output,
            },
            (state, _) => {
                simulation.state = state;
                return Err(internal("mismatched symmetric wave result"));
            }
        };
        Ok(())
    }

    fn resume_wave_apply(
        &mut self,
        mut descent: Descent,
        node_index: usize,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<WaveSimulationState<G, C>> {
        if applied.rejected.is_some() {
            self.releasable.graphs.push(applied.after);
            self.mask_action(node_index, action);
            return Ok(WaveSimulationState::Running(descent));
        }
        let player = self.nodes[node_index].board.player;
        let context = self.identity.context(applied.after_hash);
        if self.config.no_backtrack && descent.seen[player.index()].contains(&context) {
            self.releasable.graphs.push(applied.after);
            self.mask_action(node_index, action);
            return Ok(WaveSimulationState::Running(descent));
        }
        descent.seen[player.index()].insert(context);
        self.created_graphs.push(applied.after);
        let edge = &mut self.nodes[node_index].edges[action];
        edge.after_graph = Some(applied.after);
        edge.after_context = Some(context);
        let mut board = self.nodes[node_index].board;
        board.graphs[player.index()] = applied.after;
        board.contexts[player.index()] = Some(context);
        board.rewrites[player.index()] += 1;
        board.player = player.opponent();
        Ok(WaveSimulationState::Resolve {
            descent,
            board,
            attach: Attach {
                slot: AttachSlot::Action {
                    node: node_index,
                    action,
                },
                turns: 1,
            },
        })
    }

    fn resume_wave_expand(
        &mut self,
        descent: Descent,
        mut board: Board<G>,
        mut attach: Attach,
        result: ExpandResult<C>,
    ) -> EngineResult<WaveSimulationState<G, C>> {
        self.created_candidates.extend(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.candidate),
        );
        let context = self.identity.context(result.graph_hash);
        let player = board.player;
        if let Some(expected) = board.contexts[player.index()]
            && expected != context
        {
            return Err(internal("symmetric wave graph context mismatch"));
        }
        board.contexts[player.index()] = Some(context);
        if board.graphs[0] == board.graphs[1] {
            board.contexts[player.opponent().index()].get_or_insert(context);
        }

        if result.candidates.is_empty() {
            board.inactive[player.index()] = true;
            board.player = player.opponent();
            attach.turns = attach.turns.saturating_add(1);
            return Ok(WaveSimulationState::Resolve {
                descent,
                board,
                attach,
            });
        }

        let mut candidates = Vec::with_capacity(result.candidates.len());
        let mut eval_actions = Vec::with_capacity(result.candidates.len() + 1);
        for candidate in result.candidates {
            let candidate_ref = PortableCandidateRef::new(context, candidate.candidate_hash);
            eval_actions.push(EvalAction::candidate(
                candidate_ref,
                candidate.kind,
                candidate.tags,
                candidate.static_prior,
            ));
            candidates.push(CandidateEntry {
                handle: candidate.candidate,
                hash: candidate.candidate_hash,
                summary: SearchCandidateSummary {
                    kind: candidate.kind,
                    tags: candidate.tags,
                    static_prior: candidate.static_prior,
                },
            });
        }
        eval_actions.push(EvalAction::stop(context));
        Ok(WaveSimulationState::EvalReady {
            descent,
            board,
            attach,
            expansion: Expansion {
                context,
                candidates,
                eval_actions,
            },
        })
    }

    fn complete_wave_eval(
        &mut self,
        mut descent: Descent,
        board: Board<G>,
        attach: Attach,
        expansion: Expansion<C>,
        request: EvalRequest,
        output: EvalOutput,
    ) -> EngineResult<WaveSimulationState<G, C>> {
        output
            .validate_for(&request)
            .map_err(eval_error_to_engine_error)?;
        let candidate_count = expansion.candidates.len();
        let action_count = if self.config.mask_stop {
            candidate_count
        } else {
            candidate_count + 1
        };
        let logits = output.policy_logits[..action_count].to_vec();
        let priors = softmax(&logits);
        let candidate_metadata = expansion
            .candidates
            .iter()
            .map(|candidate| RootCandidate {
                hash: candidate.hash,
                summary: candidate.summary,
            })
            .collect::<Vec<_>>();
        let node_index = self.nodes.len();
        self.nodes.push(Node {
            board,
            context: expansion.context,
            candidates: expansion
                .candidates
                .into_iter()
                .map(|candidate| candidate.handle)
                .collect(),
            candidate_metadata,
            logits,
            priors,
            value: output.value,
            model_version: output.model_version,
            masked: vec![false; action_count],
            edges: (0..action_count).map(|_| ActionEdge::default()).collect(),
            pass: None,
        });
        self.eval_count += 1;
        let value = self.attach_target(&mut descent, attach, BranchTarget::Node(node_index))?;
        Ok(WaveSimulationState::Outcome { descent, value })
    }

    fn attach_target(
        &mut self,
        descent: &mut Descent,
        attach: Attach,
        target: BranchTarget,
    ) -> EngineResult<f32> {
        let branch = Branch {
            target,
            flip: attach.turns % 2 == 1,
        };
        match attach.slot {
            AttachSlot::Action { node, action } => {
                let edge = &mut self.nodes[node].edges[action];
                if edge.branch.is_some() {
                    return Err(internal("symmetric action branch already installed"));
                }
                edge.branch = Some(branch);
                descent.path.push(PathStep::Decision {
                    node,
                    action,
                    flip: branch.flip,
                });
            }
            AttachSlot::Pass { node } => {
                if self.nodes[node].pass.is_some() {
                    return Err(internal("symmetric pass branch already installed"));
                }
                self.nodes[node].pass = Some(branch);
                descent.path.push(PathStep::Transform { flip: branch.flip });
            }
        }
        let value = match target {
            BranchTarget::Node(node) => self.nodes[node].value,
            BranchTarget::Terminal(value) => value,
        };
        Ok(value)
    }
}
