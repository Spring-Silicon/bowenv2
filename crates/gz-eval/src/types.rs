use crate::{EvalError, EvalResult};
use gz_engine::{
    CandidateKindId, CandidateTags, ModelVersion, PortableCandidateRef, PortableSearchActionRef,
    ReplayGraphContext,
};

#[derive(Clone, Debug, PartialEq)]
pub struct EvalRequest {
    pub context: ReplayGraphContext,
    pub actions: Vec<EvalAction>,
}

impl EvalRequest {
    pub fn new(context: ReplayGraphContext, actions: Vec<EvalAction>) -> EvalResult<Self> {
        Self { context, actions }.validate()
    }

    #[must_use]
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    pub fn validate(self) -> EvalResult<Self> {
        self.validate_ref()?;
        Ok(self)
    }

    pub fn validate_ref(&self) -> EvalResult<()> {
        if self.actions.is_empty() {
            return Err(EvalError::EmptyActions);
        }

        let mut stop_index = None;

        for (index, action) in self.actions.iter().enumerate() {
            let actual = action.action_ref.context();
            if actual != self.context {
                return Err(EvalError::ActionContextMismatch {
                    expected: Box::new(self.context),
                    actual: Box::new(actual),
                });
            }

            match (action.metadata, action.action_ref) {
                (
                    EvalActionMetadata::Candidate { static_prior, .. },
                    PortableSearchActionRef::Candidate(_),
                ) => {
                    if !static_prior.is_finite() {
                        return Err(EvalError::NonFiniteStaticPrior {
                            action_index: index,
                            static_prior,
                        });
                    }
                }
                (EvalActionMetadata::Stop, PortableSearchActionRef::Stop { .. }) => {
                    if let Some(first) = stop_index {
                        return Err(EvalError::DuplicateStop {
                            first,
                            second: index,
                        });
                    }
                    stop_index = Some(index);
                }
                _ => {
                    return Err(EvalError::ActionKindMismatch {
                        action_index: index,
                    });
                }
            }
        }

        let Some(stop_index) = stop_index else {
            return Err(EvalError::MissingStop);
        };
        let last = self.actions.len() - 1;

        if stop_index != last {
            return Err(EvalError::StopNotLast {
                index: stop_index,
                last,
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvalAction {
    pub action_ref: PortableSearchActionRef,
    pub metadata: EvalActionMetadata,
}

impl EvalAction {
    #[must_use]
    pub const fn candidate(
        candidate: PortableCandidateRef,
        kind: CandidateKindId,
        tags: CandidateTags,
        static_prior: f32,
    ) -> Self {
        Self {
            action_ref: PortableSearchActionRef::candidate(candidate),
            metadata: EvalActionMetadata::Candidate {
                kind,
                tags,
                static_prior,
            },
        }
    }

    #[must_use]
    pub const fn stop(context: ReplayGraphContext) -> Self {
        Self {
            action_ref: PortableSearchActionRef::stop(context),
            metadata: EvalActionMetadata::Stop,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvalActionMetadata {
    Candidate {
        kind: CandidateKindId,
        tags: CandidateTags,
        static_prior: f32,
    },
    Stop,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvalOutput {
    pub model_version: ModelVersion,
    pub policy_logits: Vec<f32>,
    pub value: f32,
}

impl EvalOutput {
    pub fn validate_for(&self, request: &EvalRequest) -> EvalResult<()> {
        if self.policy_logits.len() != request.actions.len() {
            return Err(EvalError::PolicyLenMismatch {
                expected: request.actions.len(),
                actual: self.policy_logits.len(),
            });
        }

        for (index, value) in self.policy_logits.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(EvalError::NonFinitePolicyLogit { index, value });
            }
        }

        if !self.value.is_finite() {
            return Err(EvalError::NonFiniteValue { value: self.value });
        }

        Ok(())
    }
}

pub trait Evaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()>;

    fn evaluate_one(&mut self, request: &EvalRequest) -> EvalResult<EvalOutput> {
        let mut out = Vec::with_capacity(1);
        self.evaluate_batch(std::slice::from_ref(request), &mut out)?;

        if out.len() != 1 {
            return Err(EvalError::OutputCountMismatch {
                expected: 1,
                actual: out.len(),
            });
        }

        Ok(out.pop().expect("length checked"))
    }
}

pub fn validate_outputs(requests: &[EvalRequest], outputs: &[EvalOutput]) -> EvalResult<()> {
    if outputs.len() != requests.len() {
        return Err(EvalError::OutputCountMismatch {
            expected: requests.len(),
            actual: outputs.len(),
        });
    }

    for (request, output) in requests.iter().zip(outputs) {
        output.validate_for(request)?;
    }

    Ok(())
}
