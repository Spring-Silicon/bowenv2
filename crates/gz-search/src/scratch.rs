pub struct GreedyScratch<C> {
    pub candidates: Vec<C>,
}

impl<C> Default for GreedyScratch<C> {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }
}
