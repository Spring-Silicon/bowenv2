use gz_engine::run_engine_contract;
use gz_engine_whittle::WhittleContractFixture;

#[test]
fn whittle_engine_satisfies_graph_engine_contract() {
    run_engine_contract(&WhittleContractFixture).unwrap();
}
