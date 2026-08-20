from cell_load.data_modules import PerturbationDataModule

dm = PerturbationDataModule(
    # Required parameters
    toml_config_path="/research/projects/rna_lab/perturb_benchmark/state/xatlas_config.toml",
    embed_key="X_hvg",
    num_workers=24,
    batch_col="gem_group",
    pert_col="gene",
    cell_type_key="cell_type",
    control_pert="non-targeting",
    
    # Optional parameters
    barcode=True,  # Enable cell barcode output
    output_space="gene",
    basal_mapping_strategy="random",
    n_basal_samples=1,
    should_yield_control_cells=True,
    batch_size=128,
)
dm.setup()