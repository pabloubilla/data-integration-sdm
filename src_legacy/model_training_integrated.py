


def _fourier_features(
    X: np.ndarray,
    n_freqs: int = 6,
    include_input: bool = True,
) -> np.ndarray:
    """
    Simple Fourier features: [x, sin(2^k x), cos(2^k x)] for k=0..n_freqs-1, per dimension.
    X is expected to be scaled already (roughly zero mean, unit var).
    """
    feats = []
    if include_input:
        feats.append(X)

    # Frequencies: 1,2,4,... (works well as a simple default)
    for k in range(n_freqs):
        w = 2.0 ** k
        feats.append(np.sin(w * X))
        feats.append(np.cos(w * X))

    return np.concatenate(feats, axis=1).astype(np.float32)



def run_experiment_popa_bias(
    name: str,
    X_po_df: pd.DataFrame, Y_po_df: pd.DataFrame,
    X_pa_tr_df: pd.DataFrame, Y_pa_tr_df: pd.DataFrame,
    X_pa_te_df: pd.DataFrame, Y_pa_te_df: pd.DataFrame,
    covs: List[str], covs_bias: List[str], species: List[str],
    output_dir: str, region: str, group: str,
    epochs: int = 300, lr: float = 1e-4,
    batch_size: int = 256, w_po: float = .5, w_pa: float = .5,
    # NEW: Fourier feature controls (safe defaults)
    bias_fourier_freqs: int = 6,
    bias_fourier_include_raw: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, f"scaler_{name}_{region}{group}.pkl")

    # -------------------------
    # 1) Scale species covs (fit on PO + PA train)
    # -------------------------
    scaler = StandardScaler().fit(pd.concat([X_po_df[covs], X_pa_tr_df[covs]], axis=0))

    X_po_s = X_po_df.copy()
    X_pa_tr_s = X_pa_tr_df.copy()
    X_pa_te_s = X_pa_te_df.copy()
    X_po_s[covs] = scaler.transform(X_po_s[covs])
    X_pa_tr_s[covs] = scaler.transform(X_pa_tr_s[covs])
    X_pa_te_s[covs] = scaler.transform(X_pa_te_s[covs])

    # -------------------------
    # 2) Scale bias covs, then expand with Fourier features
    # -------------------------
    scaler_bias = StandardScaler().fit(pd.concat([X_po_df[covs_bias], X_pa_tr_df[covs_bias]], axis=0))

    # scaled raw bias covs
    X_po_bias_scaled = scaler_bias.transform(X_po_df[covs_bias])
    X_pa_tr_bias_scaled = scaler_bias.transform(X_pa_tr_df[covs_bias])
    X_pa_te_bias_scaled = scaler_bias.transform(X_pa_te_df[covs_bias])

    # Fourier-expanded bias covs (this is what the bias net will see)
    X_po_bias_ff = _fourier_features(
        X_po_bias_scaled, n_freqs=bias_fourier_freqs, include_input=bias_fourier_include_raw
    )
    X_pa_tr_bias_ff = _fourier_features(
        X_pa_tr_bias_scaled, n_freqs=bias_fourier_freqs, include_input=bias_fourier_include_raw
    )
    X_pa_te_bias_ff = _fourier_features(
        X_pa_te_bias_scaled, n_freqs=bias_fourier_freqs, include_input=bias_fourier_include_raw
    )

    # keep original dataframes around for plotting coords etc.
    # (we only need numpy arrays for datasets)

    # -------------------------
    # 3) Model (only change: n_bias_covariates)
    # -------------------------


    hidden_species_arquitecture = (HIDDEN_SIZE,) * HIDDEN_LAYERS

    model = SDMWithBias(
        n_species_covariates=len(covs),
        n_species=len(species),
        n_bias_covariates=X_po_bias_ff.shape[1],  # UPDATED
        hidden_species=hidden_species_arquitecture,
        hidden_bias=(2000, 2000)
    )

    # -------------------------
    # 4) Loaders (same idea as before)
    # -------------------------
    # proportion_po = len(X_po_s) / (len(X_po_s) + len(X_pa_tr_s))
    # proportion_pa = len(X_pa_tr_s) / (len(X_po_s) + len(X_pa_tr_s))
    # batch_size_po = max(1, int(batch_size * proportion_po))
    # batch_size_pa = max(1, int(batch_size * proportion_pa))
    
    batch_size_pa = min(batch_size, int(len(X_pa_tr_s) * MAX_BATCH_PERCENTAGE))
    proportion_batch = batch_size / len(X_pa_tr_s)
    batch_size_po = int(len(X_po_s)*proportion_batch)

    print('Batch for PO: ', batch_size_po)
    print('Batch for PA: ', batch_size_pa)

    # # set both to the min (batch_size_po, batch_size_pa)
    # min_out_batch = min(batch_size_po, batch_size_pa)
    # batch_size_po = min_out_batch
    # batch_size_pa = min_out_batch

    po_ds = XZYDataset(
        X_po_s[covs].values.astype(np.float32),
        X_po_bias_ff,  # UPDATED: Fourier features
        Y_po_df[species].values.astype(np.float32)
    )
    pa_ds = XYDataset(
        X_pa_tr_s[covs].values.astype(np.float32),
        Y_pa_tr_df[species].values.astype(np.float32)
    )

    po_loader = DataLoader(po_ds, batch_size=batch_size_po, shuffle=True, drop_last=False)
    pa_loader = DataLoader(pa_ds, batch_size=batch_size_pa, shuffle=True, drop_last=False)

    # -------------------------
    # 5) Train (unchanged)
    # -------------------------
    def train_popa_model_bias(
        model: torch.nn.Module,
        po_loader: DataLoader,
        pa_loader: DataLoader,
        po_ds: Dataset,
        pa_ds: Dataset,
        criterion_species: torch.nn.Module,
        epochs: int = 300,
        lr: float = 1e-4,
        dev: Optional[torch.device] = None,
        w_po: float = .5,
        w_pa: float = .5,
    ):
        dev = dev or device()
        model.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)

        model.train()
        loss_fn = PoissonCountAndPresenceLoss(w_pa = 3)
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for (xb_po, zb_po, yb_po), (xb_pa, yb_pa, _) in zip(po_loader, pa_loader):
                xb_po = xb_po.to(dev)
                zb_po = zb_po.to(dev)
                yb_po = yb_po.to(dev)
                xb_pa = xb_pa.to(dev)
                yb_pa = yb_pa.to(dev)

                optimizer.zero_grad()
                score_po, bias_pred_po = model(xb_po, zb_po)
                score_pa, _ = model(xb_pa)
                loss = loss_fn(score_po, score_pa, bias_pred_po, yb_po, yb_pa)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * (xb_po.size(0) + xb_pa.size(0))

    loss_fn = None
    train_popa_model_bias(
        model=model,
        po_loader=po_loader,
        pa_loader=pa_loader,
        po_ds=po_ds,
        pa_ds=pa_ds,
        criterion_species=loss_fn,
        epochs=epochs,
        lr=lr,
        dev=device(),
        w_po=w_po,
        w_pa=w_pa
    )

    # -------------------------
    # 6) Evaluate (unchanged)
    # -------------------------
    X_te_np = X_pa_te_s[covs].values.astype(np.float32)
    scores = predict(model, X_te_np, dev=device(), bias_model=True)
    aucs = per_species_auc(Y_pa_te_df[species], scores, species)
    avg_auc = np.nanmean(list(aucs.values()))

    aucs_site = per_site_auc(Y_pa_te_df[species], scores)
    avg_auc_site = np.nanmean(list(aucs_site.values()))

    # -------------------------
    # 7) Predict bias on PO points (UPDATED: use Fourier features)
    # -------------------------
    X_po_np = X_po_s[covs].values.astype(np.float32)
    X_po_tensor = torch.tensor(X_po_np, dtype=torch.float32, device=device())

    X_po_bias_tensor = torch.tensor(X_po_bias_ff, dtype=torch.float32, device=device())
    scores_t, bias_scores_t = model(X_po_tensor, X_po_bias_tensor)
    bias_scores = bias_scores_t.detach().cpu().numpy()

    # plot the bias (same as before)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(X_po_s['x'], X_po_s['y'], c=bias_scores, cmap='viridis', s=2, alpha=0.7)
    plt.colorbar(label='Bias Score')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Predicted Bias Scores at PO Locations')
    # add parameters to filename (bias arquitecture)
    # bias_shape = '_'.join([str(h) for h in model.hidden_bias])
    # file_name = f"bias_map_{name}_{region}{group}_biasnet{bias_shape}.png"
    file_name = f"bias_map_{name}_{region}{group}.png"
    plt.savefig(file_name)

    model_path = None

    return avg_auc, aucs, avg_auc_site, model_path, scaler_path