function [model, R2, whiteness, Y_hat, runtime] = nonlinear_HLGSDE(Y, TR, doPreProc, options, k, use_parallel, test_range)
%NONLINEAR_HLGSDE Hodge-Langevin extension of the MINDy system-ID model.
%
% HL-GSDE v2 is a Hodge-MINDy model. It keeps MINDy's smooth transfer
% function and ROI-wise decay, but replaces a free connectivity matrix with
% four GC/Hodge topology channels and a separately saved residual.
%
% dx_i = sum_q gamma(i,q) [F_q psi_alpha(x)]_i - D_i x_i + epsilon_i,
% where F_q is W_sym, F_grad, F_curl, or F_harm.

if nargin < 2 || isempty(TR); TR = 0.72; end
if nargin < 3 || isempty(doPreProc); doPreProc = 'n'; end
if nargin < 4 || isempty(options); options = struct(); end
if nargin < 5 || isempty(k); k = 1; end
if nargin < 6 || isempty(use_parallel); use_parallel = 0; end %#ok<NASGU>
if nargin < 7 || isempty(test_range); test_range = [0.8 1]; end
if ~isequal(doPreProc, 'n')
    warning('HL-GSDE fits the supplied BOLD data directly; HRF preprocessing is not applied.');
end
this_dir = fileparts(mfilename('fullpath'));
utilities_dir = fullfile(fileparts(this_dir), 'external', 'rest-system-id', 'Utilities');
if exist('tt_decomp', 'file') ~= 2 && exist(utilities_dir, 'dir') == 7
    addpath(utilities_dir);
end

opts = hlgsde_options(options);
if opts.topology_only
    if ~strcmpi(opts.topology_mode, 'hodge')
        error('topology_only is currently supported only for topology_mode=hodge.');
    end
    if iscell(Y)
        Y_train_cell = Y;
    else
        Y_train_cell = {Y};
    end
    runtime_train_start = tic;
    [X, dX] = hlgsde_training_pairs(Y_train_cell, TR);
    model = hlgsde_fit_topology_only(X, dX, opts, TR);
    runtime.train = toc(runtime_train_start);
    runtime.test = 0;
    runtime.total = runtime.train;
    R2 = NaN;
    whiteness = struct('p', NaN, 'stat', NaN, 'sig_thr', NaN);
    Y_hat = [];
    return
end
[Y_train_cell, Y_test_cell, break_ind, test_ind, N_test_vec, n, N] = tt_decomp(Y, test_range);

runtime_train_start = tic;
model = hlgsde_fit(Y_train_cell, opts, TR);
model.TR = TR;
model.DeltaT = TR;
runtime.train = toc(runtime_train_start);

runtime_test_start = tic;
[R2, whiteness, Y_hat, Y_test_hat_full] = hlgsde_evaluate( ...
    model, Y, Y_test_cell, break_ind, test_ind, N_test_vec, n, N, k, ...
    opts.compute_whiteness);
core_model = model;
core_model.W_residual(:) = 0;
if isfield(core_model, 'residual_projection')
    core_model.residual_projection(:) = 0;
end
if ~any(model.W_residual(:))
    core_R2 = R2;
    core_whiteness = whiteness;
    core_Y_hat = Y_hat;
else
    [core_R2, core_whiteness, core_Y_hat] = hlgsde_evaluate( ...
        core_model, Y, Y_test_cell, break_ind, test_ind, N_test_vec, n, N, k, ...
        opts.compute_whiteness);
end
model.core_validation = struct('R2', core_R2, 'whiteness', core_whiteness, ...
    'Y_hat', core_Y_hat);
if strcmpi(opts.topology_mode, 'hodge')
    model.mechanistic_fold = hlgsde_mechanistic_fold(model, Y_train_cell);
else
    model.mechanistic_fold = hlgsde_nonhodge_mechanistic_fold(model);
end
if opts.compute_ablation
    model.mechanistic_fold.ablation = hlgsde_channel_ablation( ...
        core_model, Y, Y_test_cell, break_ind, test_ind, N_test_vec, n, N, k, ...
        core_R2, core_whiteness);
else
    model.mechanistic_fold.ablation = hlgsde_empty_ablation();
end
runtime.test = toc(runtime_test_start);
runtime.total = runtime.train + runtime.test;
end


function opts = hlgsde_options(options)
opts.alpha_grid = [0.10 0.25 0.50 1.00 2.00];
opts.ridge = 1e-3;
opts.ridge_grid = [1e-4 1e-3 1e-2];
opts.min_decay = 1e-4;
opts.hodge_edge_density = 0.08;
opts.hodge_dir = '';
opts.gc_ridge = 1e-3;
opts.gc_fc_density = 0.08;
opts.gc_mask_override = [];
opts.use_signed_hodge = false;
opts.residual_ridge = 1e-3;
opts.residual_ridge_grid = [0.01 0.1 1 10];
opts.residual_gate_grid = 0; % Core is the default proposed model.
opts.residual_max_drift_ratio = 0.5;
opts.diffusion_rank_grid = 0;
opts.diffusion_shrinkage_grid = [0.25 0.5 0.75 1];
opts.compute_ablation = false;
opts.compute_whiteness = true;
opts.topology_mode = 'hodge';
opts.topology_only = false;
if ~isstruct(options); error('HL-GSDE options must be a struct or empty.'); end
has_ridge_grid = isfield(options, 'ridge_grid');
names = fieldnames(options);
for i_name = 1:numel(names)
    if ~isfield(opts, names{i_name}); error('Unknown HL-GSDE option: %s', names{i_name}); end
    opts.(names{i_name}) = options.(names{i_name});
end
if ~has_ridge_grid && isfield(options, 'ridge')
    opts.ridge_grid = options.ridge;
end
opts.alpha_grid = opts.alpha_grid(:)';
opts.ridge_grid = opts.ridge_grid(:)';
opts.diffusion_rank_grid = opts.diffusion_rank_grid(:)';
if ~(ischar(opts.topology_mode) || ...
        (isstring(opts.topology_mode) && isscalar(opts.topology_mode)))
    error('topology_mode must be hodge, gc_direct, or sym_anti.');
end
opts.topology_mode = lower(char(opts.topology_mode));
if ~any(strcmp(opts.topology_mode, {'hodge','gc_direct','sym_anti'}))
    error('topology_mode must be hodge, gc_direct, or sym_anti.');
end
if isempty(opts.alpha_grid) || any(~isfinite(opts.alpha_grid)) || any(opts.alpha_grid <= 0)
    error('alpha_grid must contain positive finite values.');
end
if isempty(opts.ridge_grid) || any(~isfinite(opts.ridge_grid)) || any(opts.ridge_grid <= 0)
    error('ridge_grid must contain positive finite values.');
end
if isempty(opts.diffusion_rank_grid) || any(~isfinite(opts.diffusion_rank_grid)) || ...
        any(opts.diffusion_rank_grid < 0) || any(opts.diffusion_rank_grid ~= round(opts.diffusion_rank_grid))
    error('diffusion_rank_grid must contain non-negative integer values.');
end
if opts.hodge_edge_density <= 0 || opts.hodge_edge_density > 1
    error('hodge_edge_density must be in (0, 1].');
end
end


function model = hlgsde_fit(Y_train_cell, opts, TR)
[X, dX] = hlgsde_training_pairs(Y_train_cell, TR);
[W_gc, W_signed, FC, FC_mask, gc_meta] = hlgsde_fc_masked_gc( ...
    X, dX, opts.gc_ridge, opts.gc_fc_density, TR, opts.gc_mask_override);
W_hodge = W_gc;
hodge_source = 'nonnegative_gc_score';
if strcmp(opts.topology_mode, 'hodge') && opts.use_signed_hodge
    W_hodge = W_signed;
    hodge_source = 'signed_lagged_coefficient';
end
W_anti_signed = W_signed - W_signed';
switch opts.topology_mode
    case 'hodge'
        % The default uses non-negative GC strength, producing a signed
        % net flow for HHD. The signed-coefficient option remains a control.
        W_sym = min(W_hodge, W_hodge');
        W_anti = W_hodge - W_hodge';
        [W_anti_hodge, sparsity] = hlgsde_sparse_net_flow( ...
            W_anti, opts.hodge_edge_density);
        [F_grad, F_curl, F_harm, hodge_meta] = ...
            hlgsde_hodge_decompose(W_anti_hodge, opts.hodge_dir);
        hodge_meta.input_definition = [hodge_source ' then sparse net flow'];
        component_names = {'sym','grad','curl','harm'};
        raw_components = {W_sym, F_grad, F_curl, F_harm};
    case 'gc_direct'
        % Direct control: use the non-negative GC score and skip HHD.
        W_sym = min(W_gc, W_gc');
        W_anti = W_gc - W_gc';
        W_anti_hodge = zeros(size(W_gc));
        sparsity = hlgsde_skipped_sparsity(size(W_gc, 1), ...
            'skipped for gc_direct control');
        F_grad = zeros(size(W_gc)); F_curl = F_grad; F_harm = F_grad;
        hodge_meta = hlgsde_no_hodge_meta(size(W_gc, 1), opts.hodge_dir, ...
            'not computed: direct nonnegative GC control');
        component_names = {'gc','inactive','inactive','inactive'};
        raw_components = {W_gc, F_grad, F_curl, F_harm};
    case 'sym_anti'
        % No-HHD control: use the same sparse net flow as HHD, without
        % splitting it into gradient, curl, and harmonic components.
        W_sym = min(W_gc, W_gc');
        W_anti = W_gc - W_gc';
        [W_anti_hodge, sparsity] = hlgsde_sparse_net_flow( ...
            W_anti, opts.hodge_edge_density);
        F_grad = zeros(size(W_gc)); F_curl = F_grad; F_harm = F_grad;
        hodge_meta = hlgsde_no_hodge_meta(size(W_gc, 1), opts.hodge_dir, ...
            'not computed: sparse sym/anti control');
        component_names = {'sym','anti','inactive','inactive'};
        raw_components = {W_sym, W_anti_hodge, F_curl, F_harm};
end
gc_meta.hodge_source = hodge_source;
gc_meta.topology_mode = opts.topology_mode;

components = cell(size(raw_components));
scales = zeros(1, numel(raw_components));
for i_component = 1:numel(raw_components)
    scales(i_component) = norm(raw_components{i_component}, 'fro');
    if scales(i_component) > 0
        components{i_component} = raw_components{i_component} ./ scales(i_component);
    else
        components{i_component} = zeros(size(W_gc));
        scales(i_component) = 1;
    end
end

best = struct();
n_time = size(X, 2);
validation_split = max(2, min(n_time - 1, floor(0.8 * n_time)));
has_inner_validation = n_time - validation_split >= 2;
alpha_validation_sse = inf(numel(opts.alpha_grid), numel(opts.ridge_grid));
for i_alpha = 1:numel(opts.alpha_grid)
    alpha = opts.alpha_grid(i_alpha);
    Z = hlgsde_transfer(X, alpha);
    channel_input = cellfun(@(F) F * Z, components, 'UniformOutput', false);
    for i_ridge = 1:numel(opts.ridge_grid)
        fit_opts = opts;
        fit_opts.ridge = opts.ridge_grid(i_ridge);
        if has_inner_validation
            fit_input = cellfun(@(V) V(:, 1:validation_split), ...
                channel_input, 'UniformOutput', false);
            [gain, decay] = hlgsde_fit_channels( ...
                X(:, 1:validation_split), dX(:, 1:validation_split), ...
                fit_input, fit_opts);
            validation_input = cellfun(@(V) V(:, validation_split + 1:end), ...
                channel_input, 'UniformOutput', false);
            prediction = hlgsde_predict_channels( ...
                gain, decay, validation_input, X(:, validation_split + 1:end));
            alpha_validation_sse(i_alpha, i_ridge) = sum( ...
                (dX(:, validation_split + 1:end) - prediction).^2, 'all');
        else
            [~, ~, ~, sse] = hlgsde_fit_channels(X, dX, channel_input, fit_opts);
            alpha_validation_sse(i_alpha, i_ridge) = sse;
        end
    end
end

[~, best_index] = min(alpha_validation_sse(:));
[best_alpha_index, best_ridge_index] = ind2sub(size(alpha_validation_sse), best_index);
best.alpha = opts.alpha_grid(best_alpha_index);
best.ridge = opts.ridge_grid(best_ridge_index);
best_opts = opts;
best_opts.ridge = best.ridge;
Z = hlgsde_transfer(X, best.alpha);
channel_input = cellfun(@(F) F * Z, components, 'UniformOutput', false);
[best.gain, best.decay, best.residual, best_sse] = ...
    hlgsde_fit_channels(X, dX, channel_input, best_opts);
alpha_sse = min(alpha_validation_sse, [], 2)';

effective_W = zeros(size(W_gc));
for i_component = 1:numel(components)
    effective_W = effective_W + best.gain(:, i_component) .* components{i_component};
end

[residual_model, residual_selection] = hlgsde_select_masked_residual( ...
    X, dX, channel_input, Z, best.residual, FC_mask, opts);
total_residual = best.residual - hlgsde_residual_drift(residual_model, Z, channel_input, X);
[noise_core, noise_selection] = hlgsde_structured_noise(best.residual, opts);

if nnz(residual_model.W_residual) == 0
    switch opts.topology_mode
        case 'gc_direct'
            model.name = 'GC-GSDE-Direct';
        case 'sym_anti'
            model.name = 'GC-GSDE-SymAnti-noHHD';
        otherwise
            model.name = 'HL-GSDE-Core';
    end
else
    switch opts.topology_mode
        case 'gc_direct'
            model.name = 'GC-GSDE-Direct+MaskedResidual';
        case 'sym_anti'
            model.name = 'GC-GSDE-SymAnti-noHHD+MaskedResidual';
        otherwise
            model.name = 'HL-GSDE-Core+MaskedResidual';
    end
end
if strcmp(opts.topology_mode, 'hodge')
    drift_label = 'f_{\mathrm{HL}}';
else
    drift_label = 'f_{\mathrm{GC}}';
end
model.eq = ['$$r_t=(x_{t+1}-x_t)/\mathrm{TR}=' drift_label '+e_t,\quad ' ...
    'x_{t+1}=x_t+\mathrm{TR}' drift_label '(x_t)+\sqrt{\mathrm{TR}}' ...
    '(\mathrm{diag}(\sigma_{diff})\xi_t+L_{diff}\zeta_t)$$'];
model.alpha = best.alpha;
model.b = 20/3;
model.D = best.decay;
model.sigma = sqrt(mean(total_residual.^2, 2)); % Legacy rate-residual field.
model.sigma_rate = noise_core.sigma;
model.sigma_core = model.sigma_rate;
model.sigma_diffusion = sqrt(TR) * model.sigma_rate;
model.noise_factor_core = noise_core.factor; % Legacy rate-residual factor.
model.noise_factor_diffusion_core = sqrt(TR) * noise_core.factor;
model.generation_sigma = model.sigma_diffusion;
model.generation_noise_factor = model.noise_factor_diffusion_core;
model.generation_use_core_drift = true;
model.diffusion_rank = noise_selection.rank;
model.diffusion_shrinkage = noise_selection.shrinkage;
model.diffusion_validation_error = noise_selection.validation_error;
model.channel_gain = best.gain;
model.W_effective = effective_W;
model.W_residual = residual_model.W_residual;
model.W_masked_residual = residual_model.W_residual;
model.residual_projection = residual_model.projection;
model.W_total_effective = effective_W + residual_model.W_residual;
% This is an edge-constrained residual, not a low-rank residual.
model.residual_active_edges = nnz(residual_model.W_residual);
model.residual_ridge = residual_selection.ridge;
model.residual_gate = residual_selection.gate;
model.residual_drift_ratio = residual_selection.drift_ratio;
model.residual_mask_edges = nnz(residual_model.mask);
model.residual_validation_sse = residual_selection.validation_sse;
model.residual_selection = residual_selection;
model.W_gc = W_gc;
model.W_signed = W_signed;
model.W_hodge = W_hodge;
model.hodge_source = hodge_source;
model.W_anti_signed = W_anti_signed;
model.FC_train = FC;
model.FC_mask = FC_mask;
model.gc = gc_meta;
model.W_sym = W_sym;
model.W_anti = W_anti;
model.W_anti_hodge = W_anti_hodge;
model.F_grad = F_grad;
model.F_curl = F_curl;
model.F_harm = F_harm;
model.component_scales = scales;
model.gamma_effective = bsxfun(@rdivide, best.gain, scales);
model.topology_mode = opts.topology_mode;
model.component_names = component_names;
model.component_matrices = raw_components;
model.active_component_indices = find(cellfun(@(F) any(F(:) ~= 0), raw_components));
model.active_component_names = component_names(model.active_component_indices);
model.n_active_channels = numel(model.active_component_indices);
model.n_active_drift_parameters = size(W_gc, 1) * ...
    (model.n_active_channels + 1);
model.n_drift_parameters = model.n_active_drift_parameters;
if opts.use_signed_hodge && strcmp(opts.topology_mode, 'hodge')
    model.W_sym_definition = 'min(W_signed, W_signed^T)';
    model.W_anti_definition = 'W_signed - W_signed^T';
else
    model.W_sym_definition = 'min(W_gc, W_gc^T)';
    model.W_anti_definition = 'W_gc - W_gc^T';
end
model.hodge = hodge_meta;
model.sparsity = sparsity;
model.options = opts;
model.compute_whiteness = opts.compute_whiteness;
model.training_sse_core = best_sse;
model.training_sse = sum(total_residual(:).^2);
model.alpha_grid = opts.alpha_grid;
model.alpha_sse = alpha_sse;
model.alpha_sse_relative = alpha_sse ./ max(min(alpha_sse), eps);
model.alpha_validation_sse = alpha_validation_sse;
model.ridge_grid = opts.ridge_grid;
model.selected_ridge = best.ridge;
model.fit_validation_fraction = 1 - validation_split / n_time;
model.residual_units = 'state/time';
model.diffusion_units = 'state/sqrt(time)';
model.noise_scaling_definition = ['sigma_rate is the standard deviation of the ' ...
    'derivative residual; sigma_diffusion=sqrt(TR)*sigma_rate; rollout uses ' ...
    'sqrt(TR)*sigma_diffusion.'];
model.biological_gradient = 'not used: no independently validated ROI gradient supplied';
end


function model = hlgsde_fit_topology_only(X, dX, opts, TR)
% Extract training-only GC/Hodge topology without fitting prediction gains.
[W_gc, W_signed, FC, FC_mask, gc_meta] = hlgsde_fc_masked_gc( ...
    X, dX, opts.gc_ridge, opts.gc_fc_density, TR, opts.gc_mask_override);
W_hodge = W_gc;
hodge_source = 'nonnegative_gc_score';
if opts.use_signed_hodge
    W_hodge = W_signed;
    hodge_source = 'signed_lagged_coefficient';
end
W_sym = min(W_hodge, W_hodge');
W_anti = W_hodge - W_hodge';
[W_anti_hodge, sparsity] = hlgsde_sparse_net_flow( ...
    W_anti, opts.hodge_edge_density);
[F_grad, F_curl, F_harm, hodge_meta] = ...
    hlgsde_hodge_decompose(W_anti_hodge, opts.hodge_dir);
hodge_meta.input_definition = [hodge_source ' then sparse net flow'];
gc_meta.hodge_source = hodge_source;
gc_meta.topology_mode = 'hodge';
model = struct('name', 'HL-GSDE-TopologyOnly', 'TR', TR, 'DeltaT', TR, ...
    'topology_only', true, 'topology_mode', 'hodge', ...
    'hodge_source', hodge_source, 'W_gc', W_gc, 'W_signed', W_signed, ...
    'W_hodge', W_hodge, 'W_anti_signed', W_signed - W_signed', ...
    'FC_train', FC, 'FC_mask', FC_mask, 'gc', gc_meta, ...
    'W_sym', W_sym, 'W_anti', W_anti, 'W_anti_hodge', W_anti_hodge, ...
    'F_grad', F_grad, 'F_curl', F_curl, 'F_harm', F_harm, ...
    'component_matrices', {{W_sym, F_grad, F_curl, F_harm}}, ...
    'component_names', {{'sym','grad','curl','harm'}}, ...
    'hodge', hodge_meta, 'sparsity', sparsity, 'options', opts);
end


function [W, W_signed, FC, mask, meta] = hlgsde_fc_masked_gc( ...
        X, dX, ridge_weight, density, TR, mask_override)
% Conditional lag-1 GC inside a training-only functional-connectivity mask.
% dX is a derivative, so multiply by TR to recover the next sample.
if nargin < 5 || isempty(TR); TR = 1; end
if nargin < 6; mask_override = []; end
Y_next = X + TR * dX;
X = X - mean(X, 2);
Y_next = Y_next - mean(Y_next, 2);
scale = std(X, 0, 2);
scale(~isfinite(scale) | scale < eps) = 1;
X = X ./ scale;
Y_next = Y_next ./ scale;
n = size(X, 1);
n_sample = size(X, 2);
FC = corr(X');
FC(~isfinite(FC)) = 0;
FC(1:n+1:end) = 1;
if isempty(mask_override)
    n_source = max(1, min(n - 1, round(density * (n - 1))));
    mask = false(n);
else
    if ~isequal(size(mask_override), [n n])
        error('gc_mask_override must be an nROI-by-nROI matrix.');
    end
    mask = logical(mask_override);
    mask(1:n+1:end) = false;
    n_source = NaN;
end
W = zeros(n);
W_signed = zeros(n);
for i = 1:n
    if isempty(mask_override)
        strength = abs(FC(i, :));
        strength(i) = -inf;
        [~, order] = sort(strength, 'descend');
        source = order(1:n_source);
        mask(i, source) = true;
    else
        source = find(mask(i, :));
    end
    predictor = [i source];
    Xi = X(predictor, :);
    gram = Xi * Xi';
    ridge = max(1e-8, ridge_weight * trace(gram) / numel(predictor));
    gram = gram + ridge * eye(numel(predictor));
    R = chol(gram);
    beta = R \ (R' \ (Xi * Y_next(i, :)'));
    gram_inv_diag = diag(R \ (R' \ eye(numel(predictor))));
    rss = max(sum((Y_next(i, :) - beta' * Xi).^2), eps);
    delta_rss = (beta(2:end).^2) ./ max(gram_inv_diag(2:end), eps);
    W(i, source) = log1p(delta_rss' ./ rss);
    W_signed(i, source) = beta(2:end)';
end
W(~isfinite(W)) = 0;
W = max(W, 0);
W(1:n+1:end) = 0;
W_signed(~isfinite(W_signed)) = 0;
W_signed(1:n+1:end) = 0;
meta = struct('name', 'fc_masked_conditional_gc', 'n_samples', n_sample, ...
    'n_roi', n, 'sources_per_target', mean(sum(mask, 2)), ...
    'sources_per_target_vector', sum(mask, 2), ...
    'requested_density', density, 'actual_density', nnz(mask) / (n * (n - 1)), ...
    'ridge_weight', ridge_weight, 'tr', TR, ...
    'target_definition', 'X + TR*dX = Y(t+1)', ...
    'standardized_training_only', true, ...
    'score_definition', 'conditional ridge RSS score', ...
    'signed_definition', 'standardized lagged ridge coefficient', ...
    'mask_override_used', ~isempty(mask_override));
end


function [X, dX] = hlgsde_training_pairs(Y_cell, TR)
if nargin < 2 || isempty(TR); TR = 1; end
X = [];
dX = [];
for i_scan = 1:numel(Y_cell)
    Y = Y_cell{i_scan};
    if size(Y, 2) < 3; continue; end
    X = [X, Y(:, 1:end-1)]; %#ok<AGROW>
    dX = [dX, diff(Y, 1, 2) ./ TR]; %#ok<AGROW>
end
if isempty(X); error('HL-GSDE requires a training segment with at least three time points.'); end
end


function W_gc = hlgsde_conditional_gc(X, dX, ridge_weight, TR)
% Full conditional lag-1 MVAR GC. Row i, column j denotes j -> i.
if nargin < 4 || isempty(TR); TR = 1; end
n = size(X, 1);
Y_next = X + TR * dX;
gram = X * X';
ridge = max(1e-8, ridge_weight * trace(gram) / max(n, 1));
gram = gram + ridge * eye(n);
R = chol(gram);
cross = X * Y_next';
beta = R \ (R' \ cross);
gram_inv = R \ (R' \ eye(n));
residual = Y_next - beta' * X;
rss = max(sum(residual.^2, 2), eps);
inv_diag = max(diag(gram_inv), eps);
delta_rss = (beta.^2) ./ inv_diag;
W_gc = log1p((delta_rss ./ rss')');
W_gc(~isfinite(W_gc)) = 0;
W_gc = max(W_gc, 0);
W_gc(1:n+1:end) = 0;
end


function [W_sparse, meta] = hlgsde_sparse_net_flow(W_anti, edge_density)
n = size(W_anti, 1);
upper_ind = find(triu(true(n), 1));
strength = abs(W_anti(upper_ind));
edge_count = max(1, round(edge_density * numel(strength)));
[~, order] = sort(strength, 'descend');
edge_count = min(edge_count, numel(strength));
selected = order(1:edge_count);
mask = false(n);
mask(upper_ind(selected)) = true;
mask = mask | mask';
W_sparse = W_anti .* mask;
meta.requested_edge_density = edge_density;
meta.actual_edge_density = nnz(triu(W_sparse ~= 0, 1)) / max(numel(upper_ind), 1);
meta.n_possible_edges = numel(upper_ind);
meta.n_retained_edges = nnz(triu(W_sparse ~= 0, 1));
meta.absolute_weight_threshold = min(strength(selected));
meta.skipped = false;
end


function meta = hlgsde_skipped_sparsity(n, reason)
meta = struct('requested_edge_density', NaN, 'actual_edge_density', NaN, ...
    'n_possible_edges', n * (n - 1) / 2, 'n_retained_edges', 0, ...
    'absolute_weight_threshold', NaN, 'skipped', true, 'reason', reason);
end


function meta = hlgsde_no_hodge_meta(n, configured_dir, definition)
if nargin < 2 || isempty(configured_dir)
    configured_dir = '';
end
meta = struct('hodge_dir', configured_dir, 'edge_density', NaN, ...
    'n_edges', 0, 'n_triangles', 0, 'residual_norm', NaN, ...
    'phi', [], 'psi', [], 'computed', false, ...
    'input_definition', definition, 'n_roi', n);
end


function [F_grad, F_curl, F_harm, meta] = hlgsde_hodge_decompose(W_anti, configured_dir)
if ~isnumeric(W_anti) || ~isreal(W_anti) || ~ismatrix(W_anti) || ...
        size(W_anti, 1) ~= size(W_anti, 2) || any(~isfinite(W_anti(:)))
    error('HLGSDE:ExactHodgeInput', 'Hodge flow must be a finite real square matrix.');
end
if nargin < 2 || isempty(configured_dir); configured_dir = getenv('HLGSDE_HODGE_DIR'); end
if isempty(configured_dir)
    this_dir = fileparts(mfilename('fullpath'));
    configured_dir = fullfile(fileparts(this_dir), 'external', 'hodge-main');
end
if exist(fullfile(configured_dir, 'Hodge_decompose.m'), 'file') ~= 2
    error(['Hodge toolbox not found at %s. Set HLGSDE_HODGE_DIR or ' ...
        'options.hodge_dir.'], configured_dir);
end
% hodge-main ships a legacy cell2mat.m that shadows MATLAB's function.
% Remove every hodge-main entry first, then expose only its root while HHD
% runs. The cleanup also executes if exact projection throws an error.
hlgsde_clear_hodge_path(configured_dir);
addpath(configured_dir, '-begin');
cleanup_path = onCleanup(@() hlgsde_clear_hodge_path(configured_dir));

Y = 0.5 * (W_anti - W_anti');
Y(1:size(Y, 1)+1:end) = 0;
n = size(Y, 1);
if ~any(Y(:))
    % Legacy skeleton/vector helpers cannot construct an empty edge list.
    skeleton = {(1:n)', zeros(0, 2), zeros(0, 3)};
    boundary = {sparse(n, 0), sparse(0, 0)};
    Yvec = zeros(0, 1);
else
    skeleton = Hodge_2Skeleton(Y);
    boundary = PH_boundary(skeleton);
    Yvec = Hodge_vec(Y);
end
[Yg, Yc, Yh, phi, psi, meta] = hlgsde_exact_hodge(Yvec, boundary{1}, boundary{2});
elist = skeleton{2};
F_grad = zeros(n); F_curl = zeros(n); F_harm = zeros(n);
for i_edge = 1:size(elist, 1)
    i = elist(i_edge, 1); j = elist(i_edge, 2);
    F_grad(i, j) = Yg(i_edge); F_grad(j, i) = -Yg(i_edge);
    F_curl(i, j) = Yc(i_edge); F_curl(j, i) = -Yc(i_edge);
    F_harm(i, j) = Yh(i_edge); F_harm(j, i) = -Yh(i_edge);
end
meta.hodge_dir = configured_dir;
meta.edge_density = nnz(triu(Y ~= 0, 1)) / max(n * (n - 1) / 2, 1);
meta.n_edges = size(elist, 1);
meta.n_triangles = size(skeleton{3}, 1);
meta.residual_norm = norm(Y - F_grad - F_curl - F_harm, 'fro') / max(norm(Y, 'fro'), eps);
meta.phi = phi;
meta.psi = psi;
end


function [gain, decay, residual, sse] = hlgsde_fit_channels(X, dX, channel_input, opts)
n = size(X, 1);
gain = zeros(n, 4); decay = zeros(n, 1); prediction = zeros(size(dX));
for i_node = 1:n
    U = vertcat(channel_input{1}(i_node, :), channel_input{2}(i_node, :), ...
        channel_input{3}(i_node, :), channel_input{4}(i_node, :));
    feature = [U; -X(i_node, :)];
    gram = feature * feature';
    ridge_scale = max(trace(gram) / size(gram, 1), 1);
    ridge = diag([repmat(opts.ridge * ridge_scale, 1, 4), 1e-10]);
    beta = (gram + ridge) \ (feature * dX(i_node, :)');
    decay(i_node) = max(opts.min_decay, beta(5));
    gain(i_node, :) = ((U * U' + opts.ridge * ridge_scale * eye(4)) \ ...
        (U * (dX(i_node, :) + decay(i_node) * X(i_node, :))'))';
    prediction(i_node, :) = gain(i_node, :) * U - decay(i_node) * X(i_node, :);
end
residual = dX - prediction;
sse = sum(residual(:).^2);
end


function prediction = hlgsde_predict_channels(gain, decay, channel_input, X)
prediction = zeros(size(X));
for i_component = 1:4
    prediction = prediction + bsxfun(@times, gain(:, i_component), ...
        channel_input{i_component});
end
prediction = prediction - bsxfun(@times, decay, X);
end


function [residual_model, selection] = hlgsde_select_masked_residual( ...
        X, dX, channel_input, Z, core_residual, FC_mask, opts)
% Fit a sparse residual only on FC edges and select its strength internally.
n = size(X, 1); n_time = size(X, 2);
split = max(1, min(n_time - 1, floor(0.8 * n_time)));
% Core is the default model. Do not build residual projections when every
% candidate gate is zero; this used to spend most of the extra fit time on a
% model that was immediately discarded.
if isscalar(opts.residual_gate_grid) && opts.residual_gate_grid == 0
    residual_model = struct('W_residual', zeros(n), ...
        'projection', zeros(n, 5), 'mask', FC_mask);
    selection = struct('ridge', NaN, 'gate', 0, ...
        'validation_sse', NaN, 'core_validation_sse', NaN, ...
        'drift_ratio', 0, 'accepted', false, ...
        'validation_fraction', 1 - split / n_time);
    return
end
V_fit = Z(:, 1:split); V_val = Z(:, split + 1:end);
projection_grid = cell(1, numel(opts.residual_ridge_grid));
source_grid = cell(1, numel(opts.residual_ridge_grid));
for ir = 1:numel(opts.residual_ridge_grid)
    ridge = opts.residual_ridge_grid(ir);
    projection = zeros(n, 5); source_weights = zeros(n);
    for i = 1:n
        source = find(FC_mask(i, :));
        if isempty(source); continue; end
        core_features = [channel_input{1}(i, :); channel_input{2}(i, :); ...
            channel_input{3}(i, :); channel_input{4}(i, :); -X(i, :)];
        F_fit = core_features(:, 1:split);
        G = F_fit * F_fit' + ridge * eye(5);
        P = (V_fit(source, :) * F_fit') / G;
        Vp_fit = V_fit(source, :) - P * F_fit;
        gram = Vp_fit * Vp_fit' + ridge * eye(numel(source));
        beta = (core_residual(i, 1:split) * Vp_fit') / gram;
        projection(i, :) = beta * P;
        source_weights(i, source) = beta;
    end
    projection_grid{ir} = projection;
    source_grid{ir} = source_weights;
end
core_val = core_residual(:, split + 1:end);
core_sse = sum(core_val(:).^2);
validation_sse = inf(numel(opts.residual_ridge_grid), numel(opts.residual_gate_grid));
drift_ratio = inf(size(validation_sse));
for ir = 1:numel(opts.residual_ridge_grid)
    for ig = 1:numel(opts.residual_gate_grid)
        gate = opts.residual_gate_grid(ig);
        candidate = source_grid{ir} * gate;
        predicted = candidate * V_val;
        for i = 1:n
            core_features = [channel_input{1}(i, split + 1:end); ...
                channel_input{2}(i, split + 1:end); ...
                channel_input{3}(i, split + 1:end); ...
                channel_input{4}(i, split + 1:end); -X(i, split + 1:end)];
            predicted(i, :) = predicted(i, :) - ...
                gate * projection_grid{ir}(i, :) * core_features;
        end
        ratio = sqrt(mean(predicted(:).^2)) / max(sqrt(mean(core_val(:).^2)), eps);
        drift_ratio(ir, ig) = ratio;
        if ratio <= opts.residual_max_drift_ratio
            validation_sse(ir, ig) = sum((core_val - predicted).^2, 'all');
        end
    end
end
[best_sse, best_index] = min(validation_sse(:));
if ~isfinite(best_sse) || best_sse >= core_sse
    gate = 0; ir = 1; ig = 1;
else
    [ir, ig] = ind2sub(size(validation_sse), best_index);
    gate = opts.residual_gate_grid(ig);
end
residual_model = struct('W_residual', source_grid{ir} * gate, ...
    'projection', projection_grid{ir} * gate, 'mask', FC_mask);
selection = struct('ridge', opts.residual_ridge_grid(ir), 'gate', gate, ...
    'validation_sse', validation_sse, 'core_validation_sse', core_sse, ...
    'drift_ratio', drift_ratio(ir, ig), 'accepted', gate > 0, ...
    'validation_fraction', 1 - split / n_time);
end


function drift = hlgsde_residual_drift(residual_model, Z, channel_input, X)
drift = residual_model.W_residual * Z;
for i = 1:size(X, 1)
    features = [channel_input{1}(i, :); channel_input{2}(i, :); ...
        channel_input{3}(i, :); channel_input{4}(i, :); -X(i, :)];
    drift(i, :) = drift(i, :) - residual_model.projection(i, :) * features;
end
end


function drift = hlgsde_drift(model, X)
Z = hlgsde_transfer(X, model.alpha);
if isfield(model, 'component_matrices')
    components = model.component_matrices;
else
    components = {model.W_sym, model.F_grad, model.F_curl, model.F_harm};
end
drift = zeros(size(X));
for i_component = 1:numel(components)
    F = components{i_component} ./ model.component_scales(i_component);
    drift = drift + model.channel_gain(:, i_component) .* (F * Z);
end
if isfield(model, 'W_residual')
    if isfield(model, 'residual_projection')
        residual_drift = model.W_residual * Z;
        for i_node = 1:size(X, 1)
            core_features = zeros(5, size(X, 2));
            for i_component = 1:4
                core_features(i_component, :) = ...
                    (components{i_component}(i_node, :) ./ ...
                    model.component_scales(i_component)) * Z;
            end
            core_features(5, :) = -X(i_node, :);
            residual_drift(i_node, :) = residual_drift(i_node, :) - ...
                model.residual_projection(i_node, :) * core_features;
        end
        drift = drift + residual_drift;
    else
        drift = drift + model.W_residual * Z;
    end
end
drift = drift - model.D .* X;
end


function [R2, whiteness, Y_hat, Y_test_hat_full] = hlgsde_evaluate( ...
        model, Y, Y_test_cell, break_ind, test_ind, N_test_vec, n, N, k, compute_whiteness)
if nargin < 10 || isempty(compute_whiteness); compute_whiteness = true; end
Y_test_hat_cell = arrayfun(@(N_test) nan(n, N_test), N_test_vec, 'UniformOutput', false);
for i_test = 1:numel(Y_test_cell)
    Y_test = Y_test_cell{i_test};
    if size(Y_test, 2) <= k; continue; end
    state = Y_test(:, 1:end-k);
    for i_step = 1:k
        state = state + model.DeltaT * hlgsde_drift(model, state);
    end
    Y_test_hat_cell{i_test} = [nan(n, k), state];
end
Y_test_full = horzcat(Y_test_cell{:});
Y_test_hat_full = horzcat(Y_test_hat_cell{:});
Y_hat = [nan(n, test_ind(1)), Y_test_hat_full, nan(n, N - test_ind(end))];
if iscell(Y); Y_hat = mat2cell(Y_hat, n, diff(break_ind)); end
valid = all(isfinite(Y_test_hat_full), 1);
E_test = Y_test_hat_full(:, valid) - Y_test_full(:, valid);
Y_valid = Y_test_full(:, valid);
R2 = 1 - sum(E_test.^2, 2) ./ sum((Y_valid - mean(Y_valid, 2)).^2, 2);
if compute_whiteness
    [whiteness.p, whiteness.stat, whiteness.sig_thr] = my_whitetest_multivar(E_test);
else
    % Fragmented AD tests must not concatenate residuals for a time-series test.
    whiteness = struct('p', NaN, 'stat', NaN, 'sig_thr', NaN, ...
        'reason', 'skipped: compute_whiteness=false');
end
end


function mechanism = hlgsde_mechanistic_fold(model, Y_train_cell)
[X, dX] = hlgsde_training_pairs(Y_train_cell, model.DeltaT);
Z = hlgsde_transfer(X, model.alpha);
raw = {model.W_sym, model.F_grad, model.F_curl, model.F_harm};
names = {'sym','grad','curl','harm'};
core_drift = -model.D .* X;
mechanism.channel_drift_rms = zeros(1, 4);
for i = 1:4
    contribution = model.channel_gain(:, i) .* ...
        ((raw{i} ./ model.component_scales(i)) * Z);
    core_drift = core_drift + contribution;
    mechanism.channel_drift_rms(i) = sqrt(mean(contribution(:).^2));
    mechanism.(['E_' names{i}]) = norm(raw{i}, 'fro')^2;
end
mechanism.E_anti = norm(model.W_anti_hodge, 'fro')^2;
for i = 2:4
    mechanism.([names{i} '_energy_fraction']) = ...
        mechanism.(['E_' names{i}]) / max(mechanism.E_anti, eps);
end
mechanism.anti_to_gc_energy_ratio = norm(model.W_anti, 'fro')^2 / ...
    max(norm(model.W_gc, 'fro')^2, eps);
mechanism.anti_energy_fraction = mechanism.anti_to_gc_energy_ratio;
mechanism.phi = model.hodge.phi(:);
mechanism.phi_range = range(mechanism.phi);
mechanism.phi_variance = var(mechanism.phi);
mechanism.psi_energy = mean(model.hodge.psi(:).^2, 'omitnan');
mechanism.curl_triangle_participation = mean(abs(model.hodge.psi(:)) > 1e-10);
mechanism.regional_grad_strength = sqrt(mean(model.F_grad.^2, 2));
mechanism.regional_curl_strength = sqrt(mean(model.F_curl.^2, 2));
mechanism.regional_harm_strength = sqrt(mean(model.F_harm.^2, 2));
mechanism.D = model.D(:);
mechanism.sigma = model.sigma(:);
if any(model.W_residual(:))
    residual_drift = hlgsde_residual_drift( ...
        struct('W_residual', model.W_residual, ...
        'projection', model.residual_projection), Z, ...
        {model.W_sym ./ model.component_scales(1) * Z, ...
        model.F_grad ./ model.component_scales(2) * Z, ...
        model.F_curl ./ model.component_scales(3) * Z, ...
        model.F_harm ./ model.component_scales(4) * Z}, X);
else
    residual_drift = zeros(size(dX));
end
tss = sum((dX - mean(dX, 2)).^2, 'all');
mechanism.core_explained_variance = 1 - sum((dX - core_drift).^2, 'all') / max(tss, eps);
mechanism.total_explained_variance = 1 - ...
    sum((dX - core_drift - residual_drift).^2, 'all') / max(tss, eps);
mechanism.residual_incremental_variance = ...
    mechanism.total_explained_variance - mechanism.core_explained_variance;
mechanism.residual_drift_rms = sqrt(mean(residual_drift(:).^2));
mechanism.hodge_reconstruction_residual = model.hodge.residual_norm;
end


function ablation = hlgsde_channel_ablation(model, Y, Y_test_cell, ...
        break_ind, test_ind, N_test_vec, n, N, k, full_R2, full_whiteness)
names = {'sym','grad','curl','harm'};
ablation.channel = names;
ablation.delta_R2 = nan(1, 4);
ablation.delta_whiteness = nan(1, 4);
full_white = full_whiteness.stat / max(full_whiteness.sig_thr, eps);
for i = 1:4
    reduced = model;
    reduced.channel_gain(:, i) = 0;
    [R2, whiteness] = hlgsde_evaluate(reduced, Y, Y_test_cell, ...
        break_ind, test_ind, N_test_vec, n, N, k, model.options.compute_whiteness);
    ablation.delta_R2(i) = mean(full_R2, 'omitnan') - mean(R2, 'omitnan');
    ablation.delta_whiteness(i) = ...
        whiteness.stat / max(whiteness.sig_thr, eps) - full_white;
end


end


function mechanism = hlgsde_nonhodge_mechanistic_fold(model)
% GC controls retain fit metadata but are not Hodge mechanism estimates.
mechanism = struct('computed', false, 'topology_mode', model.topology_mode, ...
    'component_names', {model.component_names}, ...
    'channel_drift_rms', nan(1, 4), 'hodge_interpretation', ...
    'not applicable to non-Hodge control');
end


function ablation = hlgsde_empty_ablation()
ablation = struct('channel', {{'sym','grad','curl','harm'}}, ...
    'delta_R2', nan(1, 4), 'delta_whiteness', nan(1, 4), ...
    'computed', false);
end


function [noise, selection] = hlgsde_structured_noise(residual, opts)
% Low-rank correlated innovations with a diagonal remainder.
n = size(residual, 1); n_time = size(residual, 2);
split = max(2, min(n_time - 1, floor(0.8 * n_time)));
% The default is diagonal diffusion. Computing an n-by-n covariance and its
% eigendecomposition is unnecessary when no correlated rank is requested.
if isscalar(opts.diffusion_rank_grid) && opts.diffusion_rank_grid == 0
    fit_diag = mean(residual(:, 1:split).^2, 2);
    val_diag = mean(residual(:, split + 1:end).^2, 2);
    fit_diag = max(fit_diag, eps);
    val_diag = max(val_diag, eps);
    validation_error = sum((val_diag - fit_diag).^2) / ...
        max(sum(val_diag.^2), eps);
    noise = struct('factor', zeros(n, 0), 'sigma', sqrt(fit_diag));
    selection = struct('rank', 0, 'shrinkage', 1, ...
        'validation_error', validation_error, ...
        'validation_fraction', 1 - split / n_time);
    return
end
C_fit = residual(:, 1:split) * residual(:, 1:split)' / split;
C_val = residual(:, split + 1:end) * residual(:, split + 1:end)' / ...
    max(n_time - split, 1);
best_error = inf; best = struct('factor', zeros(n, 0), ...
    'sigma', sqrt(max(diag(C_fit), eps)), 'rank', 0, 'shrinkage', 1);
for shrinkage = opts.diffusion_shrinkage_grid
    C = (1 - shrinkage) * C_fit + shrinkage * diag(diag(C_fit));
    [U, S] = eig((C + C') / 2, 'vector');
    [S, order] = sort(max(S, 0), 'descend'); U = U(:, order);
    for requested_rank = opts.diffusion_rank_grid
        rank = min(requested_rank, n);
        factor = U(:, 1:rank) .* sqrt(S(1:rank))';
        sigma2 = max(diag(C) - sum(factor.^2, 2), eps);
        C_model = factor * factor' + diag(sigma2);
        error = norm(C_val - C_model, 'fro')^2 / max(norm(C_val, 'fro')^2, eps);
        if error < best_error
            best_error = error;
            best = struct('factor', factor, 'sigma', sqrt(sigma2), ...
                'rank', rank, 'shrinkage', shrinkage);
        end
    end
end
noise = rmfield(best, {'rank','shrinkage'});
selection = struct('rank', best.rank, 'shrinkage', best.shrinkage, ...
    'validation_error', best_error, 'validation_fraction', 1 - split / n_time);
end


function Z = hlgsde_transfer(X, alpha)
b = 20 / 3;
Z = sqrt(alpha.^2 + (b * X + 0.5).^2) - sqrt(alpha.^2 + (b * X - 0.5).^2);
end
