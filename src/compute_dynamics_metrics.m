function [metrics, diagnostics] = compute_dynamics_metrics(Y_true, Y_hat_cv, TR, options)
%COMPUTE_DYNAMICS_METRICS Compare caller-matched trajectory statistics.
% Temporal calculations never cross a CV-fold gap represented by NaNs.
% For autonomous trajectories use score_free_run_by_fold with
% strict_generation=true; predicted and autonomous inputs remain distinct.

if nargin < 4
    options = struct();
end
options = apply_defaults(options);
n_method = numel(Y_hat_cv);

metric_names = {'mle_real', 'mle_model', 'mle_abs_error', ...
    'mle_real_ci_low', 'mle_real_ci_high', 'mle_model_ci_low', ...
    'mle_model_ci_high', 'mle_regime_real', 'mle_regime_model', ...
    'mle_regime_match', 'sw2_distance', 'irreversibility_real', ...
    'irreversibility_model', 'irreversibility_abs_error', 'fcd_ks', ...
    'static_fc_error', 'persistence_skill', 'n_valid_points', ...
    'n_segments', 'n_fcd_windows', 'fcd_variance', ...
    'fcd_repeat_fraction', 'fcd_collapsed', 'fcd_ks_raw'};
for i_field = 1:numel(metric_names)
    metrics.(metric_names{i_field}) = nan(1, n_method);
end

empty_diagnostic = struct('real_phase', [], 'model_phase', [], ...
    'real_fcd', [], 'model_fcd', [], 'valid_time_index', [], ...
    'fcd_variance', NaN, 'fcd_repeat_fraction', NaN, 'fcd_collapsed', false, ...
    'strict_rejected', false, 'status', "undefined", 'reason', "", ...
    'metric_support', struct(), 'metric_reason', struct(), ...
    'mle_repeated_states', false, 'correlation_undefined', false);
diagnostics = repmat(empty_diagnostic, 1, n_method);
prediction_cache = cell(1, n_method);
for i_method = 1:n_method
    prediction_cache{i_method} = normalize_prediction( ...
        Y_hat_cv{i_method}, size(Y_true));
end
reference_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');

for i_method = 1:n_method
    Y_pred = prediction_cache{i_method};
    if options.strict_generation && (~isequal(size(Y_true), size(Y_pred)) || ...
            any(~isfinite(Y_true), 'all') || any(~isfinite(Y_pred), 'all'))
        diagnostics(i_method).strict_rejected = true;
        diagnostics(i_method).status = "error";
        diagnostics(i_method).reason = "strict_generation requires same-shape, fully finite trajectories";
        [diagnostics(i_method).metric_support, diagnostics(i_method).metric_reason] = ...
            metric_contract(metrics, i_method, diagnostics(i_method).reason);
        continue
    end
    [real_segments, model_segments, valid_index] = matched_segments( ...
        Y_true, Y_pred, options.min_segment_points);
    metrics.n_valid_points(i_method) = nnz(valid_index);
    metrics.n_segments(i_method) = numel(real_segments);
    diagnostics(i_method).valid_time_index = valid_index;

    if isempty(real_segments) || nnz(valid_index) < 2 * options.min_segment_points
        diagnostics(i_method).reason = "insufficient matched contiguous finite samples";
        [diagnostics(i_method).metric_support, diagnostics(i_method).metric_reason] = ...
            metric_contract(metrics, i_method, diagnostics(i_method).reason);
        continue
    end

    model_all = [model_segments{:}];
    cache_key = valid_mask_key(valid_index);
    if isKey(reference_cache, cache_key)
        reference = reference_cache(cache_key);
    else
        reference = prepare_reference_context(real_segments, TR, options);
        reference_cache(cache_key) = reference;
    end
    model_state = project_segments(model_segments, reference.basis, reference.center);

    [lambda_model, ci_model, model_mle_repeated] = rosenstein_segmented(model_state, TR, ...
        options.mle_horizon, options.mle_theiler, options.strict_generation);
    metrics.mle_real(i_method) = reference.lambda;
    metrics.mle_model(i_method) = lambda_model;
    metrics.mle_abs_error(i_method) = abs(lambda_model - reference.lambda);
    metrics.mle_real_ci_low(i_method) = reference.lambda_ci(1);
    metrics.mle_real_ci_high(i_method) = reference.lambda_ci(2);
    metrics.mle_model_ci_low(i_method) = ci_model(1);
    metrics.mle_model_ci_high(i_method) = ci_model(2);
    metrics.mle_regime_real(i_method) = reference.lambda_regime;
    metrics.mle_regime_model(i_method) = lyapunov_regime(ci_model);
    if isfinite(metrics.mle_regime_real(i_method)) && ...
            metrics.mle_regime_real(i_method) ~= 0
        metrics.mle_regime_match(i_method) = ...
            metrics.mle_regime_real(i_method) == metrics.mle_regime_model(i_method);
    end

    metrics.sw2_distance(i_method) = sliced_wasserstein2( ...
        reference.state_points, [model_state{:}]', options.sw2_projections, 1);

    epr_model_state = apply_projected_scale(model_state, ...
        reference.epr_scale, reference.epr_dim);
    irr_model = transition_irreversibility_model( ...
        epr_model_state, reference.epr_centroids, TR);
    metrics.irreversibility_real(i_method) = reference.irreversibility;
    metrics.irreversibility_model(i_method) = irr_model;
    metrics.irreversibility_abs_error(i_method) = ...
        abs(irr_model - reference.irreversibility);

    [fcd_model, n_windows_model, model_fcd_status] = fcd_distribution(model_segments, ...
        reference.window_points, reference.step_points, options.strict_generation);
    fcd_variance = var(fcd_model, 'omitnan');
    if isempty(fcd_model)
        repeat_fraction = NaN;
    else
        repeat_fraction = 1 - numel(unique(round(fcd_model, 8))) / numel(fcd_model);
    end
    collapsed = numel(fcd_model) < 20 || fcd_variance < 1e-6 || repeat_fraction > 0.9;
    metrics.fcd_ks(i_method) = ks_distance(reference.fcd, fcd_model);
    metrics.fcd_ks_raw(i_method) = metrics.fcd_ks(i_method);
    if collapsed && ~options.strict_generation; metrics.fcd_ks(i_method) = NaN; end
    metrics.n_fcd_windows(i_method) = min(reference.n_fcd_windows, n_windows_model);
    metrics.fcd_variance(i_method) = fcd_variance;
    metrics.fcd_repeat_fraction(i_method) = repeat_fraction;
    metrics.fcd_collapsed(i_method) = collapsed;
    metrics.static_fc_error(i_method) = static_fc_distance( ...
        reference.static_fc, model_all, options.strict_generation);

    phase_dim = min(2, size(reference.basis, 2));
    diagnostics(i_method).real_phase = [reference.state{:}];
    diagnostics(i_method).model_phase = [model_state{:}];
    diagnostics(i_method).real_phase = diagnostics(i_method).real_phase(1:phase_dim, :)';
    diagnostics(i_method).model_phase = diagnostics(i_method).model_phase(1:phase_dim, :)';
    if phase_dim == 1
        diagnostics(i_method).real_phase(:, 2) = 0;
        diagnostics(i_method).model_phase(:, 2) = 0;
    end
    diagnostics(i_method).real_fcd = reference.fcd;
    diagnostics(i_method).model_fcd = fcd_model;
    diagnostics(i_method).fcd_variance = fcd_variance;
    diagnostics(i_method).fcd_repeat_fraction = repeat_fraction;
    diagnostics(i_method).fcd_collapsed = collapsed;
    diagnostics(i_method).mle_repeated_states = reference.mle_repeated_states || model_mle_repeated;
    diagnostics(i_method).correlation_undefined = reference.fcd_status.undefined_correlation || ...
        model_fcd_status.undefined_correlation || ~isfinite(metrics.static_fc_error(i_method));
    diagnostics(i_method).status = "completed";
    diagnostics(i_method).reason = "";
    [diagnostics(i_method).metric_support, diagnostics(i_method).metric_reason] = ...
        metric_contract(metrics, i_method, "");
end

% Fair forecast skill relative to method 1, the increment-zero/persistence
% predictor. Positive values beat persistence on exactly matched points.
if n_method >= 1
    persistence = prediction_cache{1};
    for i_method = 1:n_method
        prediction = prediction_cache{i_method};
        valid = all(isfinite(Y_true), 1) & all(isfinite(persistence), 1) & ...
            all(isfinite(prediction), 1);
        if ~any(valid); continue; end
        reference_error = Y_true(:, valid) - persistence(:, valid);
        model_error = Y_true(:, valid) - prediction(:, valid);
        reference_sse = sum(reference_error(:).^2);
        if reference_sse > eps
            metrics.persistence_skill(i_method) = ...
                1 - sum(model_error(:).^2) / reference_sse;
        end
    end
end
end

function key = valid_mask_key(valid)
key = char('0' + valid(:)');
end

function reference = prepare_reference_context(real_segments, TR, options)
real_all = [real_segments{:}];
[reference.basis, reference.center] = real_pca_basis(real_all, ...
    options.pca_max_dim, options.pca_variance);
reference.state = project_segments(real_segments, ...
    reference.basis, reference.center);
reference.state_points = [reference.state{:}]';
[reference.lambda, reference.lambda_ci, reference.mle_repeated_states] = rosenstein_segmented( ...
    reference.state, TR, options.mle_horizon, options.mle_theiler, options.strict_generation);
reference.lambda_regime = lyapunov_regime(reference.lambda_ci);
reference.epr_dim = min(options.epr_pca_dim, size(reference.basis, 2));
[epr_state, reference.epr_scale] = standardize_projected( ...
    reference.state, reference.epr_dim);
[reference.irreversibility, reference.epr_centroids] = ...
    transition_irreversibility_reference(epr_state, TR);
reference.window_points = max(4, round(options.fcd_window_seconds / TR));
reference.step_points = max(1, round(options.fcd_step_seconds / TR));
[reference.fcd, reference.n_fcd_windows, reference.fcd_status] = fcd_distribution( ...
    real_segments, reference.window_points, reference.step_points, options.strict_generation);
reference.static_fc = static_fc_matrix(real_all, options.strict_generation);
end

function options = apply_defaults(options)
defaults = struct('pca_max_dim', 10, 'pca_variance', 0.90, ...
    'mle_horizon', 4, 'mle_theiler', 5, 'sw2_projections', 128, ...
    'epr_pca_dim', 3, 'fcd_window_seconds', 40, ...
    'fcd_step_seconds', 4, 'min_segment_points', 12, ...
    'strict_generation', false);
names = fieldnames(defaults);
for i = 1:numel(names)
    if ~isfield(options, names{i}) || isempty(options.(names{i}))
        options.(names{i}) = defaults.(names{i});
    end
end
end

function Y_pred = normalize_prediction(value, expected_size)
if iscell(value)
    if isempty(value)
        Y_pred = nan(expected_size);
        return
    end
    Y_pred = cell2mat(value(:)');
else
    Y_pred = value;
end
if ~isequal(size(Y_pred), expected_size)
    Y_pred = nan(expected_size);
end
end

function [real_segments, model_segments, valid] = matched_segments(Y, Y_hat, min_length)
valid = all(isfinite(Y), 1) & all(isfinite(Y_hat), 1);
changes = diff([false valid false]);
starts = find(changes == 1);
stops = find(changes == -1) - 1;
keep = (stops - starts + 1) >= min_length;
starts = starts(keep);
stops = stops(keep);
real_segments = cell(1, numel(starts));
model_segments = cell(1, numel(starts));
valid(:) = false;
for i = 1:numel(starts)
    index = starts(i):stops(i);
    real_segments{i} = Y(:, index);
    model_segments{i} = Y_hat(:, index);
    valid(index) = true;
end
end

function [basis, center] = real_pca_basis(Y, max_dim, variance_fraction)
center = mean(Y, 2);
centered = (Y - center)';
[~, singular_values, basis_all] = svd(centered, 'econ');
latent = diag(singular_values).^2;
if isempty(latent) || sum(latent) <= 0
    basis = eye(size(Y, 1), 1);
    return
end
n_variance = find(cumsum(latent) / sum(latent) >= variance_fraction, 1);
n_component = max(2, min([max_dim, n_variance, size(basis_all, 2)]));
basis = basis_all(:, 1:n_component);
end

function projected = project_segments(segments, basis, center)
projected = cell(size(segments));
for i = 1:numel(segments)
    projected{i} = basis' * (segments{i} - center);
end
end

function [lambda, confidence_interval, repeated_states] = rosenstein_segmented(segments, TR, max_horizon, theiler, strict_generation)
lambda = NaN;
confidence_interval = [NaN NaN];
repeated_states = false;
lengths = cellfun(@(x)size(x, 2), segments);
if isempty(lengths)
    return
end
max_horizon = min(max_horizon, min(lengths) - 1);
if max_horizon < 2
    return
end

start_counts = max(0, lengths - max_horizon);
n_start_total = sum(start_counts);
states = zeros(size(segments{1}, 1), n_start_total);
segment_id = zeros(1, n_start_total);
local_index = zeros(1, n_start_total);
cursor = 0;
for i_segment = 1:numel(segments)
    last_start = start_counts(i_segment);
    index = cursor + (1:last_start);
    states(:, index) = segments{i_segment}(:, 1:last_start);
    segment_id(index) = i_segment;
    local_index(index) = 1:last_start;
    cursor = cursor + last_start;
end
if size(states, 2) < 20
    return
end

n_start = size(states, 2);
neighbor = nan(1, n_start);
for i = 1:n_start
    delta = states - states(:, i);
    distance = sqrt(sum(delta.^2, 1));
    exclude = segment_id == segment_id(i) & ...
        abs(local_index - local_index(i)) <= theiler;
    distance(exclude) = inf;
    [best_distance, neighbor(i)] = min(distance);
    if strict_generation && best_distance <= stable_tolerance(states)
        repeated_states = true;
        neighbor(i) = NaN;
        continue
    end
    if ~isfinite(best_distance)
        neighbor(i) = NaN;
    end
end
valid_pair = find(isfinite(neighbor));
if numel(valid_pair) < 20
    return
end

log_divergence = nan(numel(valid_pair), max_horizon + 1);
for i_pair = 1:numel(valid_pair)
    i = valid_pair(i_pair);
    j = neighbor(i);
    for h = 0:max_horizon
        xi = segments{segment_id(i)}(:, local_index(i) + h);
        xj = segments{segment_id(j)}(:, local_index(j) + h);
        separation = norm(xi - xj);
        if strict_generation && separation <= stable_tolerance(states)
            repeated_states = true;
        else
            log_divergence(i_pair, h + 1) = log(max(separation, 1e-12));
        end
    end
end
if strict_generation && repeated_states
    return
end
mean_log_divergence = mean(log_divergence, 1, 'omitnan')';
time = (0:max_horizon)' * TR;
design = [ones(size(time)), time];
coefficients = design \ mean_log_divergence;
lambda = coefficients(2);
residual = mean_log_divergence - design * coefficients;
dof = max(1, numel(time) - 2);
sigma2 = sum(residual.^2) / dof;
covariance = sigma2 * pinv(design' * design);
slope_se = sqrt(max(covariance(2, 2), 0));
confidence_interval = lambda + 1.96 * [-slope_se slope_se];
end

function regime = lyapunov_regime(confidence_interval)
if any(~isfinite(confidence_interval))
    regime = NaN;
elseif confidence_interval(1) > 0
    regime = 1;
elseif confidence_interval(2) < 0
    regime = -1;
else
    regime = 0;
end
end

function distance = sliced_wasserstein2(X, Y, n_projection, seed)
if isempty(X) || isempty(Y)
    distance = NaN;
    return
end
n_dim = size(X, 2);
stream = RandStream('mt19937ar', 'Seed', 1000 + seed);
directions = randn(stream, n_dim, n_projection);
directions = directions ./ max(vecnorm(directions, 2, 1), eps);
X_projected = X * directions;
Y_projected = Y * directions;
n_quantile = min(size(X_projected, 1), size(Y_projected, 1));
probability = ((1:n_quantile)' - 0.5) / n_quantile;
w2_squared = nan(1, n_projection);
for i = 1:n_projection
    x = sort(X_projected(:, i));
    y = sort(Y_projected(:, i));
    qx = interp1(((1:numel(x))' - 0.5) / numel(x), x, probability, 'linear', 'extrap');
    qy = interp1(((1:numel(y))' - 0.5) / numel(y), y, probability, 'linear', 'extrap');
    w2_squared(i) = mean((qx - qy).^2);
end
distance = sqrt(mean(w2_squared, 'omitnan'));
end

function [standardized, scale] = standardize_projected(segments, n_dim)
all_values = [segments{:}];
scale = std(all_values(1:n_dim, :), 0, 2);
scale(scale <= eps | ~isfinite(scale)) = 1;
standardized = cellfun(@(x)x(1:n_dim, :) ./ scale, segments, 'UniformOutput', false);
end

function standardized = apply_projected_scale(model_segments, scale, n_dim)
standardized = cellfun(@(x)x(1:n_dim, :) ./ scale, model_segments, 'UniformOutput', false);
end

function [real_rate, centroids] = transition_irreversibility_reference(real_segments, TR)
real_points = [real_segments{:}]';
n_state = min(8, max(3, floor(sqrt(size(real_points, 1) / 3))));
if size(real_points, 1) < 3 * n_state
    real_rate = NaN;
    centroids = [];
    return
end

previous_rng = rng;
rng_cleanup = onCleanup(@()rng(previous_rng)); %#ok<NASGU>
rng(2718, 'twister');
[~, centroids] = kmeans(real_points, n_state, 'Replicates', 5, ...
    'MaxIter', 300, 'Display', 'off');
real_counts = transition_count_matrix(real_segments, centroids);
real_rate = transition_kl(real_counts) / TR;
end

function model_rate = transition_irreversibility_model(model_segments, centroids, TR)
if isempty(centroids)
    model_rate = NaN;
    return
end
model_counts = transition_count_matrix(model_segments, centroids);
model_rate = transition_kl(model_counts) / TR;
end

function counts = transition_count_matrix(segments, centroids)
n_state = size(centroids, 1);
counts = zeros(n_state);
for i_segment = 1:numel(segments)
    points = segments{i_segment}';
    distance2 = sum(points.^2, 2) + sum(centroids.^2, 2)' - ...
        2 * (points * centroids');
    [~, labels] = min(distance2, [], 2);
    for t = 1:numel(labels) - 1
        counts(labels(t), labels(t + 1)) = ...
            counts(labels(t), labels(t + 1)) + 1;
    end
end
end

function divergence = transition_kl(counts)
% Jeffreys smoothing prevents infinite KL from finite unobserved transitions.
probability = counts + 0.5;
probability = probability / sum(probability, 'all');
reverse_probability = probability';
divergence = sum(probability .* log(probability ./ reverse_probability), 'all');
end

function [values, n_window, status] = fcd_distribution(segments, window_length, step, strict_generation)
if nargin < 4; strict_generation = false; end
status = struct('valid', false, 'reason', "", 'undefined_correlation', false);
n_region = size(segments{1}, 1);
upper = triu(true(n_region), 1);
n_edge = nnz(upper);
window_counts = cellfun(@(Y)max(0, floor((size(Y, 2) - window_length) / step) + 1), ...
    segments);
features = zeros(sum(window_counts), n_edge);
cursor = 0;
for i_segment = 1:numel(segments)
    Y = segments{i_segment};
    if size(Y, 2) < window_length
        continue
    end
    starts = 1:step:(size(Y, 2) - window_length + 1);
    for start_index = starts
        window = Y(:, start_index:start_index + window_length - 1);
        [FC, fc_valid] = row_correlation(window, strict_generation);
        if strict_generation && ~fc_valid
            values = [];
            n_window = cursor;
            status.reason = "window FC contains undefined correlation";
            status.undefined_correlation = true;
            return
        end
        cursor = cursor + 1;
        features(cursor, :) = FC(upper);
    end
end
n_window = cursor;
features = features(1:n_window, :);
if n_window < 3
    values = [];
    status.reason = "fewer than three FCD windows";
    return
end
[FCD, fcd_valid] = row_correlation(features, strict_generation);
if strict_generation && ~fcd_valid
    values = [];
    status.reason = "FCD vector correlation is undefined";
    status.undefined_correlation = true;
    return
end
values = FCD(tril(true(n_window), -1));
if strict_generation && any(~isfinite(values))
    values = [];
    status.reason = "FCD contains undefined vector correlations";
    status.undefined_correlation = true;
    return
end
status.valid = true;
end

function distance = ks_distance(x, y)
x = x(isfinite(x));
y = y(isfinite(y));
if isempty(x) || isempty(y)
    distance = NaN;
    return
end
[~, ~, distance] = kstest2(x, y);
end

function FC = static_fc_matrix(Y, strict_generation)
[FC, ~] = row_correlation(Y, strict_generation);
end

function distance = static_fc_distance(FC_real, Y_hat, strict_generation)
[FC_model, model_valid] = row_correlation(Y_hat, strict_generation);
if strict_generation && (~model_valid || any(~isfinite(FC_real), 'all'))
    distance = NaN;
    return
end
upper = triu(true(size(FC_real)), 1);
delta = FC_real(upper) - FC_model(upper);
distance = sqrt(mean(delta.^2));
end

function [correlation, valid] = row_correlation(values, strict_generation)
if nargin < 2; strict_generation = false; end
values = values - mean(values, 2);
norms = sqrt(sum(values.^2, 2));
correlation = (values * values') ./ (norms * norms');
if strict_generation
    low_variance = norms <= stable_tolerance(values);
    correlation(low_variance, :) = NaN;
    correlation(:, low_variance) = NaN;
end
valid = all(isfinite(correlation(~eye(size(correlation)))), 'all');
if ~strict_generation
    correlation(~isfinite(correlation)) = 0;
end
correlation(1:size(correlation, 1)+1:end) = 1;
end

function tolerance = stable_tolerance(values)
scale = max(1, max(abs(values(:))));
tolerance = 100 * eps(scale) * sqrt(max(1, size(values, 1)));
end

function [support, reason] = metric_contract(metrics, i_method, fallback_reason)
names = {'sw2_distance', 'fcd_ks', 'static_fc_error', ...
    'irreversibility_abs_error', 'mle_abs_error'};
support = struct();
reason = struct();
for i_name = 1:numel(names)
    name = names{i_name};
    support.(name) = isfinite(metrics.(name)(i_method));
    if support.(name)
        reason.(name) = "";
    elseif strlength(string(fallback_reason)) > 0
        reason.(name) = string(fallback_reason);
    else
        reason.(name) = "undefined by metric contract";
    end
end
end
