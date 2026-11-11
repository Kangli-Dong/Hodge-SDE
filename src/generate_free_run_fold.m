function [Y_generated, supported, reason] = generate_free_run_fold( ...
        Y, models, model_names, test_range, options)
%GENERATE_FREE_RUN_FOLD Autonomous stochastic rollout for one CV fold.
% Every model receives the same observed warm-up at the start of each test
% segment. After warm-up, no true test samples are supplied to the model.

if nargin < 5; options = struct(); end
options = defaults(options);
[Y_train, Y_test, ~, test_ind, ~, n, N] = tt_decomp(Y, test_range);
n_method = numel(models);
Y_generated = repmat({nan(n, N)}, n_method, 1);
supported = false(1, n_method);
reason = repmat({''}, 1, n_method);

train_all = [Y_train{:}];
normalization.mu = mean(train_all, 2);
normalization.sigma = std(train_all, 0, 2);
normalization.sigma(~isfinite(normalization.sigma) | ...
    normalization.sigma <= eps) = 1;
scale = normalization.sigma;

for method = 1:n_method
    if isstruct(models{method}) && isfield(models{method}, 'external_free_runs') && ...
            size(models{method}.external_free_runs, 1) == n && ...
            size(models{method}.external_free_runs, 2) == N
        seed0 = 73001;
        if isfield(models{method}, 'generation_seed'); seed0 = models{method}.generation_seed; end
        repeat = 1 + mod(floor((options.seed - seed0) / 100000), ...
            size(models{method}.external_free_runs, 3));
        Y_generated{method} = models{method}.external_free_runs(:, :, repeat);
        supported(method) = any(isfinite(Y_generated{method}(:)));
        reason{method} = 'author baseline autonomous stochastic rollout';
        continue
    end
    if isstruct(models{method}) && isfield(models{method}, 'external_free_run') && ...
            isequal(size(models{method}.external_free_run), [n N])
        Y_generated{method} = models{method}.external_free_run;
        supported(method) = any(isfinite(Y_generated{method}(:)));
        reason{method} = 'author baseline autonomous rollout';
        continue
    end
    if isstruct(models{method}) && isfield(models{method}, 'external_artifact_file')
        [external_runs, external_ok, external_message] = external_free_runs( ...
            models{method}.external_artifact_file, n, N);
        if external_ok
            seed0 = 73001;
            if isfield(models{method}, 'generation_seed'); seed0 = models{method}.generation_seed; end
            repeat = 1 + mod(floor((options.seed - seed0) / 100000), ...
                size(external_runs, 3));
            Y_generated{method} = external_runs(:, :, repeat);
            supported(method) = any(isfinite(Y_generated{method}(:)));
            reason{method} = 'author baseline autonomous stochastic rollout';
        else
            reason{method} = external_message;
        end
        continue
    end
    generated = nan(n, N);
    method_ok = true;
    method_reason = '';
    for segment = 1:numel(Y_test)
        observed = Y_test{segment};
        warmup = min(options.warmup_points, size(observed, 2) - 2);
        if warmup < 2
            method_ok = false; method_reason = 'test segment is too short'; break
        end
        try
            segment_options = options;
            segment_options.seed = options.seed + segment - 1;
            rollout = rollout_segment(observed, Y_train, models{method}, ...
                char(model_names{method}), warmup, normalization, segment_options);
        catch ME
            method_ok = false;
            method_reason = sprintf('%s: %s', ME.identifier, ME.message);
            break
        end
        generated_index = test_ind(segment) + (warmup + 1:size(observed, 2));
        values = rollout(:, warmup + 1:end);
        standardized_amplitude = max(abs(values) ./ scale, [], 'all');
        if any(~isfinite(values), 'all') || ...
                standardized_amplitude > options.max_standardized_amplitude
            method_ok = false;
            method_reason = sprintf('free run diverged (max standardized amplitude %.3g)', ...
                standardized_amplitude);
            break
        end
        generated(:, generated_index) = values;
    end
    if method_ok
        Y_generated{method} = generated;
        supported(method) = true;
        reason{method} = 'autonomous stochastic dynamics with training residual noise';
    else
        reason{method} = method_reason;
    end
end
end


function [runs, ok, message] = external_free_runs(filename, n, N)
runs = [];
ok = false;
message = 'external baseline artifact is unavailable';
if ~(ischar(filename) || (isstring(filename) && isscalar(filename))) || ...
        exist(filename, 'file') ~= 2
    return
end
try
    loaded = load(filename, 'free_runs');
    if ~isfield(loaded, 'free_runs') || size(loaded.free_runs, 1) ~= n || ...
            size(loaded.free_runs, 2) ~= N || isempty(loaded.free_runs)
        message = 'external baseline artifact has no compatible free_runs';
        return
    end
    runs = loaded.free_runs;
    ok = true;
catch ME
    message = sprintf('external baseline artifact could not be read: %s', ME.message);
end
end


function options = defaults(options)
default = struct('warmup_points', 20, 'max_standardized_amplitude', 20, ...
    'execution_environment', 'cpu', 'stochastic', true, 'seed', 73001);
names = fieldnames(default);
for i = 1:numel(names)
    if ~isfield(options, names{i}) || isempty(options.(names{i}))
        options.(names{i}) = default.(names{i});
    end
end
end


function output = rollout_segment(observed, training, model, name, ...
        warmup, normalization, options)
output = nan(size(observed));
output(:, 1:warmup) = observed(:, 1:warmup);
name_lower = lower(name);
stream = RandStream('mt19937ar', 'Seed', options.seed);

if contains(name_lower, 'w/ hrf')
    error('HLGSDE:UnsupportedGenerator', ...
        'the fitted HRF model does not save a reusable latent-state initializer');
elseif strcmpi(name, 'Zero')
    for t = warmup+1:size(output, 2); output(:, t) = output(:, t-1); end
elseif contains(name_lower, 'linear') || contains(name_lower, 'var-') || ...
        contains(name_lower, 'ar-')
    output = rollout_ar(output, model, warmup, options, stream);
elseif strcmpi(name, 'Subspace')
    output = rollout_subspace(output, model, warmup);
elseif strcmpi(name, 'NMM')
    for t = warmup+1:size(output, 2)
        output(:, t) = mindy_step(model, output(:, t-1));
    end
elseif strcmpi(name, 'Manifold')
    output = rollout_manifold(output, training, model, warmup);
elseif strcmpi(name, 'MLP')
    output = rollout_feedforward(output, model, warmup, normalization, ...
        options, 'mlp');
elseif strcmpi(name, 'CNN')
    output = rollout_feedforward(output, model, warmup, normalization, ...
        options, 'cnn');
elseif strcmpi(name, 'LSTM1') || strcmpi(name, 'LSTM2')
    output = rollout_lstm(output, training, model, warmup, normalization, ...
        options, strcmpi(name, 'LSTM1'));
elseif contains(name_lower, 'hl-gsde') || contains(name_lower, 'gc-gsde') || ...
        contains(name_lower, 'hodge-mindy')
    dt = model_step_dt(model);
    for t = warmup+1:size(output, 2)
        output(:, t) = output(:, t-1) + dt * hlgsde_mean_drift(model, output(:, t-1)) + ...
            sqrt(dt) * innovation(model, options, stream, size(output, 1));
    end
else
    error('HLGSDE:UnsupportedGenerator', 'no autonomous generator for %s', name);
end
end


function output = rollout_ar(output, model, warmup, options, stream)
for t = warmup+1:size(output, 2)
    previous = output(:, t-1);
    increment = zeros(size(previous));
    if isfield(model, 'W'); increment = increment + model.W * previous; end
    if isfield(model, 'D') && isfield(model, 'Tau')
        for i = 1:numel(model.Tau)
            lag = model.Tau(i);
            if t-lag < 1; error('HLGSDE:WarmupTooShort', 'AR warm-up is too short.'); end
            increment = increment + model.D{i} * output(:, t-lag);
        end
    end
    output(:, t) = previous + increment + ...
        innovation(model, options, stream, size(output, 1));
end
end


function output = rollout_subspace(output, model, warmup)
A = eye(size(model.W)) + model.W;
C = model.C;
Q = symmetric_psd(model.Q);
R = symmetric_psd(model.R);
x = zeros(size(A, 1), 1);
P = eye(size(A));
for t = 1:warmup
    x = A * x;
    P = A * P * A' + Q;
    innovation_cov = C * P * C' + R;
    gain = (P * C') / (innovation_cov + 1e-9 * eye(size(R)));
    x = x + gain * (output(:, t) - C * x);
    P = P - gain * C * P;
end
for t = warmup+1:size(output, 2)
    x = A * x;
    output(:, t) = C * x;
end
end


function matrix = symmetric_psd(matrix)
matrix = 0.5 * (matrix + matrix');
[V, D] = eig(matrix, 'vector');
matrix = V * diag(max(real(D), 0)) * V';
matrix = real(0.5 * (matrix + matrix'));
end


function next = mindy_step(model, state)
b = model.b;
transfer = sqrt(model.alpha.^2 + (b * state + 0.5).^2) - ...
    sqrt(model.alpha.^2 + (b * state - 0.5).^2);
next = (1 - model.D) .* state + model.W * transfer;
end


function output = rollout_manifold(output, training, model, warmup)
d = model.d;
[train_lags, train_diff] = manifold_training_pairs(training, d);
h_original = 830;
base_median_distance = 565;
for t = warmup+1:size(output, 2)
    query = lag_vector(output, t, d);
    centered = train_lags - query;
    distance = sqrt(sum(centered.^2, 1));
    h = h_original * median(distance) / base_median_distance;
    weight = exp(-distance.^2 / max(2 * h^2, eps));
    design = [ones(1, size(centered, 2)); centered];
    weighted_design = design .* weight;
    theta = (train_diff .* weight) * design' * pinv(weighted_design * design');
    output(:, t) = output(:, t-1) + theta(:, 1);
end
end


function [lags, difference] = manifold_training_pairs(training, d)
lags = []; difference = [];
for i = 1:numel(training)
    Y = training{i};
    if size(Y, 2) <= d; continue; end
    current_lags = cell2mat(arrayfun(@(lag)Y(:, 1+d-lag:end-lag), ...
        (1:d)', 'UniformOutput', false));
    lags = [lags current_lags]; %#ok<AGROW>
    difference = [difference diff(Y(:, d:end), 1, 2)]; %#ok<AGROW>
end
end


function vector = lag_vector(history, t, d)
vector = cell2mat(arrayfun(@(lag)history(:, t-lag), ...
    (1:d)', 'UniformOutput', false));
end


function output = rollout_feedforward(output, model, warmup, norm_info, ...
        options, architecture)
normalized = (output - norm_info.mu) ./ norm_info.sigma;
if strcmp(architecture, 'cnn'); d = model.d; else; d = 1; end
for t = warmup+1:size(output, 2)
    if strcmp(architecture, 'cnn')
        lag_tensor = zeros(size(output, 1), 1, d);
        for lag = 1:d; lag_tensor(:, 1, lag) = normalized(:, t-lag); end
        input = permute(lag_tensor, [4 3 1 2]);
    else
        input = permute(normalized(:, t-1), [3 4 1 2]);
    end
    delta = predict(model.f, input, ...
        'ExecutionEnvironment', options.execution_environment);
    delta = reshape(delta, [], 1);
    normalized(:, t) = normalized(:, t-1) + delta;
    output(:, t) = norm_info.sigma .* normalized(:, t) + norm_info.mu;
end
end


function output = rollout_lstm(output, training, model, warmup, norm_info, ...
        options, infinite_memory)
normalized = (output - norm_info.mu) ./ norm_info.sigma;
net = resetState(model.f);
if infinite_memory
    training_normalized = cellfun(@(Y)(Y - norm_info.mu) ./ norm_info.sigma, ...
        training, 'UniformOutput', false);
    net = predictAndUpdateState(net, training_normalized', ...
        'ExecutionEnvironment', options.execution_environment, 'MiniBatchSize', 1);
    for t = 1:warmup-1
        [net, ~] = predictAndUpdateState(net, normalized(:, t), ...
            'ExecutionEnvironment', options.execution_environment, 'MiniBatchSize', 1);
    end
    for t = warmup+1:size(output, 2)
        [net, delta] = predictAndUpdateState(net, normalized(:, t-1), ...
            'ExecutionEnvironment', options.execution_environment, 'MiniBatchSize', 1);
        normalized(:, t) = normalized(:, t-1) + reshape(delta, [], 1);
        output(:, t) = norm_info.sigma .* normalized(:, t) + norm_info.mu;
    end
else
    for t = warmup+1:size(output, 2)
        delta = predict(net, normalized(:, t-1), ...
            'ExecutionEnvironment', options.execution_environment, 'MiniBatchSize', 1);
        normalized(:, t) = normalized(:, t-1) + reshape(delta, [], 1);
        output(:, t) = norm_info.sigma .* normalized(:, t) + norm_info.mu;
    end
end
end


function drift = hlgsde_mean_drift(model, state)
b = model.b;
transfer = sqrt(model.alpha.^2 + (b * state + 0.5).^2) - ...
    sqrt(model.alpha.^2 + (b * state - 0.5).^2);
if isfield(model, 'component_matrices')
    components = model.component_matrices;
else
    components = {model.W_sym, model.F_grad, model.F_curl, model.F_harm};
end
drift = zeros(size(state));
for component = 1:numel(components)
    topology = components{component} ./ model.component_scales(component);
    drift = drift + model.channel_gain(:, component) .* (topology * transfer);
end
if isfield(model, 'W_residual') && ...
        ~(isfield(model, 'generation_use_core_drift') && model.generation_use_core_drift)
    residual_drift = model.W_residual * transfer;
    if isfield(model, 'residual_projection')
        for i = 1:numel(state)
            features = [ ...
                (components{1}(i, :) ./ model.component_scales(1)) * transfer; ...
                (components{2}(i, :) ./ model.component_scales(2)) * transfer; ...
                (components{3}(i, :) ./ model.component_scales(3)) * transfer; ...
                (components{4}(i, :) ./ model.component_scales(4)) * transfer; ...
                -state(i)];
            residual_drift(i) = residual_drift(i) - ...
                model.residual_projection(i, :) * features;
        end
    end
    drift = drift + residual_drift;
end
drift = drift - model.D .* state;
end


function dt = model_step_dt(model)
if isfield(model, 'DeltaT') && isfinite(model.DeltaT) && model.DeltaT > 0
    dt = model.DeltaT;
else
    dt = 1;
end
end


function value = innovation(model, options, stream, n)
value = zeros(n, 1);
if ~options.stochastic; return; end
if isfield(model, 'generation_sigma')
    sigma = model.generation_sigma;
elseif isfield(model, 'sigma')
    sigma = model.sigma;
else
    sigma = [];
end
if numel(sigma) == n
    value = sigma(:) .* randn(stream, n, 1);
end
if isfield(model, 'generation_noise_factor')
    factor = model.generation_noise_factor;
elseif isfield(model, 'noise_factor')
    factor = model.noise_factor;
else
    factor = [];
end
if ~isempty(factor)
    value = value + factor * randn(stream, size(factor, 2), 1);
end
end
