function trajectory = simulate_hodge_parameters(parameters, initial_state, n_steps, seed, stochastic)
%SIMULATE_HODGE_PARAMETERS Run saved Hodge-SDE parameters without source data.
% Returns ROI x (n_steps+1), including the user-supplied initial state.
% Matches the original Euler-Maruyama generator and its saved diffusion scale.
if nargin<4; seed=73001; end
if nargin<5; stochastic=true; end
assert(isstruct(parameters) && strcmp(parameters.format,'hodge_sde_parameters_v1'));
n = numel(parameters.D);
assert(isnumeric(initial_state) && isreal(initial_state) && isequal(size(initial_state),[n 1]) && all(isfinite(initial_state)));
assert(isscalar(n_steps) && n_steps>=0 && n_steps==floor(n_steps));
assert(isscalar(stochastic) && ismember(stochastic,[false true]));
stream = RandStream('mt19937ar','Seed',seed);
dt = parameters.DeltaT;
trajectory = zeros(n,n_steps+1);
trajectory(:,1) = initial_state;
for t = 1:n_steps
    state = trajectory(:,t);
    transfer = sqrt(parameters.alpha.^2+(parameters.b*state+0.5).^2) - ...
        sqrt(parameters.alpha.^2+(parameters.b*state-0.5).^2);
    drift = zeros(n,1);
    components = parameters.component_matrices;
    for q = 1:4
        drift = drift + parameters.channel_gain(:,q) .* ...
            ((components{q}./parameters.component_scales(q))*transfer);
    end
    if ~parameters.generation_use_core_drift
        residual = parameters.W_residual*transfer;
        for i = 1:n
            features = [(components{1}(i,:)./parameters.component_scales(1))*transfer; ...
                (components{2}(i,:)./parameters.component_scales(2))*transfer; ...
                (components{3}(i,:)./parameters.component_scales(3))*transfer; ...
                (components{4}(i,:)./parameters.component_scales(4))*transfer; -state(i)];
            residual(i) = residual(i)-parameters.residual_projection(i,:)*features;
        end
        drift = drift+residual;
    end
    drift = drift-parameters.D.*state;
    noise = zeros(n,1);
    if stochastic
        noise = parameters.generation_sigma.*randn(stream,n,1);
        factor = parameters.generation_noise_factor;
        if ~isempty(factor); noise = noise+factor*randn(stream,size(factor,2),1); end
    end
    trajectory(:,t+1) = state+dt*drift+sqrt(dt)*noise;
    assert(all(isfinite(trajectory(:,t+1))),'Generated trajectory became nonfinite.');
end
end
