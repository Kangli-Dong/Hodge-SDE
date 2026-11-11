function test_parameter_export
% In-memory synthetic test: allowlist, exact noise scaling, and determinism.
addpath(fullfile(fileparts(mfilename('fullpath')),'src'));
n=4;
m=struct('topology_mode','hodge','hodge_source','nonnegative_gc_score', ...
    'alpha',1,'b',20/3,'D',0.1*ones(n,1),'channel_gain',zeros(n,4), ...
    'component_scales',ones(1,4),'component_matrices',{{zeros(n),zeros(n),zeros(n),zeros(n)}}, ...
    'W_residual',zeros(n),'residual_projection',zeros(n,5),'TR',.72,'DeltaT',.72, ...
    'generation_sigma',.2*ones(n,1),'generation_noise_factor',.01*ones(n,2), ...
    'generation_use_core_drift',true,'Subject','synthetic_id', ...
    'source_path','private-path-placeholder','core_validation',struct('Y_hat',ones(n,50)));
p=export_hodge_parameters(m);
assert(~any(isfield(p,{'Subject','source_path','core_validation'})));
x0=(1:n)'/10; steps=12; seed=73;
actual=simulate_hodge_parameters(p,x0,steps,seed,true);
stream=RandStream('mt19937ar','Seed',seed);
expected=zeros(n,steps+1); expected(:,1)=x0;
for t=1:steps
    noise=.2*randn(stream,n,1)+.01*ones(n,2)*randn(stream,2,1);
    expected(:,t+1)=expected(:,t)+.72*(-.1*expected(:,t))+sqrt(.72)*noise;
end
assert(max(abs(actual-expected),[],'all')<1e-14);
deterministic=simulate_hodge_parameters(p,x0,steps,seed,false);
assert(max(abs(deterministic-x0*(1-.072).^(0:steps)),[],'all')<1e-14);
assert(isequal(actual,simulate_hodge_parameters(p,x0,steps,seed,true)));
fprintf('PARAMETER_EXPORT_PASS: numeric allowlist, diffusion scaling and reproducible trajectories.\n');
end
