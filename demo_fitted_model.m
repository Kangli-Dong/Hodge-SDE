function trajectory = demo_fitted_model(roi, stochastic)
%DEMO_FITTED_MODEL Simulate a bundled fitted model from a synthetic initial state.
% The returned trajectory is simulated, not a participant observation.
if nargin<1; roi=150; end
if nargin<2; stochastic=true; end
assert(ismember(roi,[150 250 450]),'Bundled resolutions are 150, 250 and 450.');
root=fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'));
saved=load(fullfile(root,'models',sprintf('hcp%d_example.mat',roi)),'parameters');
stream=RandStream('mt19937ar','Seed',101);
initial=randn(stream,roi,1);
trajectory=simulate_hodge_parameters(saved.parameters,initial,300,73001,stochastic);
fprintf('FITTED_MODEL_DEMO_PASS: %d ROI, %d simulated steps, stochastic=%d.\n',roi,size(trajectory,2)-1,stochastic);
end
