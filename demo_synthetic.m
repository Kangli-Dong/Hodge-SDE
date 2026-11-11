function result = demo_synthetic(show_figure)
%DEMO_SYNTHETIC Offline metric contract demo, not a fit or paper reproduction.
if nargin<1; show_figure=false; end
here=fileparts(mfilename('fullpath'));
saved_path=path; saved_rng=rng;
cleanup=onCleanup(@()restore_state(saved_path,saved_rng)); %#ok<NASGU>
addpath(fullfile(here,'src')); rng(241,'twister'); started=tic;
n=4; T=96; X=zeros(n,T); G=X;
A=[.6 .08 0 0; -.05 .7 .1 0; 0 -.08 .5 .1; 0 0 -.1 .65];
for t=2:T
    X(:,t)=A*X(:,t-1)+.3*randn(n,1);
    G(:,t)=A*G(:,t-1)+.3*randn(n,1);
end
options=struct('strict_generation',true,'fcd_window_seconds',8, ...
    'fcd_step_seconds',8,'sw2_projections',16);
identity=struct('Profile','synthetic','Subject','synthetic_example','Model','self');
self=score_free_run_by_fold({X},{X},1,options,identity);
names={'sw2_distance','fcd_ks','static_fc_error', ...
    'irreversibility_abs_error','mle_abs_error'};
assert(all(abs(self{1,names})<1e-10),'Self-comparison must give five zero errors.');
identity.Model='independent_synthetic_rollout';
generated=score_free_run_by_fold({X},{G},1,options,identity);
assert(all(isfinite(generated{1,names})));
constant=score_free_run_by_fold({X},{ones(n,T)},1,options,identity);
assert(isfinite(constant.sw2_distance) && isnan(constant.static_fc_error) ...
    && isnan(constant.fcd_ks) && isnan(constant.mle_abs_error));
gap=G; gap(:,48)=NaN;
invalid=score_free_run_by_fold({X},{gap},1,options,identity);
assert(invalid.ExecutionStatus=="failed" && invalid.NValidPoints==0);
result=[self(:,['Model' names]); generated(:,['Model' names])];
disp(result);
fprintf('CANDIDATE_DEMO_PASS: self=0; finite rollout; constant undefined correlations; gap rejected.\n');
fprintf('Compute time %.3f seconds (excludes MATLAB startup).\n',toc(started));
if show_figure
    figure('Color','w','Name','Synthetic metric demonstration');
    plot(0:T-1,X(1,:),'LineWidth',1); hold on;
    plot(0:T-1,G(1,:),'LineWidth',1);
    xlabel('Synthetic sample (TR = 1 s)'); ylabel('State, ROI 1');
    legend('Synthetic reference','Independent synthetic rollout');
    title('Synthetic example only; not a paper figure');
end
end

function restore_state(saved_path,saved_rng)
path(saved_path); rng(saved_rng);
end
