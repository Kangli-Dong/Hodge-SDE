function test_whiteness_synthetic
% In-memory cache-loader contract; no third-party source or subject data.
addpath(fullfile(fileparts(mfilename('fullpath')),'src'));
idx = [1 2 3 6 9 10 12 13 17]';
source = table(repmat("synthetic",9,1),repmat(150,9,1), ...
    "model"+string((1:9)'),idx,repmat(5,9,1),repmat(2.2,9,1), ...
    'VariableNames',{'PlotKey','Atlas','Model','ModelIndex','FoldCount','Q_over_Qthr'});
mapping = table("synthetic","synthetic_subject",true, ...
    'VariableNames',{'InternalPlotKey','Subject','PassedAllChecks'});
[candidate, folds] = rescore_whiteness_nbe(source,mapping,'unused','hcp150',@synthetic_loader);
assert(height(candidate)==9 && height(folds)==45);
assert(all(abs(candidate.Q_over_Qthr-5/3)<1e-12));
assert(all(abs(candidate.Mean_Q-3)<1e-12));
assert(all(abs(candidate.Mean_Qthr-1.8)<1e-12));
bad = source; bad.Q_over_Qthr(:) = 0;
rejected = false;
try; rescore_whiteness_nbe(bad,mapping,'unused','hcp150',@synthetic_loader);
catch; rejected = true; end
assert(rejected,'Mismatched cache/source must be rejected.');
fprintf('NBE_WHITENESS_SYNTHETIC_PASS: five-fold ratio of means and source guard.\n');
end

function cache = synthetic_loader(filename,~)
token = regexp(filename,'_(\d+)\.mat$','tokens','once');
fold = str2double(token{1});
values = repmat(struct('stat',fold,'sig_thr',1+(fold==5)*4),1,17);
cache = struct('whiteness',values);
end
