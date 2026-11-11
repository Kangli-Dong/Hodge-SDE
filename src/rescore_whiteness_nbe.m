function [candidate, foldRows] = rescore_whiteness_nbe(source, mapping, foldDir, profile, loader)
%RESCORE_WHITENESS_NBE Reaggregate cached NBE Q and thresholds over five folds.
% SOURCE and MAPPING are caller-supplied tables; FOLDDIR contains per-fold
% fitted-result MAT files. The NBE whiteness implementation and cache loader
% are external dependencies. This function does not train or run permutations.
if nargin < 5; loader = @load_incremental_fold; end
profile = string(profile);
assert(ismember(profile,["hcp150","hcp250","hcp450"]),'Unknown profile.');
atlas = str2double(extractAfter(profile,"hcp"));
required = {'PlotKey','Atlas','Model','ModelIndex','FoldCount','Q_over_Qthr'};
assert(all(ismember(required,source.Properties.VariableNames)),'Incomplete source table.');
assert(all(ismember({'InternalPlotKey','Subject','PassedAllChecks'},mapping.Properties.VariableNames)), ...
    'Incomplete subject mapping.');
candidate = source(source.Atlas==atlas,:);
candidate.PlotKey = string(candidate.PlotKey);
mapping.InternalPlotKey = string(mapping.InternalPlotKey);
mapping.Subject = string(mapping.Subject);
assert(~isempty(candidate) && all(candidate.FoldCount==5),'Expected five-fold source rows.');
assert(height(unique(candidate(:,{'PlotKey','ModelIndex'})))==height(candidate), ...
    'Duplicate source keys.');
assert(numel(unique(mapping.InternalPlotKey))==height(mapping),'Duplicate subject mapping.');
keys = unique(candidate.PlotKey,'stable');
[found, loc] = ismember(keys,mapping.InternalPlotKey);
assert(all(found) && all(mapping.PassedAllChecks(loc)),'Subject mapping/QC mismatch.');
indices = [1 2 3 6 9 10 12 13 17];
candidate.Old_Q_over_Qthr = candidate.Q_over_Qthr;
candidate.Q_over_Qthr(:) = NaN;
candidate.Mean_Q = nan(height(candidate),1);
candidate.Mean_Qthr = nan(height(candidate),1);
foldRows = table();
for s = 1:numel(keys)
    rows = find(candidate.PlotKey==keys(s));
    assert(numel(rows)==9 && isequal(sort(candidate.ModelIndex(rows))',indices), ...
        'Nine-model source set mismatch.');
    modelIndices = candidate.ModelIndex(rows);
    Q = nan(5,9); threshold = nan(5,9); legacyRatio = nan(5,9);
    for fold = 1:5
        filename = fullfile(foldDir,sprintf('%s_%d.mat',mapping.Subject(loc(s)),fold));
        cache = loader(filename,'whiteness');
        for m = 1:9
            value = cache.whiteness(modelIndices(m));
            assert(isscalar(value.stat) && isscalar(value.sig_thr),'Invalid whiteness scalar.');
            Q(fold,m) = value.stat;
            threshold(fold,m) = value.sig_thr;
            legacyRatio(fold,m) = value.stat ./ value.sig_thr;
        end
    end
    assert(all(isfinite(Q(:))) && all(isfinite(threshold(:))) && all(threshold(:)>0), ...
        'Incomplete or invalid five-fold whiteness metrics.');
    old = mean(legacyRatio,1)';
    tolerance = 1e-10 + 1e-8*abs(candidate.Old_Q_over_Qthr(rows));
    assert(all(abs(old-candidate.Old_Q_over_Qthr(rows))<=tolerance), ...
        'Cached results differ from source; refusing mixed-version scoring.');
    meanQ = mean(Q,1)'; meanThreshold = mean(threshold,1)';
    candidate.Q_over_Qthr(rows) = meanQ./meanThreshold;
    candidate.Mean_Q(rows) = meanQ;
    candidate.Mean_Qthr(rows) = meanThreshold;
    block = table(repmat(keys(s),45,1),repmat(atlas,45,1), ...
        repelem(string(candidate.Model(rows)),5),repelem(modelIndices,5), ...
        repmat((1:5)',9,1),Q(:),threshold(:), ...
        'VariableNames',{'PlotKey','Atlas','Model','ModelIndex','Fold','Q','Qthr'});
    foldRows = [foldRows; block]; %#ok<AGROW>
end
end
