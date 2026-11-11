function status = audit_segment_status(Y, minPoints, strictGeneration)
%AUDIT_SEGMENT_STATUS Label valid, invalid, and collapsed score segments.
% A missing sample ends a segment; no statistic may bridge it.
if nargin < 2; minPoints = 12; end
if nargin < 3; strictGeneration = false; end
status = struct('label',"invalid",'reason',"",'n_points',0,'n_segments',0, ...
    'has_undefined_correlation',true,'variance_min',NaN, ...
    'is_complete_finite',false,'is_single_contiguous',false, ...
    'is_collapsed',false,'variance_tolerance',NaN);
if ~isnumeric(Y) || ndims(Y) ~= 2 || isempty(Y)
    status.reason = "not a nonempty 2-D numeric trajectory"; return
end
valid = all(isfinite(Y), 1);
status.is_complete_finite = all(valid);
edge = diff([false valid false]); starts = find(edge == 1); stops = find(edge == -1) - 1;
lengths = stops - starts + 1; keep = lengths >= minPoints;
starts = starts(keep); stops = stops(keep);
status.n_segments = numel(starts); status.n_points = sum(stops - starts + 1);
if isempty(starts)
    status.reason = "no finite contiguous segment reaches minPoints"; return
end
V = [];
for i = 1:numel(starts); V = [V var(Y(:,starts(i):stops(i)),0,2)]; end %#ok<AGROW>
status.variance_min = min(V,[],'all');
scale = max(1, max(abs(Y(:,valid)), [], 'all'));
status.variance_tolerance = (100 * eps(scale))^2;
status.is_single_contiguous = status.n_segments == 1 && status.is_complete_finite;
if strictGeneration && ~status.is_single_contiguous
    status.reason = "strict_generation requires one complete finite trajectory; gaps are not scoreable"; return
end
if strictGeneration
    collapse_tolerance = status.variance_tolerance;
else
    collapse_tolerance = eps;
end
if any(~isfinite(V),'all') || any(V <= collapse_tolerance,'all')
    status.label = "collapsed"; status.is_collapsed = true;
    status.reason = "at least one ROI has undefined correlation (zero variance)"; return
end
status.label = "valid"; status.reason = "finite contiguous segments with defined ROI variance";
status.has_undefined_correlation = false;
end
