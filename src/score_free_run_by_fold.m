function [foldRows, subjectRows] = score_free_run_by_fold(YTrue, YGenerated, TR, options, identity)
%SCORE_FREE_RUN_BY_FOLD Score subject x fold x repeat independently.
% YTrue/YGenerated are cell arrays (fold,repeat); each item is ROI x time.
% Legacy stitched values are deliberately not computed here.
if nargin < 4; options = struct(); end
if nargin < 5; identity = struct(); end
required = {'Profile','Subject','Model'};
for i=1:numel(required)
    if ~isfield(identity,required{i}); identity.(required{i}) = "unknown"; end
end
assert(isequal(size(YTrue),size(YGenerated)), 'Fold/repeat cell layouts must match.');
metrics = {'sw2_distance','fcd_ks','static_fc_error','irreversibility_abs_error','mle_abs_error'};
strict = getfield_default(options,'strict_generation',false);
if strict; protocol = "fair_v2"; else; protocol = "legacy"; end
support_names = strcat(metrics,'_Supported');
reason_names = strcat(metrics,'_Reason');
foldRows = table();
for fold=1:size(YTrue,1)
    for repeat=1:size(YTrue,2)
        a = audit_segment_status(YTrue{fold,repeat}, getfield_default(options,'min_segment_points',12),strict);
        b = audit_segment_status(YGenerated{fold,repeat}, getfield_default(options,'min_segment_points',12),strict);
        row = table(string(identity.Profile),string(identity.Subject),string(identity.Model),fold,repeat, ...
            protocol,"completed",a.label,b.label,"undefined",a.reason+" | "+b.reason, ...
            a.is_collapsed,b.is_collapsed,false,a.has_undefined_correlation || b.has_undefined_correlation,false,0,0, ...
            'VariableNames',{'Profile','Subject','Model','Fold','Repeat','Protocol','ExecutionStatus','ReferenceStatus','GeneratedStatus','ScoreStatus','Reason','ReferenceCollapsed','GeneratedCollapsed','FCDCollapsed','CorrelationUndefined','MLERepeatedStates','NValidPoints','NFCDWindows'});
        for k=1:numel(metrics)
            row.(metrics{k}) = NaN;
            row.(support_names{k}) = false;
            row.(reason_names{k}) = "not evaluated";
        end
        allowed = a.label == "valid" && b.label == "valid";
        if strict
            allowed = a.label ~= "invalid" && b.label ~= "invalid" && ...
                isequal(size(YTrue{fold,repeat}),size(YGenerated{fold,repeat})) && ...
                a.is_single_contiguous && b.is_single_contiguous;
        end
        if allowed
            try
                [m,d] = compute_dynamics_metrics(YTrue{fold,repeat},{YGenerated{fold,repeat}},TR,options);
                for k=1:numel(metrics)
                    row.(metrics{k}) = m.(metrics{k})(1);
                    row.(support_names{k}) = d(1).metric_support.(metrics{k});
                    row.(reason_names{k}) = d(1).metric_reason.(metrics{k});
                end
                row.FCDCollapsed = d(1).fcd_collapsed;
                row.CorrelationUndefined = d(1).correlation_undefined;
                row.MLERepeatedStates = d(1).mle_repeated_states;
                row.NValidPoints = m.n_valid_points(1);
                row.NFCDWindows = m.n_fcd_windows(1);
                if d(1).strict_rejected || d(1).status == "error"
                    row.ExecutionStatus = "failed"; row.ScoreStatus = "error"; row.Reason = d(1).reason;
                elseif ~strict && any(row{1,support_names})
                    row.ScoreStatus = "valid";
                elseif ~strict
                    row.ScoreStatus = "undefined"; row.Reason = "no metric has support; retained as NaN";
                elseif all(row{1,support_names})
                    row.ScoreStatus = "valid";
                elseif any(row{1,support_names})
                    row.ScoreStatus = "partial";
                else
                    row.ScoreStatus = "undefined"; row.Reason = "no metric has support; retained as NaN";
                end
            catch ME
                row.ExecutionStatus = "failed"; row.ScoreStatus = "error";
                row.Reason = string(ME.identifier)+": "+string(ME.message);
            end
        else
            if strict
                row.ExecutionStatus = "failed";
                row.ScoreStatus = "error";
                row.Reason = "strict trajectory contract failed: "+a.reason+" | "+b.reason;
            else
                row.ScoreStatus = "undefined";
                row.Reason = "not scoreable: "+a.reason+" | "+b.reason;
            end
        end
        foldRows = [foldRows; row]; %#ok<AGROW>
    end
end
subjectRows = table();
for metric = string(metrics)
    ok = foldRows.([char(metric) '_Supported']); total=height(foldRows); collapsed=nnz(foldRows.GeneratedCollapsed | foldRows.ReferenceCollapsed);
    subjectRows = [subjectRows; table(string(identity.Profile),string(identity.Subject),string(identity.Model),metric,total,nnz(ok),total-nnz(ok),collapsed,collapsed/total,mean(foldRows.(metric)(ok),'omitnan'),std(foldRows.(metric)(ok),'omitnan'), ...
        'VariableNames',{'Profile','Subject','Model','Metric','NTotalFoldRepeat','NValid','NInvalid','NCollapsed','CollapseRate','Mean','SD'})]; %#ok<AGROW>
end
end

function value = getfield_default(s,name,default)
if isfield(s,name) && ~isempty(s.(name)); value=s.(name); else; value=default; end
end
