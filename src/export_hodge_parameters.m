function parameters = export_hodge_parameters(model, filename)
%EXPORT_HODGE_PARAMETERS Save only the fitted Hodge-SDE numerical parameters.
% MODEL is an already loaded, authorized fitted model, not a fold-result file.
% Excludes all observations, predictions, fit diagnostics, IDs and source paths.
% FILENAME is optional. Existing output files are never overwritten.
assert(isstruct(model) && isscalar(model),'Expected one fitted model.');
assert(strcmp(model.topology_mode,'hodge'),'Expected a four-channel Hodge model.');
assert(strcmp(model.hodge_source,'nonnegative_gc_score'),'Expected the current nonnegative-GC model.');
fields = {'alpha','b','D','channel_gain','component_scales','component_matrices', ...
    'W_residual','residual_projection','TR','DeltaT','generation_sigma', ...
    'generation_noise_factor','generation_use_core_drift'};
assert(all(isfield(model,fields)),'Incomplete fitted parameter fields.');
parameters = struct('format','hodge_sde_parameters_v1');
for k = 1:numel(fields)
    value = model.(fields{k});
    if strcmp(fields{k},'component_matrices')
        assert(iscell(value) && numel(value)==4,'Expected four channel matrices.');
        assert(all(cellfun(@(x) isnumeric(x) && isreal(x) && all(isfinite(x(:))),value)));
    else
        assert((isnumeric(value)||islogical(value)) && isreal(value) && all(isfinite(value(:))));
    end
    parameters.(fields{k}) = value;
end
n = numel(parameters.D);
assert(isequal(size(parameters.D),[n 1]) && isequal(size(parameters.channel_gain),[n 4]));
assert(all(cellfun(@(x) isequal(size(x),[n n]),parameters.component_matrices)));
assert(numel(parameters.component_scales)==4 && all(parameters.component_scales>0));
assert(isscalar(parameters.alpha) && isscalar(parameters.b));
assert(isscalar(parameters.DeltaT) && parameters.DeltaT>0 && parameters.TR==parameters.DeltaT);
assert(isequal(size(parameters.W_residual),[n n]) && isequal(size(parameters.residual_projection),[n 5]));
assert(isequal(size(parameters.generation_sigma),[n 1]) && all(parameters.generation_sigma>=0));
assert(size(parameters.generation_noise_factor,1)==n);
assert(isscalar(parameters.generation_use_core_drift));
if nargin > 1
    assert(exist(filename,'file')~=2,'Refusing to overwrite an existing model.');
    save(filename,'parameters','-v7');
end
end
