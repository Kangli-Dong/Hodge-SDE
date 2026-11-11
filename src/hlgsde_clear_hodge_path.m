function hlgsde_clear_hodge_path(hodge_dir)
%HLGSDE_CLEAR_HODGE_PATH Remove hodge-main paths that shadow MATLAB tools.
%
% hodge-main contains legacy cell2mat.m and mat2vec.m helpers. They are
% required only while Hodge_decompose is executing and must not remain on
% MATLAB's global search path between NBE baseline models or CV folds.

if nargin < 1 || isempty(hodge_dir)
    hodge_dir = getenv('HLGSDE_HODGE_DIR');
end
if isempty(hodge_dir)
    this_dir = fileparts(mfilename('fullpath'));
    hodge_dir = fullfile(fileparts(this_dir), 'external', 'hodge-main');
end
if exist(hodge_dir, 'dir')
    hodge_paths = strsplit(genpath(hodge_dir), pathsep);
    active_paths = strsplit(path, pathsep);
    paths_to_remove = intersect(hodge_paths, active_paths, 'stable');
    if ~isempty(paths_to_remove)
        rmpath(strjoin(paths_to_remove, pathsep));
    end
end
% MATLAB can keep a resolved same-name function in its cache after rmpath.
% Clear exactly the conflicting Hodge helpers so the next baseline resolves
% MATLAB's own cell2mat rather than hodge-main/cell2mat.m.
clear cell2mat mat2vec Hodge_2Skeleton PH_boundary Hodge_vec Hodge_decompose
rehash
end
