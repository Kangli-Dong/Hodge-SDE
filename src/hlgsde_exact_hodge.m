function [Yg, Yc, Yh, phi, psi, meta] = hlgsde_exact_hodge(Yvec, B1, B2, tol, maxit)
%HLGSDE_EXACT_HODGE Validated sparse orthogonal projections in edge space.
% B1: node-by-edge incidence (-1 at tail, +1 at head).
% B2: edge-by-triangle boundary; use sparse(numel(Yvec),0) without faces.
% Optional tol/maxit control direct B2 LSQR; invalid solves throw, not cache.
if nargin < 4 || isempty(tol); tol = 1e-12; end
if nargin < 5 || isempty(maxit); maxit = 10000; end
if ~isnumeric(Yvec) || ~isreal(Yvec) || (~isvector(Yvec) && ~isempty(Yvec)) || ...
        ~isnumeric(B1) || ~isreal(B1) || ~ismatrix(B1) || ...
        ~isnumeric(B2) || ~isreal(B2) || ~ismatrix(B2) || ...
        size(B1, 2) ~= numel(Yvec) || size(B2, 1) ~= numel(Yvec) || ...
        any(~isfinite(Yvec(:))) || any(~isfinite(nonzeros(B1))) || ...
        any(~isfinite(nonzeros(B2))) || ~isscalar(tol) || ~isfinite(tol) || ...
        tol <= eps || tol >= 1 || ~isscalar(maxit) || ~isfinite(maxit) || ...
        maxit < 1 || maxit ~= floor(maxit)
    error('HLGSDE:ExactHodgeInput', 'Invalid flow, boundary dimensions, or solver controls.');
end
B1 = sparse(double(B1)); B2 = sparse(double(B2));
Yvec = full(double(Yvec(:)));
if any(abs(nonzeros(B1)) ~= 1) || any(full(sum(spones(B1), 1)) ~= 2) || ...
        any(full(sum(B1, 1)) ~= 0) || nnz(B1 * B2) ~= 0
    error('HLGSDE:ExactHodgeInput', 'Expected graph incidence and B1*B2=0.');
end
flow_norm = norm(Yvec);
if ~isfinite(flow_norm)
    error('HLGSDE:ExactHodgeInvalid', 'Flow norm is not finite.');
end
% Solve at unit flow scale; diagnostic ratios stay meaningful for tiny flows.
scale = flow_norm;
if scale == 0; scale = 1; end
y = Yvec / scale;
n = size(B1, 1);
lap = B1 * B1';
adjacency = lap;
adjacency(1:n+1:end) = 0;
adjacency = abs(adjacency);
labels = zeros(n, 1);
if n > 0; labels = conncomp(graph(adjacency, 'upper'))'; end
[~, pins] = unique(labels, 'stable');
keep = true(n, 1); keep(pins) = false;
p = zeros(n, 1);
gradient = struct('method', 'gauge_fixed_sparse_laplacian', 'flag', 0, ...
    'iterations', 0, 'status', 'zero_flow', 'relative_residual', 0);
if any(keep) && flow_norm > 0
    rhs = B1 * y;
    p(keep) = lap(keep, keep) \ rhs(keep);
    % Match the zero-mean potential convention on each connected component.
    means = accumarray(labels, p) ./ accumarray(labels, 1);
    p = p - means(labels);
    gradient.status = 'solved';
    gradient.relative_residual = norm(lap * p - rhs);
end
g = full(B1' * p);
z = zeros(size(B2, 2), 1);
curl = struct('method', 'direct_B2_lsqr', 'tolerance', tol, ...
    'max_iterations', maxit, 'flag', 0, 'iterations', 0, ...
    'status', 'no_triangles', 'relative_residual', 0, 'normal_residual', 0);
curl.zero_projection_normal_tolerance = min(1e-12, 10 * eps * max(norm(B2, 'fro'), 1));
if size(B2, 2) > 0 && flow_norm > 0
    % Never form B2'*B2: it squares conditioning and may be enormous.
    % An orthogonal RHS already has zero projection; avoid LSQR breakdown.
    if norm(B2' * y) <= curl.zero_projection_normal_tolerance
        curl.status = 'zero_projection';
        curl.relative_residual = 1;
    elseif size(B2, 2) == 1
        % One face is a scalar projection, with no iterative breakdown.
        z = full(B2' * y) / sum(nonzeros(B2).^2);
        curl.method = 'single_face_direct_projection';
        curl.status = 'solved';
        curl.relative_residual = norm(y - full(B2 * z));
    else
        [z, curl.flag, curl.relative_residual, curl.iterations] = lsqr(B2, y, tol, maxit);
        curl.status = 'converged';
    end
elseif size(B2, 2) > 0
    curl.status = 'zero_flow';
end
c = full(B2 * z);
h = y - g - c;
validation_tolerance = 1e-8;
component_relative_floor = 1e-10;
norms_before = [norm(g), norm(c), norm(h)];
zeroed = norms_before <= component_relative_floor;
% A roundoff-only component must not become a unit channel downstream.
if zeroed(1); g(:) = 0; p(:) = 0; end
if zeroed(2); c(:) = 0; z(:) = 0; end
if zeroed(3); h(:) = 0; end
parts = [g c h];
norms = [norm(g), norm(c), norm(h)];
diagnostics.reconstruction = norm(y - g - c - h);
diagnostics.gradient_membership = norm(g - B1' * p);
diagnostics.curl_membership = norm(c - B2 * z);
diagnostics.gradient_stationarity = norm(B1 * (y - g));
diagnostics.curl_stationarity = norm(B2' * (y - c));
diagnostics.harmonic_divergence = norm(B1 * h);
diagnostics.harmonic_curl = norm(B2' * h);
curl.normal_residual = diagnostics.curl_stationarity;
products = abs(parts' * parts);
products(1:4:9) = 0;
diagnostics.orthogonality = max([0; products(:)]);
applicable = (norms' > component_relative_floor) & (norms > component_relative_floor);
applicable(1:4:9) = false;
relative = nan(3);
denominators = norms' * norms;
relative(applicable) = products(applicable) ./ denominators(applicable);
diagnostics.component_orthogonality = relative;
diagnostics.component_orthogonality_applicable = applicable;
diagnostics.max_component_orthogonality = max([0; relative(applicable)]);
gates = [diagnostics.reconstruction, diagnostics.gradient_membership, ...
    diagnostics.curl_membership, diagnostics.gradient_stationarity, ...
    diagnostics.curl_stationarity, diagnostics.harmonic_divergence, ...
    diagnostics.harmonic_curl, diagnostics.orthogonality, ...
    diagnostics.max_component_orthogonality];
Yg = g * scale; Yc = c * scale; Yh = h * scale;
phi = p * scale; psi = z * scale;
diagnostics.pass = all(isfinite(gates)) && all(gates <= validation_tolerance) && ...
    curl.flag == 0 && isfinite(gradient.relative_residual) && ...
    all(isfinite([Yg; Yc; Yh; phi; psi]));
if ~diagnostics.pass
    maximum_gate = max(gates);
    if any(~isfinite(gates)); maximum_gate = NaN; end
    error('HLGSDE:ExactHodgeInvalid', ...
        ['Exact Hodge validation failed: LSQR flag=%d, iter=%d/%d, max gate=%.3g ' ...
        '(limit %.3g), initial normal=%.3g, curl norm=%.3g.'], ...
        full(curl.flag), full(curl.iterations), maxit, full(maximum_gate), ...
        validation_tolerance, norm(B2' * y), norm(c));
end
meta = struct('version', 'exact_hodge_v1', ...
    'algorithm', 'gauge_fixed_laplacian_direct_B2_lsqr', ...
    'computed', true, 'status', 'validated', ...
    'validation_tolerance', validation_tolerance, ...
    'component_relative_floor', component_relative_floor, 'flow_norm', flow_norm, ...
    'connected_components', numel(pins), 'gauge_nodes', pins, ...
    'component_norms_before_zeroing', norms_before * flow_norm, ...
    'zeroed_components', zeroed, 'solver', struct('gradient', gradient, 'curl', curl), ...
    'diagnostics', diagnostics);
end
