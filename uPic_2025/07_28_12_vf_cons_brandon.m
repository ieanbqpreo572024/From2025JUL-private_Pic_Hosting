function Y = vf_cons_brandon(params, K_grid, V_guess, Y_sch)
% VF_cons_brandon
%
% Optimized version of VF_cons.m with:
% - Vectorized expected value calculations
% - Vectorized utility computations  
% - Pre-computed consumption matrices
% - Reduced memory allocations
%
% Solves dynamic programming problem: 
% V(AS,k) = max_{k'} { u(c) + beta * E[V(AS',k')] }
% subject to: c = (1-delta)*k + Y(AS,k) - k'
%
% INPUTS/OUTPUTS: Same as VF_cons.m

% Extract grid dimensions
nAS = length(params.Grid_AS);  % Number of aggregate states
nk = size(K_grid,2);           % Number of capital grid points

% Extract parameters for efficiency
pSigma = params.Sigma;       % Risk aversion parameter
pBeta = params.Beta;         % Discount factor
pEpsilonV = params.EpsilonV; % Convergence tolerance
niter_V = params.niter_V;    % Maximum iterations
P_AS = params.P_AS;          % Transition matrix for aggregate states
delta = params.delta;        % Capital depreciation rate

% Initialize value function iteration
V = V_guess;                    % Current value function
V_new = zeros([nAS, nk]);       % Updated value function
h_k = zeros([nAS,nk]);          % Policy function (capital choice indices)

% Pre-compute consumption matrix for all (current_k, next_k) combinations
% This avoids repeated calculations in the main loop
K_current = repmat(K_grid', 1, nk);      % nk x nk matrix of current capital
K_next = repmat(K_grid, nk, 1);          % nk x nk matrix of next capital
Cons_base = (1-delta) * K_current - K_next; % Base consumption matrix

% Vectorized utility function (handles entire matrices)
if pSigma ~= 1.0
    utility_vec = @(c) (max(c, 1e-12).^(1-pSigma) - 1) / (1-pSigma);
else
    utility_vec = @(c) log(max(c, 1e-12));
end

% Initialize iteration counters and convergence measure
iter_V = 1;
gap = 1;

% ==================================================================
% MAIN VALUE FUNCTION ITERATION LOOP - OPTIMIZED
% ==================================================================
while(gap > pEpsilonV && iter_V < niter_V)
    fprintf('It: %d    Policy: %f   \n',iter_V, gap);
    
    % Update current value function
    V = V_new;
    
    % OPTIMIZATION 1: Vectorized expected value computation
    % Compute E[V(AS',k')] = P_AS * V for all future states at once
    EV = P_AS * V;  % nAS x nk matrix of expected continuation values
    
    % Loop over current aggregate states
    for AS = 1:nAS
        % OPTIMIZATION 2: Vectorized consumption and utility computation
        % Compute consumption matrix for current state AS
        Cons_matrix = Cons_base + repmat(Y_sch(AS, :)', 1, nk);
        
        % Ensure all consumption values are positive
        Cons_matrix = max(Cons_matrix, 1e-12);
        
        % Compute utility matrix for all (current_k, next_k) pairs
        Util_matrix = utility_vec(Cons_matrix);
        
        % OPTIMIZATION 3: Vectorized value function update
        % For each current capital level, find optimal next capital
        EV_expanded = repmat(EV(AS, :), nk, 1);  % Expand EV to match dimensions
        Objective = Util_matrix + pBeta * EV_expanded;
        
        % Find optimal policy and value for each current capital level
        [V_new(AS, :), h_k(AS, :)] = max(Objective, [], 2);
    end

    % Check convergence: relative change in value function
    gap = max(abs(V_new - V), [], "all") / norm(reshape(V, 1, nAS*nk), Inf);

    % ==============================================================
    % HOWARD IMPROVEMENT STEPS - OPTIMIZED
    % ==============================================================
    iter_Howard = 1;
    while(iter_Howard < 20)
        V = V_new;
        
        % OPTIMIZATION 4: Vectorized Howard improvement
        for AS = 1:nAS
            % Extract optimal capital choices for current state
            k_choices = h_k(AS, :);  % Optimal next-period capital indices
            
            % Vectorized computation of consumption given policy
            cons_policy = (1-delta) * K_grid + Y_sch(AS, :) - K_grid(k_choices);
            cons_policy = max(cons_policy, 1e-12);
            
            % Vectorized utility computation
            util_policy = utility_vec(cons_policy);
            
            % Vectorized expected value computation using current policy
            % For each current k, get E[V(AS', k')] where k' = h_k(AS,k)
            EV_policy = zeros(1, nk);
            for ASp = 1:nAS
                % Use linear indexing for efficiency
                idx = sub2ind([nAS, nk], repmat(ASp, 1, nk), k_choices);
                EV_policy = EV_policy + P_AS(AS, ASp) * V(idx);
            end
            
            % Update value function with current policy
            V_new(AS, :) = util_policy + pBeta * EV_policy;
        end
        iter_Howard = iter_Howard + 1;
    end
    % End Howard improvement
    
    % Increment main iteration counter
    iter_V = iter_V + 1;
end

% ==================================================================
% RETURN RESULTS
% ==================================================================
Y = {V_new, h_k};  % Return converged value function and policy function
end