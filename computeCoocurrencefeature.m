function [features, empmean, CRFDATA] = computeCoocurrencefeature(features, empmean, imlist, CRFDATA, train)
% Cocurrence feature: Connecting 2nd level and 3rd level

r = length(features) + 1;
features{r} = [];
featparams = CRFDATA.featparams;
if featparams.segFeature == 1
   varname1 = 'SS';
else
   varname1 = 'SP'; 
end;
varname2 = 'Z';
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
NumStates = CRFDATA.NumClasses;
BinaryStates = 2;


% Generate Potentials
T_P = cell(NumStates, 1);
T_P_reshape = zeros(NumStates*BinaryStates, NumStates);
for i = 1 : NumStates
   T_P_i = zeros(NumStates,BinaryStates);
   T_P_i(i, 1) = -1000;
   T_P{i} = T_P_i;
   T_P_reshape(:, i) = reshape(T_P_i,NumStates*BinaryStates,1);
end;

for i = 1:length(imlist)
    [~, blockstart1, blockend1] = getvardata(vars, varname1, i);
    [~, blockstart2, ~] = getvardata(vars, varname2, i);
    
    n = blockend1 - blockstart1 + 1;
    Pair = zeros(n*NumStates, 2);
    
    TopLevel = (1 : NumStates)';
    for j = 1:n
        LowerLevel = j*ones(NumStates,1);
        Pair((j-1)*NumStates + 1:j*NumStates, :) = [LowerLevel TopLevel];   
    end
    
    % Compute Emperical Means
    if train
       % groundtruth label of supersegment will always match the Zi, so
       % empirical mean is 0
    end;

    % unary setting
    
    % supersegments
    local = cell(1, LOC(i));
    pntr = blockstart1;
   % if numel(Pair)
        for s = 1 : n
            local{pntr}.NumStates = NumStates;
            local{pntr}.pot = zeros(NumStates, 1);
            connFac = [];
            if numel(Pair)
               [connFac, ~] = find(Pair(:, 1) ==s);
            end;
            local{pntr}.connTo = connFac+M(i);
            pntr = pntr + 1;
        end;

        pntr = blockstart2;
        for s = 1 : NumStates
            local{pntr}.NumStates = BinaryStates;
            local{pntr}.pot = zeros(BinaryStates, 1);
            connFac = [];
            if numel(Pair)
               [connFac, ~] = find(Pair(:, 2) ==s);
            end;
            local{pntr}.connTo = connFac+M(i);
            pntr = pntr + 1;
        end;
   % end;
   features{r}.sample{i}.local = local;
    
    % pairwise setting
    factor = cell(1, M(i) + size(Pair, 1));
    for p = 1:size(Pair,1)
        j = mod(p - 1, NumStates) + 1;
        factor{p+M(i)}.pot = T_P_reshape(:, j);
        factor{p+M(i)}.size = [NumStates BinaryStates];    
    end
    features{r}.sample{i}.factor = factor;
    
    M(i) = M(i)+size(Pair,1);
end

CRFDATA = updateData(CRFDATA, vars, LOC, M);