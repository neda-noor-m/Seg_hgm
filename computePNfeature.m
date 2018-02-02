function [features, empmean, CRFDATA] = computePNfeature(features, empmean, imlist, CRFDATA, gamma, ALLPATHS, train)
% PN feature: connecting the 1st and 2nd level

r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
varname1 = 'SP';
varname2 = 'SS';
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
NumStates = CRFDATA.NumClasses;
T_T = gamma*(eye(NumStates) - ones(NumStates));
Pot = reshape(T_T,NumStates*NumStates,1);

for i = 1:length(imlist)
    imname = imlist{i};    
    Pair_Pn = [];
    if exist(fullfile(ALLPATHS.PN_PATH,[imname '_Pn.mat']),'file')
        PNdata = load(fullfile(ALLPATHS.PN_PATH,[imname '_Pn.mat']));
        Conn = PNdata.Conn; 
    else
        if ~exist(ALLPATHS.PN_PATH,'dir')
            mkdir(ALLPATHS.PN_PATH);
        end
        Conn = computePNEdge(imname);
    end
    
    [~, blockstart1, blockend1] = getvardata(vars, varname1, i);
    [~, blockstart2, blockend2] = getvardata(vars, varname2, i);
    
    for j = 1:length(Conn)   % connection matrix [superpixel_id, superedge_id]
        for k = 1:length(Conn{j})
            Pair_Pn = [Pair_Pn; [Conn{j}(k), j]];
        end
    end          
    
    Pair = Pair_Pn; % index of SP-SS
    if train
       % get the labels for each superpixel and supersegment and remove void
       labeledPair = zeros(size(Pair));
       labelSP = CRFDATA.labelsSPfull{i};
       labelSS = CRFDATA.labelsSSfull{i};
       labeledPair(:, 1) = labelSP(Pair(:, 1), 1);
       labeledPair(:, 2) = labelSS(Pair(:, 2), 1);
       goodpairs = find(min(labeledPair, [], 2) > 0);
       labeledPair = labeledPair(goodpairs, :);
       Pair = Pair(goodpairs, :);
       
       % move the indices
       row1 = CRFDATA.indsSP{i};
       mapping1 = zeros(size(labelSP, 1), 1);
       mapping1(row1) = (1:length(row1))';
       Pair(:, 1) = mapping1(Pair(:, 1));
       
       row2 = CRFDATA.indsSS{i};
       mapping2 = zeros(size(labelSS, 1), 1);
       mapping2(row2) = (1:length(row2))';
       Pair(:, 2) = mapping2(Pair(:, 2));
       
    % Compute Emperical Means
        if size(labeledPair,2)==1
            labeledPair = labeledPair';
        end
        [NR, ~] = find(labeledPair(:,1) - labeledPair(:,2));
        empmean(r) = empmean(r) + length(NR) * (-gamma);
    end;

    % unary setting
    %-----------------
    % superpixels
    local = cell(1, LOC(i));
    pntr = blockstart1;
    if numel(Pair)
        for s = 1 : blockend1 - blockstart1 + 1
            local{pntr}.NumStates = NumStates;
            local{pntr}.pot = zeros(NumStates, 1);
            [connFac, ~] = find(Pair(:, 1) ==s);
            local{pntr}.connTo = connFac + M(i);
            pntr = pntr + 1;
        end;

        %-----------------
        % supersegments
        pntr = blockstart2;
        for s = 1 : blockend2 - blockstart2 + 1
            local{pntr}.NumStates = NumStates;
            local{pntr}.pot = zeros(NumStates, 1);
            [connFac, ~] = find(Pair(:, 2) == s);
            local{pntr}.connTo = connFac + M(i);
            pntr = pntr + 1;
        end;
        features{r}.sample{i}.local = local;
    end;
    
    % pairwise setting
    factor = cell(1, M(i) + size(Pair, 1));
    for p = 1:size(Pair,1)
        factor{p + M(i)}.pot = Pot;
        factor{p + M(i)}.size = [NumStates NumStates];  
    end
    features{r}.sample{i}.factor = factor;
    
    M(i) = M(i) + size(Pair,1);
end 

CRFDATA = updateData(CRFDATA, vars, LOC, M);