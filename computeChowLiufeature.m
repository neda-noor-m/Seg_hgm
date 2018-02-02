function [features, empmean, CRFDATA] = computeChowLiufeature(features, empmean, imlist, CRFDATA, freq, EW, ALLPATHS, train)
% Chow-Liu feature

r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
varname = 'Z';
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;
BinaryStates = 2;

for i = 1:length(imlist)
    imname = imlist{i};    
    [~, blockstart, blockend] = getvardata(vars, varname, i);
    adjmat = EW.adjmat;
    %edge_weights = EW.edge_weights;
    [row_ col_] = find(adjmat);
    IND = row_ > col_;
    row_(IND,:) = [];
    col_(IND,:) = [];    
    Pair_Chow = [row_ col_];
    Pair = Pair_Chow;
    
    temp = 0;
    if train
        Z = CRFDATA.labelsZ{i};
        Z = Z + 1;
        
        % Compute Emperical Means
        EM = Z(Pair);
        for t = 1:size(EM,1)
            if EM(t,1) == 1 && EM(t,2) == 1
                temp = temp + freq(t,1);
            end
            if EM(t,1) == 2 && EM(t,2) == 1
                temp = temp + freq(t,2);
            end
            if EM(t,1) == 1 && EM(t,2) == 2
                temp = temp + freq(t,3);
            end
            if EM(t,1) == 2 && EM(t,2) == 2
                temp = temp + freq(t,4);
            end
        end
    end;
    empmean(r) = empmean(r) + temp;

    % unary setting
    pntr = blockstart;
    for s = 1 : blockend - blockstart + 1
        features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
        features{r}.sample{i}.local{pntr}.pot = zeros(BinaryStates, 1);
        [connFac, ~] = find(Pair==s);
        features{r}.sample{i}.local{pntr}.connTo = connFac+M(i);
        pntr = pntr + 1;
    end

    % pairwise setting
    for p = 1:size(Pair,1)
        features{r}.sample{i}.factor{p+M(i)}.pot =  freq(p,:)';
        features{r}.sample{i}.factor{p+M(i)}.size = [BinaryStates BinaryStates];    
    end

    M(i) = M(i)+size(Pair,1);
end

CRFDATA = updateData(CRFDATA, vars, LOC, M);
