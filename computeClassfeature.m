function [features, empmean, CRFDATA] = computeClassfeature(features, empmean, imlist, CRFDATA, stats, train)

r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
rstart = r;
varname = 'Z';
nim = length(imlist);
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;
NumStates = CRFDATA.NumClasses;
BinaryStates = 2;

for i = 1:length(imlist)
    imname = imlist{i};
    Ustats = stats;
    if train
        Z = CRFDATA.labelsZ{i};
    else
        Z = [];
    end   
    
    nstates = NumStates * BinaryStates;
    [vars, ~, blockstart, ~] = setvardata(vars, varname, i, LOC(i) + 1, LOC(i) + NumStates, nstates);
    LOC(i) = LOC(i) + NumStates;

    r = rstart;
    [empmean, features] = addfeature(features, nim, empmean, r, i, blockstart, Ustats, NumStates, BinaryStates, train, Z);

end    

CRFDATA = updateData(CRFDATA, vars, LOC, M);  

function [empmean, features] = addfeature(features, nim, empmean, r, i, blockstart, U, NumStates, BinaryStates, train, Z)

if r > length(features)
   features{r} = [];
   features{r}.sample = cell(1, nim);
end;

% Compute Emperical Means (fixed)
    if train
       empmean(r) = computeEmpiricalMeans(empmean(r), U, Z + 1, 1);
    end;         

    % unary setting
    pntr = blockstart;
    for s = 1 : NumStates
        features{r}.sample{i}.local{pntr}.NumStates = BinaryStates;
        features{r}.sample{i}.local{pntr}.pot = U(s,:)';
        pntr = pntr + 1;
    end

    % pairwise setting
    features{r}.sample{i}.factor= cell(0,0);
