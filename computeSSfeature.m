function [features, empmean, CRFDATA] = computeSSfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train, sceneFeature)

r = length(features) + 1;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
varname = 'SS';
rstart = r;
NumStates = CRFDATA.NumClasses;
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;

for i = 1:length(imlist)
    imname = imlist{i};
    I = load(fullfile(ALLPATHS.SUPIXEL_PATH,[imname '.mat']));
    P = I.potential;
    U1 = -P;
     %My code
%                                                                     if sceneFeature
%                                                                         data = load(fullfile(ALLPATHS.SCENE_PATH,[imname '.mat']), 'potential'); %labelSuperpixel
%                                                                         correspondence_file = load (fullfile( ALLPATHS.SCENE_PATH, ['/potential/correspondence_obj_scene.mat']));
%                                                                         correspondence_obj_scene = correspondence_file.correspondence_obj_scene;
%                                                                         U_scene = data.potential;
%                                                                         for iclass = 1:NumStates
%                                                                            if correspondence_obj_scene(iclass) ~= -1, U1(:, iclass) = U1(:, iclass) + U_scene(correspondence_obj_scene(iclass)); end
%                                                                         end
%                                                                     end
%                                                                   end of my code

    if train
       labelSS = CRFDATA.labelsSS{i};
       row = CRFDATA.indsSS{i};
       U1 = U1(row,:);
    else
        labelSS = [];
    end;
    nSS = size(U1, 1);
     
    nstates = nSS * NumStates;
    [vars, ~, blockstart, ~] = setvardata(vars, varname, i, LOC(i) + 1, LOC(i) + nSS, nstates);
    LOC(i) = LOC(i) + nSS;
    
    % Compute Emperical Means
    r = rstart;
    if train
       empmean = getempmean(empmean, r, U1, labelSS, train);
    end;

    % Compute the SS feature
    features = addpot(features, r, i, NumStates, blockstart, U1, length(imlist));
end

CRFDATA = updateData(CRFDATA, vars, LOC, M);


function features = addpot(features, r, i, NumStates, blockstart, U, nimages)

if r > length(features)
    features{r} = [];
    features{r}.sample = cell(1, nimages);
end;

pntr = blockstart;
for s = 1:size(U, 1)
    features{r}.sample{i}.local{pntr}.NumStates = NumStates;
    features{r}.sample{i}.local{pntr}.pot = U(s,:)';
    pntr = pntr + 1;
end

% pairwise setting
features{r}.sample{i}.factor= cell(0,0);


function empmean = getempmean(empmean, r, U, labelSP, train)

if train
    empmean(r) = computeEmpiricalMeans(empmean(r), U, labelSP, 1);
end; 
