% function [features, empmean, CRFDATA] = computeSPfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train)
function [features, empmean, CRFDATA] = computeSPfeature(features, empmean, imlist, CRFDATA, ALLPATHS, train, sceneFeature)
% SUPERPIXEL FEATURE

r = length(features) + 1;
rstart = r;
features{r} = [];
features{r}.sample = cell(1, length(imlist));
varname = 'SP';
NumStates = CRFDATA.NumClasses;
vars = CRFDATA.vars;
LOC = CRFDATA.COUNT_UNARY;
M = CRFDATA.COUNT_FACTORS;
featparams = CRFDATA.featparams;

for i = 1:length(imlist)
    imname = imlist{i};    
    I = load(fullfile(ALLPATHS.PIXEL_PATH,[imname '.mat'])); %labelSuperpixel
    P1 = I.potential;
    U1 = -P1;                 
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if featparams.useFgBg
       I2 = load(fullfile(ALLPATHS.PIXELFGBG_PATH,[imname '.mat'])); %labelSuperpixel
       P2 = I2.potential;        
       U2 = -P2;  
    end;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                                                    %My code
                                                                    if sceneFeature
                                                                        data = load(fullfile(ALLPATHS.SCENE_PATH,[imname '.mat']), 'potential'); %labelSuperpixel
                                                                        correspondence_file = load (fullfile( ALLPATHS.SCENE_PATH, ['/potential/correspondence_obj_scene.mat']));
                                                                        correspondence_obj_scene = correspondence_file.correspondence_obj_scene;
                                                                        U_scene = data.potential;
                                                                        for iclass = 1:NumStates
                                                                           if correspondence_obj_scene(iclass) ~= -1
                                                                               U1(:, iclass) = U1(:, iclass) + U_scene(correspondence_obj_scene(iclass)); 
                                                                           end
                                                                        end
                                                                    end
                                                                    %end of my code
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if featparams.useClassSP
       I3 = load(fullfile(ALLPATHS.PIXELCLASS_PATH,[imname '.mat'])); %labelSuperpixel
       P3 = I3.potential;        
       U3 = -P3;  
    end;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if train
       labelSP = CRFDATA.labelsSP{i};
       row = CRFDATA.indsSP{i};
       U1 = U1(row,:);      
       if featparams.useFgBg
          U2 = U2(row,:); 
       end
       if featparams.useClassSP
          U3 = U3(row,:); 
       end
    else
        labelSP = [];
    end;
    
    nstates = size(U1, 1) * NumStates;
    [vars, ~, blockstart, ~] = setvardata(vars, varname, i, LOC(i) + 1, LOC(i) + size(U1, 1), nstates);  
    LOC(i) = LOC(i) + size(U1, 1);
    
    % Compute Emperical Means and SP features
    r = rstart;
    empmean = getempmean(empmean, r, U1, labelSP, train);
    features = addpot(features, r, i, NumStates, blockstart, U1, length(imlist));
    if featparams.useFgBg
       r = r + 1;
       empmean = getempmean(empmean, r, U2, labelSP, train);
       features = addpot(features, r, i, NumStates, blockstart, U2, length(imlist));
    end;
    if featparams.useClassSP
       r = r + 1;
       empmean = getempmean(empmean, r, U3, labelSP, train);
       features = addpot(features, r, i, NumStates, blockstart, U3, length(imlist));
    end;
end   

CRFDATA = updateData(CRFDATA, vars, LOC, M);
%--------------------------------------- 


function features = addpot(features, r, i, NumStates, blockstart, U, nimages)

if r > length(features)
    features{r} = [];
    features{r}.sample = cell(1, nimages);
end;
if ~isfield(features{r}.sample{i}, 'local')
    local = cell(1, blockstart + size(U, 1) - 1);
else
    local = features{r}.sample{i}.local;
end;
numloc = length(local);
if  numloc < blockstart + size(U, 1) - 1
    local = [local, cell(1, blockstart + size(U, 1) - 1 - numloc)];
end;

pntr = blockstart;
for s = 1:size(U, 1)
    local{pntr}.NumStates = NumStates;
    local{pntr}.pot = U(s,:)';
    pntr = pntr + 1;
end
features{r}.sample{i}.local = local;
% pairwise setting
features{r}.sample{i}.factor= cell(0,0);


function empmean = getempmean(empmean, r, U, labelSP, train)

if train
    empmean(r) = computeEmpiricalMeans(empmean(r), U, labelSP, 1);
end;
