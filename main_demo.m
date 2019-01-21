%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the code for paper 298.
%
% The number of iterations T can be set from 8-10 for better MAP.
% The most time-consuming part is the learning of lyap function (F-step).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function main_demo()

%Length of hash codes
nbits_set = [16,24,32,48,64,96];


fprintf('Preparing Data...\n');

% Loading and processing data
load('nuswide.mat'); % NUS-WIDE for demo,description of this dataset can be found in our paper
WtrueTestTraining=train_label*test_label'>0;
traindata = normalize(train_data);
testdata = normalize(test_data);

load('L.mat'); % L=D-S; pre-computed
X=traindata;  A=pinv(X'*X+eye(size(X,2)))*X'; % pre-compute A before learning

% Training and Evaluation 
fprintf('Training & Evaluating...\n');

for ii=1:length(nbits_set)
    
    nbits=nbits_set(ii);
    
    % Train Our Model
    tic
    [U_logical_trn, W]= train_OUR(traindata,train_tag,L,A,nbits);
    U_logical_tst = testdata*W > 0;
    toc
    
    % MAP Evaluation
    B_compact_trn = compactbit(U_logical_trn);
    B_compact_tst = compactbit(U_logical_tst);
    DHamm = hammingDist(B_compact_tst, B_compact_trn);
    [~, orderH] = sort(DHamm, 2);
    
    MAP_OUR(ii) = calcMAP(orderH, WtrueTestTraining');
    fprintf('MAP result is %d in case of %d-bit.  \n',MAP_OUR(ii),nbits);

end
end