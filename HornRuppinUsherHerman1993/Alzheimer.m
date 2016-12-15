% Alzheimer modeling
clc;clear 
networkSize = 400;
alpha = 0.05;
p = 0.1;
T = p*(1-p)*(1-2*p)/2;

nExemplars = round(alpha*networkSize);
hopfield = Hopfield(networkSize);
hopfield.SetThreshold(T);
hopfield.SetUnitModel('V');
%% Start by creating a pattern matrix
nPatterns = round(alpha*networkSize);

hopfield.ResetWeightMatrix();
patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,p);
hopfield.StorePatternMatrix(patternMatrix,1/networkSize);
%% Simulation for reproducing figure 2 and 3
% parameter
deletionFactor = linspace(0,networkSize-1,25);
noiseLevel = 0.2;
kValues = [0 0.25 0.375 0.5 0.625 0.75 1];

% Keep a copy of the original weight matrix that we can manipulate
originalWeightMatrix = hopfield.GetWeightMatrix();
results = zeros(length(kValues),length(deletionFactor));

for delIndex = 1:length(deletionFactor)
    display(['Deletion factor: ' num2str(deletionFactor(delIndex))]);
    
    d = deletionFactor(delIndex)/networkSize;   
    deletedWeightMatrix = Hopfield.PruneWeightMatrix(originalWeightMatrix,d);
    
    for kIndex = 1:length(kValues)
        k = kValues(kIndex);
        c = 1 + ((d*k)/(1-d));
        hopfield.SetWeightMatrix(c.*deletedWeightMatrix);
        [pc,it] = hopfield.TestPatterns(hopfield, patternMatrix, noiseLevel);
        
        results(kIndex,delIndex)= mean(pc);
    end    
end
%% Figure 1
% Get the input distributions with the original weight matrix
d = 0.25;
hopfield.SetWeightMatrix(originalWeightMatrix);
[counts,bins,a,b] = hopfield.GetPotentialDistribution(100);

origActive = histfit(a(b>0));
oa(1,:) = get(origActive(2),'XData');
oa(2,:) = get(origActive(2),'YData');

origInactive = histfit(a(b<=0));
oi(1,:) = get(origInactive(2),'XData');
oi(2,:) = get(origInactive(2),'YData');

% Get the input distribution with the pruned weight matrix
prunedWeights = Hopfield.PruneWeightMatrix(originalWeightMatrix,0.3);
hopfield.SetWeightMatrix(prunedWeights);
[counts1,bins,a,b] = hopfield.GetPotentialDistribution(100);

delActive = histfit(a(b>0));
da(1,:) = get(delActive(2),'XData');
da(2,:) = get(delActive(2),'YData');

delInactive = histfit(a(b<=0));
di(1,:) = get(delInactive(2),'XData');
di(2,:) = get(delInactive(2),'YData');

% Get the input distribution with the compensated weight matrix
opc = 1/(1-d);
hopfield.SetWeightMatrix(opc.*prunedWeights);
[counts2,bins,a,b] = hopfield.GetPotentialDistribution(100);

cActive = histfit(a(b>0));
ca(1,:) = get(cActive(2),'XData');
ca(2,:) = get(cActive(2),'YData');

cInactive = histfit(a(b<=0));
ci(1,:) = get(cInactive(2),'XData');
ci(2,:) = get(cInactive(2),'YData');

clf,hold on
plot(oa(1,:),oa(2,:),'k','LineWidth',2)
plot(da(1,:),da(2,:),'b','LineWidth',2)
plot(ca(1,:),ca(2,:),'g','LineWidth',2)
plot(T,0,'vr','MarkerFaceColor','r','MarkerSize',9)
plot(oi(1,:),oi(2,:),'--k','LineWidth',2)
plot(di(1,:),di(2,:),'--b','LineWidth',2)
plot(ci(1,:),ci(2,:),'--g','LineWidth',2)
legend('Original','Deleted','OPC','Threshold')
title('Distribution of input potentials')
xlabel('Mean input potential')
ylabel('Count')
%% Figure 2
clf,hold on
plot(deletionFactor./networkSize,results')
line([0 1],[0.75 0.75],'LineStyle','--','Color','k')
line([0 1],[0.25 0.25],'LineStyle','--','Color','k')
xlabel('deletion factor')
ylabel('Performance')
title('Performance for different compensation strategies')
legend('0','0.25','0.375','0.5')

%% Figure 3
d = deletionFactor./networkSize;

transition = zeros(2,length(kValues));
for kIndex = 1:length(kValues)
    idx = find(results(kIndex,:) > 0.75);
    
    x1 = d(idx(end));
    x2 = d(idx(end)+1);
    y1 = results(kIndex,idx(end));
    y2 = results(kIndex,idx(end)+1);
    alpha = (y2-y1)/(x2-x1);
    beta = y1 - alpha*x1;
    
    transition(1,kIndex) = (0.75-beta)/alpha;
    
    idx = find(results(kIndex,:) > 0.25);
    
    x1 = d(idx(end));
    x2 = d(idx(end)+1);
    y1 = results(kIndex,idx(end));
    y2 = results(kIndex,idx(end)+1);
    alpha = (y2-y1)/(x2-x1);
    beta = y1 - alpha*x1;
    
    transition(2,kIndex) = (0.25-beta)/alpha;
end
clf,hold on
plot(transition(1,:), kValues,'-k','LineWidth',2)
plot(transition(2,:), kValues,'--k','LineWidth',2)
title('Transition region')
xlabel('d'),ylabel('k')
legend('75%','25%')