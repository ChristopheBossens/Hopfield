clc;clear;
h = Hopfield(100);
nPatterns = 50;
nUnits = 100;
p = h.GeneratePatternMatrix(nPatterns);

hebNet = Hopfield(nUnits);
storkeyNet = Hopfield(nUnits);

%% Performance metric 1:
% How many patterns can be stored as stable states
alpha = [0.1:0.1:0.5];

overlapResults = zeros(2,nPatterns);

for tIndex = 1:nPatterns
    hebNet.AddPattern(p(tIndex,:));
    storkeyNet.PerformStorkeyUpdate(p(tIndex,:));
    
    for i = 1:tIndex
        currentPattern = p(i,:);
        
        % Test the heb model
        nextState = hebNet.Iterate(currentPattern);
        overlapResults(1,tIndex) = overlapResults(1,tIndex) + (nextState*currentPattern')/(nUnits*tIndex);
        
        % Test the Storkey model
        nextState = storkeyNet.Iterate(currentPattern);
        overlapResults(2,tIndex) = overlapResults(2,tIndex) + (nextState*currentPattern')/(nUnits*tIndex);
    end
end

plot(overlapResults')
legend('Hebb rule','Storkey rule')
%% Performance metric 2:
% Mean attractor radius for different network loads
nPatterns = 8;
pm = h.GeneratePatternMatrix(nPatterns);
hebNet.ResetWeightMatrix();
storkeyNet.ResetWeightMatrix();

radiusData = zeros(2,nPatterns);

for tIndex = 1:nPatterns
    hebNet.AddPattern(pm(tIndex,:));
    storkeyNet.PerformStorkeyUpdate(pm(tIndex,:));
    
    for i = 1:tIndex
        h1 = Hopfield.HammingRadius(hebNet,pm(i,:),50);
        h2 = Hopfield.HammingRadius(storkeyNet,pm(i,:),50);
        
        radiusData(1,tIndex) = radiusData(1,tIndex) + h1/tIndex;
        radiusData(2,tIndex) = radiusData(2,tIndex) + h2/tIndex;
    end
end

plot(radiusData','LineWidth',2)
legend('Hebb rule','Storkey rule')
title('Attractor radius')
xlabel('Network load')
ylabel('Minimum distance')
%% Performance metric 3:
% Effective capacity
nPatterns = 30;
pm = h.GeneratePatternMatrix(nPatterns);
p = [0.1 0.2 0.3];

hebNet.ResetWeightMatrix();
storkeyNet.ResetWeightMatrix();

ec = zeros(2,length(p),nPatterns);
for tIndex = 1:nPatterns
    hebNet.AddPattern(pm(tIndex,:));
    storkeyNet.PerformStorkeyUpdate(pm(tIndex,:));
    
    for i = 1:tIndex
        for pIndex = 1:length(p)
           inputPattern = pm(i,:);
           distortedPattern = h.DistortPattern(inputPattern,p(pIndex));
           
           o1 = hebNet.Converge(distortedPattern);
           o2 = storkeyNet.Converge(distortedPattern);
           
           ec(1,pIndex,tIndex) = ec(1,pIndex,tIndex) + (o1*inputPattern')/(nUnits*tIndex);
           ec(2,pIndex,tIndex) = ec(2,pIndex,tIndex) + (o2*inputPattern')/(nUnits*tIndex);
        end
    end
end

subplot(1,2,1)
plot(squeeze(ec(1,:,:))')
set(gca,'YLim',[0.0 1])
axis square
title('Hebb rule')
subplot(1,2,2)
plot(squeeze(ec(2,:,:))')
set(gca,'YLim',[0.0 1])
axis square
title('Storkey rule')

%% Performance metric 4:
% Stable states when patterns are biased
nUnits      = 100;
nPatterns   = 10;
patternBias = 0.1:0.05:0.9;

recallMatrix = zeros(2,length(patternBias));
for bIndex = 1:length(patternBias)
   pm = h.GeneratePatternMatrix(nPatterns,patternBias(bIndex)); 
   
   hebNet.ResetWeightMatrix();
   storkeyNet.ResetWeightMatrix();
   
   for pIndex = 1:nPatterns;
       hebNet.AddPattern(pm(pIndex,:));
       storkeyNet.PerformStorkeyUpdate(pm(pIndex,:));
   end
   
   for tIndex = 1:nPatterns
       inputPattern = pm(tIndex,:);

       o1 = hebNet.Iterate(inputPattern);
       o2 = storkeyNet.Iterate(inputPattern);

       if sum(o1 ~= inputPattern) == 0
           recallMatrix(1,bIndex) = recallMatrix(1,bIndex)+ (1/nPatterns);
       end
       if sum(o2 ~= inputPattern) == 0
           recallMatrix(2,bIndex) = recallMatrix(2,bIndex)+ (1/nPatterns);
       end
   end

end

clf,
plot(repmat(patternBias,2,1)',recallMatrix','LineWidth',2)
legend('Hebb rule','Storkey rule')
xlabel('Pattern bias')
ylabel('Proportion stable')
title('Storage of biased patterns')
set(gca,'YLim',[0,1.3])