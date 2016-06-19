% Alzheimer modeling
clear 
networkSize = 256;
alpha = 0.1;
p = 0.1;
T = p*(1-p)*(1-2*p)/2;

nExemplars = round(alpha*networkSize);
hopfield = Hopfield(networkSize);
hopfield.SetThreshold(T);
hopfield.SetUnitModel('V');
%% Start by creating a pattern matrix
nPatterns = round(alpha*networkSize);
patternMatrix = zeros(nPatterns,networkSize);

hopfield.ResetWeights();
for i = 1:nPatterns
    pattern = hopfield.GeneratePattern(p);
    hopfield.AddPattern(pattern-p,1/networkSize);
    patternMatrix(i,:) = pattern;
end
weightMatrix = hopfield.GetWeightMatrix();

%% Stress test the network without modifications
noiseValues = [0:(1/networkSize):(50/networkSize)];
overlapMatrix = zeros(length(noiseValues),nPatterns);
iterationMatrix = zeros(length(noiseValues),nPatterns);

for noiseIdx = 1:length(noiseValues)
    for i = 1:nPatterns
        pattern = patternMatrix(i,:);
        pattern = hopfield.DistortPattern(pattern,noiseValues(noiseIdx));
        [finalState, nIterations] = hopfield.Converge(pattern);

        overlap = (1/(p*(1-p)*(networkSize)))*( (pattern-p)*finalState');
%         overlapMatrix(noiseIdx,i) = sum(finalState == patternMatrix(i,:))/networkSize;
        overlapMatrix(noiseIdx,i) = overlap;
        iterationMatrix(noiseIdx,i) = nIterations;
    end
end
% 
[activeDistribution, inactiveDistribution, binValues] = hopfield.GetActivityDistribution(100);
clf,
subplot(3,1,1),plot(noiseValues, mean(overlapMatrix,2)),title('Average pattern overlap')
xlabel('Noise level'),ylabel('Pattern overlap'),set(gca,'YLim',[0 1])
subplot(3,1,2),plot(noiseValues, mean(iterationMatrix,2)),title('Average iterations')
xlabel('Noise level'),ylabel('Iterations')

subplot(3,1,3),hold on
bar(binValues,activeDistribution,'g'),title('Active unit postsynaptic potential')
bar(binValues,inactiveDistribution,'r'),title('Inactive unit postsynaptic potential')
line([0 0],[0 max([activeDistribution inactiveDistribution])],'Color','k','LineWidth',2,'LineStyle',':')
line([T T],[0 max([activeDistribution inactiveDistribution])],'Color','k','LineWidth',2,'LineStyle','--')
set(gca,'XLim',[binValues(1) binValues(end)]),xlabel('Input potential'),ylabel('Count')
legend('Active unit','Inactive unit','Threshold','Location','NorthEastOutside')
%% Removing synaptic connections
% We introduce synaptic deletion with a deletion factor that randomly sets
% a number of incoming synaptic connections to zero
% We then test how the network copes with distorted input patterns
deletionFactor = (0:2:100);
noiseLevel     = (0:2:100);

overlapMatrix = zeros(length(deletionFactor),length(noiseLevel));
iterMatrix = zeros(length(deletionFactor),length(noiseLevel));
pcMatrix = zeros(length(deletionFactor), length(noiseLevel));

for delIndex = 1:length(deletionFactor)
    display(['Deletion factor: ' num2str(deletionFactor(delIndex))]);
    deletedWeightMatrix = weightMatrix;    
    for rowIndex = 1:size(weightMatrix,1)
        delIndices = randperm(size(weightMatrix,2));
        if deletionFactor(delIndex) > 0
            delIndices = delIndices(1:deletionFactor(delIndex));
            deletedWeightMatrix(rowIndex,delIndices) = 0;
        end
    end
    
    hopfield.SetWeightMatrix(deletedWeightMatrix);
    for noiseIndex = 1:length(noiseLevel)
        overlapVector = zeros(1,nPatterns);
        iterVector = zeros(1,nPatterns);
        for patternIndex = 1:nPatterns
            pattern = patternMatrix(patternIndex,:);
            noisePattern = hopfield.DistortPattern(pattern, noiseLevel(noiseIndex)/networkSize);
            
            [finalState,nIterations] = hopfield.Converge(noisePattern, 'async',T);
            iterVector(patternIndex) = nIterations;
            overlapVector(patternIndex) = (1/(p*(1-p)*networkSize)).*( (pattern-p) * finalState');
            
            if sum(pattern==finalState) == networkSize
                pcMatrix(delIndex,noiseIndex) = pcMatrix(delIndex, noiseIndex) + 1;
            end
        end
        overlapMatrix(delIndex,noiseIndex) = mean(overlapVector);
        iterMatrix(delIndex, noiseIndex) = mean(iterVector);
    end    
end
pcMatrix = pcMatrix./nPatterns;
%%
clf,
subplot(2,1,1),imshow(overlapMatrix,[]), colorbar
axis on
title('Overlap matrix')
set(gca,'Ydir','Normal','XTick',0:20:100,'XTickLabel',round(100.*(0:20:100)./networkSize)/100,...
    'YTick',0:20:100,'YTickLabel',round(100.*(0:20:100)./networkSize)/100)
xlabel('Pattern noise')
ylabel('Deletion factor')

subplot(2,1,2),imshow(iterMatrix,[]), colorbar
title('Iteration matrix')
set(gca,'YDir','Normal','XTick',0:20:100,'XTickLabel', round(100.*(0:20:100)./networkSize)/100,...
    'YTick',0:20:100,'YTickLabel',round(100.*(0:20:100)./networkSize)/100)
set(gca,'Ydir','Normal')
xlabel('Pattern noise')
ylabel('Deletion factor')
axis on
colormap hot
%% Synaptic compensation
% For a fixed noise level, we test with different deletion factors and
% synaptic compensation factors how the network reacts

deletionFactor = round(linspace(0,networkSize,100));
compensationFactor = (0:0.01:1);
noiseLevel     = 0.2;

overlapMatrix = zeros(length(deletionFactor),length(compensationFactor));
iterMatrix = zeros(length(deletionFactor),length(compensationFactor));
pcMatrix = zeros(length(deletionFactor), length(compensationFactor));

for delIndex = 1:length(deletionFactor)
    display(['Deletion factor: ' num2str(deletionFactor(delIndex))]);
    deletedWeightMatrix = weightMatrix;
    d = deletionFactor(delIndex)/networkSize;
    
    for rowIndex = 1:size(weightMatrix,1)
        delIndices = randperm(size(weightMatrix,2));
        if deletionFactor(delIndex) > 0
            delIndices = delIndices(1:deletionFactor(delIndex));
            deletedWeightMatrix(rowIndex,delIndices) = 0;
        end
    end
    
    
    for k = 1:length(compensationFactor)
        c = 1 + ((d*k)/(1-d));
        hopfield.SetWeightMatrix(c.*deletedWeightMatrix);
        
        overlapVector = zeros(1,nPatterns);
        iterVector = zeros(1,nPatterns);
        for patternIndex = 1:nPatterns
            pattern = patternMatrix(patternIndex,:);
            noisePattern = hopfield.DistortPattern(pattern, noiseLevel);
            
            [finalState,nIterations] = hopfield.Converge(noisePattern, 'async',T);
            iterVector(patternIndex) = nIterations;
            overlapVector(patternIndex) = (1/(p*(1-p)*networkSize)).*( (pattern-p) * finalState');
            
            if sum(pattern==finalState) == networkSize
                pcMatrix(delIndex,k) = pcMatrix(delIndex, k) + 1;
            end
        end
        overlapMatrix(delIndex,k) = mean(overlapVector);
        iterMatrix(delIndex, k) = mean(iterVector);
    end    
end
pcMatrix = pcMatrix./nPatterns;
%%
clf,
subplot(2,1,1),imshow(overlapMatrix,[]), colorbar
axis on
title('Overlap matrix')
set(gca,'Ydir','Normal')
% set(gca,'Ydir','Normal','XTick',0:20:100,'XTickLabel',round(100.*(0:20:100)./networkSize)/100,...
%     'YTick',0:20:100,'YTickLabel',round(100.*(0:20:100)./networkSize)/100)
xlabel('k')
ylabel('d')

subplot(2,1,2),imshow(pcMatrix,[]), colorbar
title('Iteration matrix')
% set(gca,'YDir','Normal','XTick',0:20:100,'XTickLabel', round(100.*(0:20:100)./networkSize)/100,...
%     'YTick',0:20:100,'YTickLabel',round(100.*(0:20:100)./networkSize)/100)
set(gca,'Ydir','Normal')
xlabel('k')
ylabel('d')
axis on
colormap hot