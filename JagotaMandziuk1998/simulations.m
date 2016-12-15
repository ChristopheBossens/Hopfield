% This section runs a simulation that probes the capacity for a network
% trained with delta learning and Hebbian learning.
clc;clear;
nUnits = 120;
nPatterns = 30;

hebNet = Hopfield(nUnits);

deltaNet = Hopfield(nUnits);
deltaNet.SetLearningRule('Delta',5,1);

sparseness = [0.1 0.3 0.5];
nSparseValues = length(sparseness);

stabilityResults = zeros(nSparseValues,nPatterns,2);

for sparsenessIndex = 1:nSparseValues
    % Generate network with specified degree of sparseness
    patternMatrix = hebNet.GeneratePatternMatrix(nPatterns,sparseness(sparsenessIndex));
    
    % Add patterns and validate if patterns are stable states of the net
    hebNet.ResetWeightMatrix();
    deltaNet.ResetWeightMatrix();
    
    for patternIndex = 1:nPatterns
       clc;display(['Adding pattern ' num2str(patternIndex) '/' num2str(nPatterns)]);
       hebNet.StorePattern(patternMatrix(patternIndex,:),1);
       deltaNet.StorePattern(patternMatrix(patternIndex,:),1);
       
       stableHebbPatterns = sum(hebNet.IsStablePattern(patternMatrix(1:patternIndex,:)));
       stableDeltaPatterns = sum(deltaNet.IsStablePattern(patternMatrix(1:patternIndex,:)));
       stabilityResults(sparsenessIndex,patternIndex,:) = [stableHebbPatterns stableDeltaPatterns]/patternIndex;
    end
end
%%
clf,
for i = 1:3
    subplot(1,3,i)
    hold on
    plot(stabilityResults(i,:,1),'b','LineWidth',2)
    plot(stabilityResults(i,:,2),'g','LineWidth',2)
    set(gca,'YLim',[0 1.1])
    title(['p = ' num2str(sparseness(i))])
    xlabel('Patterns added'),ylabel('Proportion stable')
    axis square
end

%% Error correction capabilities
% Test to what extent the Hebbian rule and the delta rule are able to
% recover from distorted patterns
nPatterns = 8;
nUnits =  120;
bitsFlipped = 1:30;
nBitsFlipped = length(bitsFlipped);
errorCorrection = zeros(2,nBitsFlipped);
iterations = zeros(2,nBitsFlipped);


patternMatrix = hebNet.GeneratePatternMatrix(nPatterns);

hebNet.ResetWeightMatrix();
deltaNet.ResetWeightMatrix();

hebNet.StorePatternMatrix(patternMatrix,1);
deltaNet.StorePatternMatrix(patternMatrix,1);
for i = 1:nBitsFlipped
    for j = 1:nPatterns
        inputPattern = patternMatrix(j,:);
        distortedPattern = hebNet.DistortPattern(inputPattern,bitsFlipped(i)/nUnits);

        [output1, it1] = hebNet.Converge(distortedPattern);
        [output2, it2] = deltaNet.Converge(distortedPattern);

        errorCorrection(1,i) = errorCorrection(1,i) + sum(output1==inputPattern)/nUnits;%(output1*inputPattern')/N;
        errorCorrection(2,i) = errorCorrection(2,i) + sum(output2==inputPattern)/nUnits;

        iterations(1,i) = iterations(1,i) + it1;
        iterations(2,i) = iterations(2,i) + it2;
    end
end

errorCorrection = errorCorrection./nPatterns;
iterations = iterations./nPatterns;
%%
clf,

subplot(1,2,1), hold on
plot(bitsFlipped./nUnits,squeeze(errorCorrection(1,:)),'b','LineWidth',2);
plot(bitsFlipped./nUnits,squeeze(errorCorrection(2,:)),'g','LineWidth',2);
title('Pattern overlap')
set(gca,'YLim',[0 1.5]),xlabel('Noise'),ylabel('Average final overlap')
axis square

subplot(1,2,2), hold on
plot(bitsFlipped./nUnits,squeeze(iterations(1,:)),'b','LineWidth',2);
plot(bitsFlipped./nUnits,squeeze(iterations(2,:)),'g','LineWidth',2);
title('Iterations')
set(gca,'YLim',[0 6]),xlabel('Noise'),ylabel('Average converging time')
axis square

legend('Hebb','Delta')