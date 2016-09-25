% This section runs a simulation that probes the capacity for a network
% trained with delta learning and Hebbian learning. For Hebbian learning,
% a second network is run which uses threshold values to adjust for low
% activity patterns
clc;clear;
N = 120;
P = 30;

hebNet = Hopfield(N);
hebNet.SetUnitModel('S');
deltaNet = Hopfield(N);

sparseness = [0.1 0.3 0.5];
threshold  = [50  40  0];
nSparseValues = length(sparseness);

stabilityResults = zeros(nSparseValues,P,3);
for sparsenessIndex = 1:nSparseValues
    % Generate network with specified degree of sparseness
    patternMatrix = hebNet.GeneratePatternMatrix(P,sparseness(sparsenessIndex));
    
    % Add patterns and validate if patterns are stable states of the net
    hebNet.ResetWeightMatrix();
    deltaNet.ResetWeightMatrix();
    for patternIndex = 1:P
       clc;display(['Adding pattern ' num2str(patternIndex) '/' num2str(P)]);
       hebNet.AddPattern(patternMatrix(patternIndex,:),1);
       deltaNet.LearnDeltaPatterns(patternMatrix(1:patternIndex,:),1);
        
       stableHebbPatterns = 0;
       stableDeltaPatterns =0;
       stableAdjPatterns  = 0;
       for testIndex = 1:patternIndex      
           stableHebbPatterns = stableHebbPatterns + hebNet.IsStablePattern(patternMatrix(testIndex,:));
           stableDeltaPatterns = stableDeltaPatterns + deltaNet.IsStablePattern(patternMatrix(testIndex,:));                  
           
           hebNet.SetThreshold(threshold(sparsenessIndex));
           stableAdjPatterns = stableAdjPatterns + hebNet.IsStablePattern(patternMatrix(testIndex,:));
           hebNet.SetThreshold(0);
       end
       stabilityResults(sparsenessIndex,patternIndex,:) = [stableHebbPatterns stableDeltaPatterns stableAdjPatterns]/patternIndex;
    end
end
%%
clf,
for i = 1:3
    subplot(1,3,i)
    hold on
    plot(stabilityResults(i,:,1),'b','LineWidth',2)
    plot(stabilityResults(i,:,2),'g','LineWidth',2)
    plot(stabilityResults(i,:,3),'--b','LineWidth',2)
    set(gca,'YLim',[0 1.1])
    title(['p = ' num2str(sparseness(i))])
    xlabel('Patterns added'),ylabel('Proportion stable')
    axis square
end

%% Error correction capabilities
% Test to what extent the Hebbian rule and the delta rule are able to
% recover from distorted patterns
P = 8;

bitsFlipped = 1:30;
nBitsFlipped = length(bitsFlipped);
errorCorrection = zeros(3,3,nBitsFlipped);
iterations = zeros(3,3,nBitsFlipped);

for sparsenessIndex = 1:nSparseValues
    a = sparseness(sparsenessIndex);
    denom = 1/(2*a*(1-a)*N);
    patternMatrix = hebNet.GeneratePatternMatrix(P,a);

    hebNet.ResetWeightMatrix();
    deltaNet.ResetWeightMatrix();

    hebNet.AddPatternMatrix(patternMatrix,1);
    deltaNet.LearnDeltaPatterns(patternMatrix,1);
    for i = 1:nBitsFlipped
        for j = 1:P
            inputPattern = patternMatrix(j,:);
            distortedPattern = hebNet.DistortPattern(inputPattern,bitsFlipped(i)/N);

            [output1, it1] = hebNet.Converge(distortedPattern);
            [output2, it2] = deltaNet.Converge(distortedPattern);
            hebNet.SetThreshold(threshold(sparsenessIndex));
            [output3, it3] = hebNet.Converge(distortedPattern);
            hebNet.SetThreshold(0);
            
            vPattern = ((inputPattern+1)./2)-a;
            errorCorrection(sparsenessIndex,1,i) = errorCorrection(sparsenessIndex,1,i) + sum(output1==inputPattern)/N;%(output1*inputPattern')/N;
            errorCorrection(sparsenessIndex,2,i) = errorCorrection(sparsenessIndex,2,i) + sum(output2==inputPattern)/N;
            errorCorrection(sparsenessIndex,3,i) = errorCorrection(sparsenessIndex,3,i) + sum(output3==inputPattern)/N;
            iterations(sparsenessIndex,1,i) = iterations(sparsenessIndex,1,i) + it1;
            iterations(sparsenessIndex,2,i) = iterations(sparsenessIndex,2,i) + it2;
            iterations(sparsenessIndex,3,i) = iterations(sparsenessIndex,3,i) + it3;
        end
    end
end
errorCorrection = errorCorrection./P;
iterations = iterations./P;
%%
clf,
for i = 1:3
    subplot(2,3,i), hold on
    plot(squeeze(errorCorrection(i,1,:)),'b','LineWidth',2);
    plot(squeeze(errorCorrection(i,2,:)),'g','LineWidth',2);
    plot(squeeze(errorCorrection(i,3,:)),'--b','LineWidth',2);
    set(gca,'YLim',[0 1.5]),xlabel('Bits flipped'),ylabel('Average final overlap')
    axis square
    title(['sparseness = ' num2str(sparseness(i))])
    subplot(2,3,i+3), hold on
    plot(squeeze(iterations(i,1,:)),'b','LineWidth',2);
    plot(squeeze(iterations(i,2,:)),'g','LineWidth',2);
    plot(squeeze(iterations(i,3,:)),'--b','LineWidth',2);
    set(gca,'YLim',[0 6]),xlabel('Bits flipped'),ylabel('Average converging time')
    axis square
end