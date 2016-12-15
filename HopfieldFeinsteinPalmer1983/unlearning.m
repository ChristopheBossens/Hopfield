% Demonstrating the effects of unlearning on network recall capabilities
clear
networkSize = 32;
nPatterns = 5;
nStateSamples = 1000;
hopfield = Hopfield(networkSize);
patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);

for patternIndex = 1:nPatterns
    hopfield.StorePattern(patternMatrix(patternIndex,:),1);
end

pc = Hopfield.TestPatterns(hopfield,patternMatrix,0);
originalWeightMatrix = hopfield.GetWeightMatrix();
hopfield.SetUpdateMode('sync');
%% Run the network through several stages of unlearning, while keeping
% track of the accessibility of the patterns
nUnlearningSteps = 300;
epsilon = 0.01;

patternAccess = zeros(nPatterns,nUnlearningSteps);
spuriousAccess = zeros(1,nUnlearningSteps);

hopfield.SetWeightMatrix(originalWeightMatrix);
for i = 1:nUnlearningSteps
    hopfield.Unlearn(epsilon);
    
    [stableStates, stateCount] = hopfield.SampleWithRandomStates(nStateSamples);
    
    [isSpuriousState, oldPatternIndex] = hopfield.AnalyseStableStates(stableStates,patternMatrix);
    for j = 1:length(stateCount)
        if isSpuriousState(j) == 0
            patternAccess(oldPatternIndex(j),i) = stateCount(j)/nStateSamples;
        else
            spuriousAccess(i) = spuriousAccess(i) + stateCount(j)/nStateSamples;
        end
    end
end
%% Replicate figure 1 from the paper
clf,hold on
plot(mean(patternAccess),'-k','LineWidth',2)
plot(spuriousAccess,'r','LineWidth',2)
plot(patternAccess','g')
xlabel('Unlearning trial')
ylabel('Accessibility')
legend('Mean stored','Mean spurious', 'Individual stored')
title('Pattern accessibility as a function of unlearning')
%% Same, but check the Hamming Radius for the store patterns
nUnlearningSteps = 300;
epsilon = 0.01;

patternHamming = zeros(nPatterns,nUnlearningSteps);
hopfield.SetWeightMatrix(originalWeightMatrix);
for i = 1:nUnlearningSteps
    hopfield.Unlearn(epsilon);
    
    for p = 1:nPatterns
        hd = Hopfield.HammingRadius(hopfield, patternMatrix(p,:),50);
        patternHamming(p,i) = hd;
    end
end
clf,hold on
plot(mean(patternHamming),'-k','LineWidth',2)
plot(patternHamming','--g')
xlabel('Unlearning step')
ylabel('Mean Hammin radius')
title('Pattern attractor radius as a function of unlearning')
legend('Group mean','individual patterns')