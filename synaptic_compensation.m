%% Mimic strategy proposed by Horn et al. where we deleted different levels
% of synapses
clear 
networkSize = 800;
alpha = 0.05;
sparseness = 0.1;
T = sparseness*(1-sparseness)*(1-2*sparseness)/2;

nPatterns = round(alpha*networkSize);
h = Hopfield(networkSize);
h.SetThreshold(T);
h.SetUnitModel('V');

patternMatrix = h.GeneratePatternMatrix(nPatterns,sparseness);
h.AddPatternMatrix(patternMatrix-sparseness,1/networkSize);

recallData = zeros(1,nPatterns);
for i = 1:nPatterns
    originalPattern = patternMatrix(i,:);
    distortedPattern  = h.DistortPattern(originalPattern,0.1);
    finalPattern = h.Converge(distortedPattern);
    
    if sum(finalPattern == originalPattern) == networkSize
        recallData(i) = 1;
    end
end

dValues = [0:0.01:.50];
nD = length(dValues);
recallData = zeros(8,nD);
or = h.GetWeightMatrix();

for i = 1:nD
    display(['Testing deletion: ' num2str(i) '/' num2str(nD)]);
    [pr, deletedWeights, remainingIndices] = h.PruneWeightMatrix(or,dValues(i));
    
    for compensationStrategy = 1:8
        % Select a specific compensation strategy
        switch compensationStrategy
            case 1
                weightMatrix = or;
            case 2
                weightMatrix = pr;
            case 3
                c = 1 + dValues(i)/(1-dValues(i));
                weightMatrix = c.*pr;
            case 4
                s = sum(deletedWeights(:));
                compensationFactor = s/length(remainingIndices(:));
                weightMatrix = pr;
                weightMatrix(remainingIndices(:)) = weightMatrix(remainingIndices(:)) + compensationFactor;
            case 5
                s = abs(sum(deletedWeights(:)));
                compensationFactor = s/length(remainingIndices(:));
                weightMatrix = pr;
                weightMatrix(remainingIndices(:)) = weightMatrix(remainingIndices(:)) + compensationFactor;
            case 6 
                s = abs(sum(deletedWeights(:)));
                compensationFactor = s/length(remainingIndices(:));
                weightMatrix = pr;
                weightMatrix(weightMatrix<0) = weightMatrix(weightMatrix<0) - compensationFactor;
                weightMatrix(weightMatrix>0) = weightMatrix(weightMatrix>0) + compensationFactor;
            case 7
                s = sum(abs(deletedWeights(:)));
                compensationFactor = s/length(remainingIndices(:));
                weightMatrix = pr;
                weightMatrix(remainingIndices(:)) = weightMatrix(remainingIndices(:)) + compensationFactor;
            case 8
                s = sum(abs(deletedWeights(:)));
                compensationFactor = s/length(remainingIndices(:));
                weightMatrix = pr;
                weightMatrix(weightMatrix<0) = weightMatrix(weightMatrix<0) - compensationFactor;
                weightMatrix(weightMatrix>0) = weightMatrix(weightMatrix>0) + compensationFactor;
        end
        h.SetWeightMatrix(weightMatrix);
        
        % Test all stored patterns with current strategy
        for j = 1:nPatterns
            originalPattern = patternMatrix(j,:);
            distortedPattern = h.DistortPattern(originalPattern, 0.1);

            finalState = h.Converge(distortedPattern);
            if sum(finalState == originalPattern) == networkSize
                recallData(compensationStrategy,i) = recallData(compensationStrategy,i) + 1;
            end
        end
        
        
    end
end
recallData = recallData./nPatterns;
%%
clf,
hold on
plot(dValues,recallData(1,:))
plot(dValues,recallData(2,:),'-.k')
xlabel('Deletion factor')
ylabel('P(correct)')