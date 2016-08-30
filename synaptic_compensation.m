clear 
% Set hopfield network parameters
networkSize = 800;
alpha = 0.05;
sparseness = 0.1;
T = sparseness*(1-sparseness)*(1-2*sparseness)/2;
dValues = [0:0.01:.50];
nD = length(dValues);

h = Hopfield(networkSize);
h.SetThreshold(T);
h.SetUnitModel('V');
strategies = {'synapse','random','neuron'};
nSimulations = 50;
recallData = zeros(nSimulations,3,8,nD);


% Generate the pattern matrix
for simulationIndex = 1:nSimulations
    nPatterns = round(alpha*networkSize);
    patternMatrix = h.GeneratePatternMatrix(nPatterns,sparseness);
    h.ResetWeightMatrix();
    h.AddPatternMatrix(patternMatrix-sparseness,1/networkSize);

    for i = 1:nPatterns
        originalPattern = patternMatrix(i,:);
        distortedPattern  = h.DistortPattern(originalPattern,0.1);
        finalPattern = h.Converge(distortedPattern);

        if sum(finalPattern == originalPattern) == networkSize
            recallData(i) = 1;
        end
    end
    or = h.GetWeightMatrix();
    % Prepare the simulation
    for strategyIndex = 1:length(strategies)
        pruningStrategy = strategies{strategyIndex};
        for i = 1:nD
            display(['Testing deletion: ' num2str(i) '/' num2str(nD)]);
            [pr, deletedWeights, remainingIndices, remainingNeurons] = h.PruneWeightMatrix(or,dValues(i),pruningStrategy);       

            % Apply different compensation strategies
            for compensationStrategy = [1 2 3 8]
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
                        compensationFactor = s/nRemainingIndices;
                        weightMatrix = pr;
                        weightMatrix(remainingIndices(:)) = weightMatrix(remainingIndices(:)) + compensationFactor;
                    case 5
                        s = abs(sum(deletedWeights(:)));
                        compensationFactor = s/nRemainingIndices;
                        weightMatrix = pr;
                        weightMatrix(remainingIndices(:)) = weightMatrix(remainingIndices(:)) + compensationFactor;
                    case 6 
                        s = abs(sum(deletedWeights(:)));
                        compensationFactor = s/nRemainingIndices;
                        weightMatrix = pr;
                        weightMatrix(weightMatrix<0) = weightMatrix(weightMatrix<0) - compensationFactor;
                        weightMatrix(weightMatrix>0) = weightMatrix(weightMatrix>0) + compensationFactor;
                    case 7
                        s = sum(abs(deletedWeights(:)));
                        compensationFactor = s/nRemainingIndices;
                        weightMatrix = pr;
                        weightMatrix(remainingIndices(:)) = weightMatrix(remainingIndices(:)) + compensationFactor;
                    case 8
                        s = sum(abs(deletedWeights(:)));
                        compensationFactor = s/nRemainingIndices;
                        weightMatrix = pr;
                        weightMatrix(weightMatrix<0) = weightMatrix(weightMatrix<0) - compensationFactor;
                        weightMatrix(weightMatrix>0) = weightMatrix(weightMatrix>0) + compensationFactor;
                end
                h.SetWeightMatrix(weightMatrix);

                % Test all stored patterns following application of the
                % compensation strategy
                for j = 1:nPatterns
                    originalPattern = patternMatrix(j,:);
                    distortedPattern = h.DistortPattern(originalPattern, 0.1);

                    finalState = h.Converge(distortedPattern);
                    if strcmp(pruningStrategy,'neuron')
                        if sum(finalState(remainingNeurons) == originalPattern(remainingNeurons)) == length(remainingNeurons)
                            recallData(compensationStrategy,i) = recallData(compensationStrategy,i) + 1;
                        end
                    else
                        if sum(finalState == originalPattern) == networkSize
                            recallData(simulationIndex,strategyIndex,compensationStrategy,i) = recallData(simulationIndex,strategyIndex,compensationStrategy,i) + 1;
                        end
                    end
                end


            end
        end
    end
end
recallData = recallData./nPatterns;
%% Plot the results from the different compensation strategies
clf,
for i = 3:8
    subplot(2,3,i-2),hold on
    plot(dValues,recallData(2,:),'-.k')
    plot(dValues,recallData(i,:),'k')
    set(gca,'YLim',[0 1])
end