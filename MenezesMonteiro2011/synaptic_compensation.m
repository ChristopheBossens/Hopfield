clear 
% Set hopfield network parameters
networkSize = 800;
alpha = 0.05;
sparseness = 0.1;
T = sparseness*(1-sparseness)*(1-2*sparseness)/2;
nPatterns = round(alpha*networkSize);

dValues = 0:0.01:.90;
nD = length(dValues);

h = Hopfield(networkSize);
h.SetThreshold(T);
h.SetUnitModel('V');
strategies = {'synapse','random','neuron'};
nSimulations = 5;
recallData = zeros(nSimulations,3,8,nD);
compensationStrategies = [1 2 3 8];
h.SetUpdateMode('sync');
% Start the simulation loop
for simulationIndex = 1:nSimulations
    % For each simulation, generate a different pattern matrix
    h.ResetWeightMatrix();
    patternMatrix = h.GeneratePatternMatrix(nPatterns,sparseness);
    h.StorePatternMatrix(patternMatrix-sparseness,1/networkSize);
    or = h.GetWeightMatrix();
    
    % Test for different deletion strategies
    for strategyIndex = 1:length(strategies)
        pruningStrategy = strategies{strategyIndex};
        
        % Test for different degrees of deletion
        for i = 1:nD
            clc;
            display(['Simulation: ' num2str(simulationIndex) '/' num2str(nSimulations)]);
            display(['Testing deletion: ' num2str(i) '/' num2str(nD)]);
            display(['Using strategy: ' pruningStrategy]);
            [pr, deletedWeights, remainingIndices, remainingNeurons] = h.PruneWeightMatrix(or,dValues(i),pruningStrategy);       
            nRemainingIndices = length(remainingIndices);
            
            % Apply different compensation strategies
            % The first and second strategy correspond to the original
            % weights and the pruned weights only to establish baseline
            % performance levels.
            % Strategy 3 corresponds to the one reported in Horn, Rupin,
            % Usher and Herman.
            % Strategies 4-8 correspond to the strategies discussed in
            % Menezes and Monteiro.
            for compensationStrategy = compensationStrategies
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
                            recallData(simulationIndex,strategyIndex,compensationStrategy,i) = recallData(simulationIndex,strategyIndex,compensationStrategy,i) + 1;
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
%%
clf,
M = squeeze(mean(recallData));
% M = squeeze((recallData));
for i = 1:3
    subplot(1,3,i), hold on
    plot(dValues,squeeze(M(i,2,:)),'--k','LineWidth',2)
    plot(dValues,squeeze(M(i,3,:)),'g','LineWidth',2)
    plot(dValues,squeeze(M(i,8,:)),'--g','LineWidth',2)
    axis square
    title(strategies(i))
    xlabel('Deletion')
    ylabel('Performance')
end
legend('Pruned','Compensated #1','Compensated #2')