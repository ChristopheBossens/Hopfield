clear;
N = 16;
P = 2;

h = Hopfield(N);
hDel = Hopfield(N-1);

kValues = 0:0.1:1;
dValues = 0:0.1:0.5;

nK = length(kValues);
nD = length(dValues);
nStrategies = 5;

overlapData =zeros(nK, nD, nStrategies);

% Generate all possible starting states
if N == 16
    startingStates=zeros(2^N,N);
    for i = 1:16
        startingStates(:,i) = repmat([zeros(2^(16-i),1); ones(2^(16-i),1)],2^(i-1),1);
    end
end

% Initialize the networks
p = h.GeneratePatternMatrix(P);
h.ResetWeightMatrix();
h.AddPatternMatrix(p,1/N);
or = h.GetWeightMatrix();

hDel.ResetWeightMatrix();
hDel.AddPatternMatrix(p(:,2:end));
finalStates2 = zeros(2^N,N-1);
for i = 1:size(startingStates,1)
    finalStates2(i,:) = hDel.Converge(startingStates(i,2:end));
end

for dIndex = 1:nD
    [prunedWeights,deletedWeights, remainingIndices] = h.PruneWeightMatrix(or,dValues(dIndex),'neuron');
    for kIndex = 1:nK  
        display(['Running : ' num2str(dIndex) '/' num2str(nD) ', ' num2str(kIndex) '/' num2str(nK)])
        k = kValues(kIndex);
        for strategyIndex = 1:nStrategies
            switch strategyIndex
                case 1
                    s = sum(deletedWeights);
                    compensationFactor = s*k/length(remainingIndices);
                    compensatedWeights = prunedWeights;
                    compensatedWeights(remainingIndices) = compensatedWeights(remainingIndices) + compensationFactor;
                case 2
                    s = abs(sum(deletedWeights));
                    compensationFactor = s*k/length(remainingIndices);
                    compensatedWeights = prunedWeights;
                    compensatedWeights(remainingIndices) = compensatedWeights(remainingIndices) + compensationFactor;
                case 3
                    s = abs(sum(deletedWeights));
                    compensationFactor = s*k/length(remainingIndices);
                    compensatedWeights = prunedWeights;
                    remainingWeights = compensatedWeights(remainingIndices);
                    neg = find(remainingWeights <= 0);
                    pos = find(remainingWeights > 0);
                    compensatedWeights(remainingIndices(neg)) = compensatedWeights(remainingIndices(neg))-compensationFactor;
                    compensatedWeights(remainingIndices(pos)) = compensatedWeights(remainingIndices(pos))+compensationFactor;
                case 4
                    s = sum(abs(deletedWeights));
                    compensationFactor = s*k/length(remainingIndices);
                    compensatedWeights = prunedWeights;
                    compensatedWeights(remainingIndices) = compensatedWeights(remainingIndices) + compensationFactor;
                case 5
                    s = sum(abs(deletedWeights));
                    compensationFactor = s*k/length(remainingIndices);
                    compensatedWeights = prunedWeights;
                    remainingWeights = compensatedWeights(remainingIndices);
                    neg = find(remainingWeights<=0);
                    pos = find(remainingWeights>0);
                    compensatedWeights(remainingIndices(neg)) = compensatedWeights(remainingIndices(neg))-compensationFactor;
                    compensatedWeights(remainingIndices(pos)) = compensatedWeights(remainingIndices(pos))+compensationFactor;
            end

            h.SetWeightMatrix(compensatedWeights);
            % Now, start the network with each possible starting state
            % for both the pruned and original network and test if it
            % leads to the same attractor
            for i = 1:size(startingStates,1)
                finalState = h.Converge(startingStates(i,:));

                if sum(finalState(2:end)==finalStates2(i,:)) == (N-1)
                    overlapData(kIndex,dIndex,strategyIndex) = overlapData(kIndex,dIndex,strategyIndex) + 1;
                end
            end
        end
    end
end
%%
clf
for i = 1:nStrategies
    subplot(2,3,i)
    imshow(squeeze(overlapData(:,:,i)),[0 1],'InitialMagnification','fit'),colorbar
    axis on
    set(gca,'XTick',1:nD,'XTickLabel',dValues,'YTick',1:nK,'YTickLabel',kValues)
    xlabel('Deletion factor'),ylabel('Compensation factor')
    colormap jet
    axis square
end