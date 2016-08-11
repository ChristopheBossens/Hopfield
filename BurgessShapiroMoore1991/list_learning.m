networkSize = 100;
nPatterns = 40;
gamma = 0.95;
epsilon = 0.1;
bins = [-1:0.2:1];
nSimulations = 30;
hopfield = Hopfield(networkSize);

weightDistribution = zeros(nPatterns,length(bins));
for simulationIndex = 1:nSimulations
    patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);
    weightMatrix = hopfield.GetWeightMatrix();
    for patternIndex = 1:nPatterns  
       weightMatrix = gamma.*weightMatrix + epsilon.*(patternMatrix(patternIndex,:)'*patternMatrix(patternIndex,:));
       weightMatrix(weightMatrix>1) = 1;
       weightMatrix(weightMatrix < -1) = -1;
       
       dist = hist(weightMatrix(:),bins);
       weightDistribution(patternIndex,:) = weightDistribution(patternIndex,:) + dist./sum(dist);
    end
end
%% Code to reproduce the weight distributions from figure 3
weightDistribution = weightDistribution./nSimulations;
X = repmat( (1:nPatterns)',1,length(bins));
Y = repmat(bins,nPatterns,1);
Z = weightDistribution;

newBin = -1:0.01:1;
newPattern = 1:1:nPatterns;
Xi = repmat( newPattern',1,length(newBin));
Yi = repmat( newBin,length(newPattern),1);
Zi = griddata(X,Y,Z,Xi,Yi,'cubic');

H = fspecial('gaussian',[9 9],1.5);
Zf = imfilter(Zi,H,'same');
clf,
surf(Xi,Yi,Zf,'EdgeColor','None')
xlabel('# patterns')
ylabel('Weights')
title(['gamma = ' num2str(gamma)])

%% Code to reproduce the serial position curves from figure 4