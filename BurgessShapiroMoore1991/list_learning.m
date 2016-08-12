networkSize = 100;
nPatterns = 40;
gamma = 1.05;
epsilon = 0.1;
bins = [-1:0.2:1];
nSimulations = 10;
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
clf,hold on
plot(bins,weightDistribution(10,:),'-k','LineWidth',2)
plot(bins,weightDistribution(20,:),'--k','LineWidth',2)
plot(bins,weightDistribution(30,:),':k','LineWidth',2)
xlabel('Weight value')
ylabel('Proportion')
title('Weight distribution')
legend('10 patterns','20 patterns','30 patterns')

%% Code to construct a figure that show how the distribution of weights evolves after each pattern is added
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

%% Simulations for producing the serial position curves in figure 4
gamma = 1.25;
epsilon = 0.2;
noiseLevel = 0.2;
nSimulations = 10;

overlapMatrix = zeros(nPatterns,nPatterns);
for simulationIndex = 1:nSimulations
    hopfield = Hopfield(networkSize);
    patternMatrix = hopfield.GeneratePatternMatrix(nPatterns,0.5);
    weightMatrix = hopfield.GetWeightMatrix();
    for patternIndex = 1:nPatterns
        weightMatrix = gamma.*weightMatrix + epsilon.*(patternMatrix(patternIndex,:)'*patternMatrix(patternIndex,:));
        weightMatrix(weightMatrix>1) = 1;
        weightMatrix(weightMatrix < -1) = -1;

        hopfield.SetWeightMatrix(weightMatrix);

        for testIndex = 1:patternIndex
            originalPattern = patternMatrix(testIndex,:);
            testPattern = hopfield.DistortPattern(originalPattern,noiseLevel);
            finalState = hopfield.Converge(testPattern);

            overlapMatrix(testIndex,patternIndex) = overlapMatrix(testIndex,patternIndex) + (originalPattern*finalState')/networkSize;
        end
    end
end
overlapMatrix = overlapMatrix./nSimulations;
%% Code for generating figure 4
clf,hold on
plot(1:10,overlapMatrix(1:10,10),'-k','LineWidth',2)
plot(1:20,overlapMatrix(1:20,20),'--k','LineWidth',2)
plot(1:30,overlapMatrix(1:30,30),':k','LineWidth',2)
legend('10 patterns','20 patterns','30 patterns')
xlabel('N° trained patterns')
ylabel('Pattern overlap')
title('Serial position curve (\gamma = 1.25, \epsilon = 0.2)')
%%
clf,
imshow(overlapMatrix,[0 1],'InitialMagnification','fit'),
colorbar
colormap jet
ylabel('Pattern tested')
xlabel('Pattern trained')
title('Overlap matrix (30 patterns)')
