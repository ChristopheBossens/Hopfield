# Hopfield

The Hopfield network is a classical recurrent artificial neural network that is able to store different patterns. This model has been analysed extensively, and extensions to the model have been developed that allow it to cope with low-activity patterns, improve its memory capacity, capture the central tendency of a distribution of patterns and forgetting of old patterns in favor of recollecting new patterns. Additionally, it has been used as a model for human explicit memory and has been used as a basis to model the deterioration that occurs in human hippocampus in Alzheimer's disease.

In this repository I will collect code for simulations that relate to several research papers on Hopfield neural networks. To this end, I have developed a general purpose Hopfield model class in Matlab that easily allows to change different parameters of the model and I have used this class to replicate the results of several papers. My hope is that by collecting these different modifications and extensions can provide a starting point for future research and simulation studies.

# How to use the code
```matlab
% Create a new Hopfield network with 50 units
myHopfield = Hopfield(50);

% The following configuration options are available
myHopfield.SetThreshold(0.2); % Sets the unit threshold (default = 0)
myHopfield.SetUnitModel('S'); % 'S' will use units with -1,1 activity values, 'V' will use units with 0,1
myHopfield.SetUpdateMode(updateMode); % Update mode can be 'sync' or 'async'


% Create a pattern in which 50% of the units are active
pattern = myHopfield.GeneratePattern(0.5)

% Add a pattern to the model
myHopfield.AddPattern(pattern);

% Create a distorted version of a pattern
distortedPattern = myHopfield.DistortPattern(pattern);

% Perform a single update iteration. In 'async' mode, each unit is updated
% sequentially (in random order). In 'sync' mode, all units are updated simultaneously
networkState = myHopfield.Iterate(distortedPattern)

% Network weights can be obtained with
weights = hopfield.GetWeightMatrix()

% Network can be reset using
myHopfield.ResetWeights()
```

# Example result
The following result was obtained by running the Hopfield_demo.m script. Patterns are loaded into the network sequentially and after each pattern is loaded we tested if the previous patterns if all loaded patterns were stable states. This was done by setting the network state to each pattern, iterating untill the network is stable and calculating the degree of overlap with the input pattern.

![Hopfield demo](/hopfield_demo.jpg?raw=true)


# Papers in progress:
- Hopfield, J. J., Feinstein, D. I., & Palmer, R. G. (1983). 'Unlearning' has a stabilizing effect in collective memories. Nature

- Burgess, N., & Shapiro, J. (1991). Neural network models of list learning. Network, 2, 399-422.

- Zochowski, M., Lewenstein, M., & Nowak, A. (1993). A memory which tentatively forgets. Journal of Physics A, 2099-2112.

- Morita, M. (1993). Associative memory with nonmonotone dynamics. Neural Networks, 6, 115-126.