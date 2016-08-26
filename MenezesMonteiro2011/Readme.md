# Synaptic compensation on Hopfield network: implications for memory rehabilitation

## Paper summary
It is has been suggested that the brain possesses compensation mechanisms to counteract the effects of synaptic or neuronal loss. In this paper, possible compensation strategies are investigate in a Hopfield network. An initial set of patterns is learned, followed by deletion of one or several neurons of the network. Synaptic compensation is modeled as the redistribution of the sum of absolute values of removed weights over the remaining synapses. A procedure in which inhibitory and excitatory weights are separately strengthened is demonstrated to yield the best result. The performance of the strategy is validated by comparing the attractor states of all possible initial states in the compensated network with the attractor states of a reduced network in which a number of neurons are removed from the beginning.


## Simulation results

## Comments
Without compensation, synaptic or neural loss can lead to a drop in recall performance when compared to an unmodified model. For example, the model might have more difficulties in coping with noise in a presented pattern, or might no longer settle in an attractor state corresponding to the original stored pattern. With this in mind, another way of assessing compensation strategy performance is to check if recall performance can be restored. That is, we should check if it is comparable to that of an original, unmodified network.