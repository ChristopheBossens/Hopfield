# Forgetting in Hopfield network

A major drawback of the stand Hopfield network implementation is that as the critical capacity is reached, the network goes into a state were none of the original patterns can be recovered and no new patterns can be stored. One approach to counteract this problem is to forget old patterns in favor of new memory patterns. This can be implemented using a method proposed by Parisi(1986). The central idea is that some saturation function should be applied to the weight matrix after adding each new pattern. This is in contrast to the original Hopfield model in which the weights are allowed to grow without bounds. 

![Hopfield demo](/Forgetting/parisi_forgetting.jpg?raw=true)

