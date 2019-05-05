# Classification of User Movements in Heterogeneous Indoor Environments

The main objective of this project is use predictive modeling for indoor user movement classification. This would
have many useful applications in Ambient Assisted Living contexts.

The data considered for this project was collected in the measurement campaign reported in [1]. The
measurement campaign included experiments on three different couple of rooms with a total surface spanning
from 50 m2 to about 60 m2. Each couple of rooms correspond to a given dataset (measurement group) referred
as dataset 1, dataset 2 and dataset 3.

Experiments consisted in measuring the Radio Signal Strength between anchors and a mobile mote attached to
the user (worn on the chest) for a set of repeated movements. Figure below shows the anchors deployed in the
environment as well as a prototypical trajectory for each type of user movement. Measures denote RSS samples
collected by sending a beacon packet from the anchors to the mobile at regular intervals, 8 times per second
(8Hz), using the full transmission power of the IRIS.

Experimentation gathered information on 6 prototypical paths that are shown in Figure below with arrows
numbered from 1 to 6: two movement types are considered for the prediction task, that are straight and curved
trajectories. Straight movements run from Room 1 to Room 2 or vice versa (paths 1 and 5 in Figure below) and
yields to a change in the spatial context of the user, while curved movements (paths 2, 3, 4 and 6 in Figure
below) maintain the same spatial context. Table 2 summarizes the statistics of the collected movement types
for each dataset: due to physical constraints, dataset 1 does not have a curved movement in Room 1 (path 3).

![Alt text](./plots/MovementAAL.jpg?raw=true "Experiement Rooms Config")
[1] https://archive.ics.uci.edu/ml/datasets/Indoor+User+Movement+Prediction+from+RSS+data

Check out this [notebook](./eda.ipynb) for a brief data exploration