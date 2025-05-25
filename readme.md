
# ASRS + Container Innovation
see the sample video `animation.mp4`. 
sample logs of an 1 hour simulation can be found in `sample_log.txt` (most interesting stats at the bottom)

files and dependences:
- file `adjacency.csv` and `warehouse_items.csv`
- salabim
- greenlet
- Pillow
- numpy
- pandas

code is confirmed working on Python 3.9
## animation

Default: animation on. you can turn this off by commenting 

```
env.animation_parameters(animate=True, width=AnimationConfig.WINDOW_WIDTH, height=AnimationConfig.WINDOW_HEIGHT)
```
and uncommenting 
``` 
env.animation_parameters(animate=False)
```
at around line 1455


## elements
nodes 1-60 are aisle nodes. the middle is the ASRS/container node and the left/right nodes are the entry/exit nodes depending on the snake pattern. paths strictly only follow incremental nodes, such as 1->2->3 and 4->5->6 resulting in the snake pattern (see fig 9).

nodes 61 62 and 0 are the drop off, charger, and depot (start) node respectively. 

#### ASRS
- orange: idle
- red: picking items with μ = 60s, σ = 45s
- text to the left shows how many items the ASRS is picking for order number x

#### container
- the container only exists when `Innovation = True`, and is the thin rectangle under the ASRS node
- orange: empty
- green: holds items, ready to drop on AGV; this is after ASRS is done picking an item.

#### AGV
- green: idle
- orange: moving, or in an ASRS queue if another AGV is at this ASRS. the AGV in the queue will wait in the aisle until the AGV in front is done.
- red: picking items
- blue: charging at the charging station
- each AGV also has a bar under it showing its current battery level. 
- text to the right of an active AGV shows the amount of items it currently holds of its assigned order, e.g. `3/6`


## characteristics
the asrs always picks items sequentally, whether an ASRS has multiple orders queued, or when an ASRS needs to pick multiple items inside the same order. this also means the only difference of ASRS behaviour between the innovation and non-innovation is that in the case of the latter, the ASRS only starts its job when the AGV arrives. 
- example: order 1 consists of 2 items at ASRS 14. ASRS will start to pick up the first item. after its done, it will drop the item to the container (the container will now be green if it isnt already). the ASRS now starts picking up the second item. after its done, it will drop the item to the container as well. now all items of this order at this ASRS is picked up, the container turns "ready", and drops it onto the AGV with a fixed time of 15s regardless of how many items.

at the end of each AGV job (the AGV is then at the drop off node), it will check if its below the charge threshold of `35%`. if this is the case, it goes to the charging node instead of to the depot immediately. after charging, it will go the depot and becomes available. 

also, as mentioned briefly before, the ASRS has a queueing system for AGV if multiple AGVs are at an ASRS. only one AGV can be "on" the ASRS. this can be for two reasons: 
- the AGV arrived too early, and waits for the ASRS to finish picking up the item(s) and putting it in the container
- the item(s) are already picked up by the ASRS and is held by the container. 
from the AGV perspective, both are the same; the AGV stays at this ASRS node until the items are picked up. while this is going on, any other AGV that also needs to be at this ASRS stays in the aisle. only if the previous AGV is done at the ASRS and goes to its next destination will the next AGV its spot "on" the ASRS. an example of this queueing can be seen in the sample video at around 0:37 (t=4875).  

the dispatcher does the dispatching logic of AGV's for each order.

the warehouse system is the highest level, and has the dispatcher as its field. it also holds all agv and asrs in its fields. 

the asrscontainerinterface handles the communication and handling between each asrs and its container.

the warehouse uses an "adjacency matrix" which holds only direct (directed) link of a node. for example, node 6 cannot reach node 5, but it can reach nodes 6 and 55 (cross-column short cut)







