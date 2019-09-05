import networkx as nx
import matplotlib.pyplot as plt
import re
import agents
import GridWorld
import constants


players = [agents.Agent(constants.PLAYER1_CHAR),
           agents.Agent(constants.PLAYER2_CHAR),
           agents.Agent(constants.DRAGON_CHAR)]

world1 = GridWorld.World(players, False)
G = world1.graph
pos = dict( (n, n) for n in G.nodes() )

for k, p in pos.items():
    a = re.findall('[0-9]{1,2}', p)
    a[0] = int(a[0])
    a[1] = int(a[1])
    new_p = tuple(a)
    pos[k]=new_p

labels = dict((i, '') for i in G.nodes())
options = {
    'node_color': 'C0',
    'node_size': 100,
}
nx.draw_networkx(G, pos=pos, labels=labels, **options)
plt.axis('off')
plt.gca().invert_yaxis()
plt.savefig('graph.png')
plt.show()
