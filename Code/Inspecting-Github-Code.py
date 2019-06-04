import requests
import json

user = 'eddie-c64'
ps = 'Eddie!1992'

# Note that credentials will be transmitted over a secure SSL connection
url = 'https://api.github.com/authorizations'
note = 'Mining the Social Web - Mining Github'
post_data = {'scopes':['repo'],'note': note }

response = requests.post(
    url,
    auth = (user, ps),
    data = json.dumps(post_data),
    )

print("API response:", response.text, '\n')
# print("Your OAuth token is", response.json()['token'], '\n')


# An unauthenticated request that doesn't contain an ?access_token=xxx query string
url = "https://api.github.com/repos/ptwobrussell/Mining-the-Social-Web/stargazers"
response = requests.get(url)

# Display one stargazer
print(json.dumps(response.json()['token'], indent=1))
print()

# Display headers
for (k,v) in response.headers.items():
    print(k, "=>", v)

# Using PyGraph to query for stargazers of a particular repository
from github import Github # pip install pygithub

# XXX: Specify your own access token here

ACCESS_TOKEN = ''

# Specify a username and repository of interest for that user.
USER = 'ptwobrussell'
REPO = 'Mining-the-Social-Web'
#REPO = 'Mining-the-Social-Web-2nd-Edition'

client = Github(ACCESS_TOKEN, per_page=100)
user = client.get_user(USER)
repo = user.get_repo(REPO)

# Get a list of people who have bookmarked the repo.
# Since you'll get a lazy iterator back, you have to traverse
# it if you want to get the total number of stargazers.

stargazers = [ s for s in repo.get_stargazers() ]
print("Number of stargazers", len(stargazers))

import networkx as nx # pip install networkx

# Create a directed graph

g = nx.DiGraph()

# Add an edge to the directed graph from X to Y

g.add_edge('X', 'Y')

# Print some statistics about the graph

print(nx.info(g))

# Get the nodes and edges from the graph

print("Nodes:", g.nodes())
print("Edges:", g.edges())
print()

# Get node properties

print("X props:", g.node['X'])
print("Y props:", g.node['Y'])
print()

# Get edge properties

print("X=>Y props:", g['X']['Y'])
print()

# Update a node property

g.node['X'].update({'prop1' : 'value1'})
print("X props:", g.node['X'])
print()

# Update an edge property

g['X']['Y'].update({'label' : 'label1'})
print("X=>Y props:", g['X']['Y'])

"""
 Constructing an ego graph of a repository and it's stargazer
 Expand the initial graph with (interest) edges pointing each direction for
 additional people interested. Take care to ensure that user and repo nodes
 do not collide by appending their type.
"""

g = nx.DiGraph()
g.add_node(repo.name + '(repo)', type='repo', lang=repo.language, owner=user.login)

for sg in stargazers:
    g.add_node(sg.login + '(user)', type='user')
    g.add_edge(sg.login + '(user)', repo.name + '(repo)', type='gazes')


# Let's Poke around in the current Graph to get a better feel for how NetworkX works
print(nx.info(g))
print(g.node['Mining-the-Social-Web(repo)'])
print(g.node['ptwobrussell(user)'])
print(g['ptwobrussell(user)']['Mining-the-Social-Web(repo)'])

print(g['ptwobrussell(user)'])
print(g['Mining-the-Social-Web(repo)'])
print(g.in_edges(['ptwobrussell(user)']))
print(g.out_edges(['ptwobrussell(user)']))
print(g.in_edges(['Mining-the-Social-Web(repo)']))
print(g.out_edges(['Mining-the-Social-Web(repo)']))

# Calculating degree, betweenness and closeness certianly measure on the Krackhardt kite Graph
from operator import itemgetter
from IPython.display import HTML
from IPython.core.display import display

display(HTML('<img src="resources/ch08-github/kite-graph.png" width="400px">'))

# The classic Krackhardt kite graph
kkg = nx.generators.small.krackhardt_kite_graph()

print("Degree Centrality")
print(sorted(nx.degree_centrality(kkg).items(),
             key=itemgetter(1), reverse=True))
print()

print("Betweenness Centrality")
print(sorted(nx.betweenness_centrality(kkg).items(),
             key=itemgetter(1), reverse=True))
print()

print("Closeness Centrality")
print(sorted(nx.closeness_centrality(kkg).items(),
             key=itemgetter(1), reverse=True))

# Adding additional interest edges to the graph through the inclusion of "follows" edge

import sys

for i, sg in enumerate(stargazers):

    # Add "follows" edges between stargazers in the graph if any relationships exist
    try:
        for follower in sg.get_followers():
            if follower.login + '(user)' in g:
                g.add_edge(follower.login + '(user)', sg.login + '(user)',
                           type='follows')
    except Exception as e: #ssl.SSLError
        print("Encountered an error fetching followers for", sg.login, \
              "Skipping.", file=sys.stderr)
        print(e, file=sys.stderr)

    print("Processed", i+1, " stargazers. Num nodes/edges in graph", \
          g.number_of_nodes(), "/", g.number_of_edges())
    print("Rate limit remaining", client.rate_limiting)


# Exploring the updated graphs
from operator import itemgetter
from collections import Counter

# Let's see how many social edges we added since last time.
print(nx.info(g))

print(len([e for e in g.edges_iter(data=True) if e[2]['type'] == 'follows']))
print(len([e
           for e in g.edges_iter(data=True)
               if e[2]['type'] == 'follows' and e[1] == 'ptwobrussell(user)']))
print(sorted([n for n in g.degree_iter()], key=itemgetter(1), reverse=True)[:10])

print(len(g.out_edges('angusshire(user)')))
print(len(g.in_edges('angusshire(user)')))


# A user who is followed by many but does not follow back.

print(len(g.out_edges('ptwobrussell(user)')))
print(len(g.in_edges('ptwobrussell(user)')))

c = Counter([e[1] for e in g.edges_iter(data=True) if e[2]['type'] == 'follows'])
popular_users = [ (u, f) for (u, f) in c.most_common() if f > 1 ]
print("Number of popular users", len(popular_users))
print("Top 10 popular users:", popular_users[:10])

# Save the graph to disk
nx.write_gpickle(g, "data/github.gpickle.1")

### Apply centrality measures to the interst graphs
from operator import itemgetter
h = g.copy()
h.remove_node('Mining-the-Social-Web(repo)')

doc = sorted(
            nx.degree_centrality(h).items(),
            key=itemgetter(1),
            reverse=True
            )

print("Degree Centrality")
print(dc[:10])
print()

bc = sorted(nx.betweenness_centrality(h).items(),
            key=itemgetter(1), reverse=True)

print("Betweenness Centrality", bc[:10])

print('Closeness Centrality')
cc = sorted(nx.closeness_centrality(h).items(), key=itemgetter(1), reverse=True)
print(cc[:10])

# Adding additional interest edges to the graphs
import sys

for i, sg in enumerate(stargazers):

    # Add 'follows' from edge repository stargazers in graphs
    try:
        for follower in sg.get_followers():
            if follow.login + '(user)' in g:
                g.add_edge(
                            follower.login + 'user',
                            sg.login + '(user)',
                            type='follows'
                            )

    except Exception as e:
        print('Encountered an error fetching followers for', sg.login,
                'Skipping.',
                file=sys.stderr)
                print(e, file=sys.stderr)

    print("Processed", i+1, "stargazers. Number of nodes/edges in graph",
            g.number_of_nodes(), '/', g.number_of_egdes())
            
    print("Rate4 limit remaining", client.rate_limiting)
