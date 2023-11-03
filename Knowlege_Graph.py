from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate
from langchain.graphs.networkx_graph import KnowledgeTriple

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/duleesha/ChatBot/Trelis-Llama-2-7b-chat-hf-function-calling-v2.Q4_K_S.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
    temperature=0,
)


#Building the graph

texts = '''Calculus is a branch of mathematics that deals with rates of change and how things change over time. It was developed in the 17th century by Sir Isaac Newton 
and German mathematician Gottfried Wilhelm Leibniz. There are two main branches of calculus: differential calculus, which deals with rates of change, and integral calculus, 
which deals with accumulation of quantities. Calculus is used in many fields such as physics, engineering, economics, and computer science.'''

index_creator = GraphIndexCreator(llm=llm)
f_index_creator = GraphIndexCreator(llm=llm)

final_graph = f_index_creator.from_text('')

for text in texts.split("."):
    triples = index_creator.from_text(text)
    for (node1, relation, node2) in triples.get_triples():
        final_graph.add_triple(KnowledgeTriple(node1,relation,node2))


import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from((source,target ,{'relation':relation}) for source, relation, target in final_graph.get_triples())

#ploting
plt.figure(figsize=(8,5),dpi =300)
pos = nx.spring_layout(G, k= 3, seed = 0)


nx.draw_networkx_nodes(G, pos,node_size=2000)
nx.draw_networkx_edges(G, pos,edge_color='gray')
nx.draw_networkx_labels(G, pos,font_size=2)

edge_lable = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_lable,font_size=10)

#plt.axis('off')
#plt.show()

##Chain
chain = GraphQAChain.from_llm(llm, graph = final_graph, verbose =True)
chain.run("What is the calculus?")