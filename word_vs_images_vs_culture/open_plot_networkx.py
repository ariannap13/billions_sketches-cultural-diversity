import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors  # for rgb2hex

tag = "image"
top_countries=100

palette_culture = ["#7db2cf",
           "#f7c585",
           "#d9a9ed",
           "#9beab4",
           "#7afbe9",
           "#ee9482",
           "#32c2b5",
           "#e4ffda",
           "#e9d3ff",
           "#8db59a"]

palette_image = ["#7db2cf",
           "#f7c585",
           "#d9a9ed",
           "#9beab4",
           "#7afbe9",
           "#ee9482",
           "#32c2b5",
           "#e4ffda",
           "#e9d3ff",
           "#8db59a"]

palette_word = ["#65bc90",
                "#dc96bf",
                "#7df8e0",
                "#fd9e93",
                "#34d7ea",
                "#eab278",
                "#8db0f8",
                "#ccc074",
                "#acc67c",
                "#fff6c0"]

if "image" in tag:
    palette = palette_image
elif "word" in tag:
    palette = palette_word
else:
    palette = palette_culture

light_grey = "#d3d3d3ac"  # color for singleton nodes

# define custom colormap from the palette
def custom_colormap():
    return colors.ListedColormap(palette)

def plot_networkx_graph(G):
    # Extract x and y positions as floats
    x_coordinate = nx.get_node_attributes(G, 'x')
    y_coordinate = nx.get_node_attributes(G, 'y')
    pos = {
        node: (float(x_coordinate[node]), float(y_coordinate[node]))
        for node in G.nodes()
        if node in x_coordinate and node in y_coordinate
    }

    # Get modularity class, default to 0 if missing
    modularity_class = nx.get_node_attributes(G, 'Modularity Class')

    # Create node_colors: light grey for singletons, palette color otherwise
    node_colors = []
    for node in G.nodes():
        if G.degree(node) == 0:
            node_colors.append(light_grey)
        else:
            class_index = int(modularity_class.get(node, 0))
            color = palette[class_index % len(palette)]
            node_colors.append(color)

    plt.figure(figsize=(12, 12))

    # Draw nodes with bigger size
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=400,
        alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    plt.axis("off")
    if "image" in tag:
        plt.savefig(f"../data/network_plot_top{top_countries}c_image.png", dpi=300, bbox_inches='tight')
    elif "word" in tag:
        plt.savefig(f"../data/network_plot_top{top_countries}c_language.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"../data/network_plot_top{top_countries}c_culture.png", dpi=300, bbox_inches='tight')


# Load graph and plot
if "image" in tag:
    G = nx.read_gexf(f"../data/image_coordinates_top{top_countries}c_xy.gexf")
elif "word" in tag:
    G = nx.read_gexf(f"../data/language_coordinates_top{top_countries}c_xy.gexf")
else:
    G = nx.read_gexf(f"../data/culture_coordinates_top{top_countries}c_xy.gexf")
plot_networkx_graph(G)