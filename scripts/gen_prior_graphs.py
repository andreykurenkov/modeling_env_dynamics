import yaml
import prior
from memsearch.graphs import *
from collections import defaultdict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

PROBS_FILE_PATH = 'priors/hardcoded_placement_probs.yaml'

def camel_to_snake(s):
    return ''.join(['_'+c.lower() if c.isupper() else c for c in s]).lstrip('_')

def gen_priors_graph(group_by_parent=True):
    # makes a hand-coded graph that is an example of a prior graph
    room_nodes = {}
    furniture_nodes = {}
    object_nodes = {}

    house_node = PriorsNode('house',NodeType.HOUSE,'house')
    floor_node1 = PriorsNode('floor1',NodeType.FLOOR,'floor')
    #floor_node2 = PriorsNode('floor2',NodeType.FLOOR)

    top_nodes = [house_node, floor_node1]
    Edge(house_node, floor_node1, EdgeType.CONTAINS, 1.0)

    room_furniture_probs = defaultdict(lambda: defaultdict(lambda: 0))
    furniture_object_probs = defaultdict(lambda: defaultdict(lambda: 0))
    room_furniture_object_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    #forget about floor 2 for now
    #Edge(house_node,floor_node2,EdgeType.CONTAINS, 0.1).add_to_nodes()
    #Edge(floor_node1,floor_node2,EdgeType.CONNECTED, 1.0).add_to_nodes()

    # hard code this for now
    room_names = ['kitchen', 'living_room','bedroom', 'bathroom']
    for room_name in room_names:
        room_nodes[room_name] = PriorsNode(room_name, NodeType.ROOM, room_name)
        Edge(floor_node1, room_nodes[room_name], EdgeType.CONTAINS, 1.0)

    # ugly piece of logic to take the old-format hardcoded file and load it into a prior graph
    with open(PROBS_FILE_PATH, 'r') as probs_file:
        probs = yaml.safe_load(probs_file)
        # ignore
        for obj_label in probs:
            obj_label_dict = probs[obj_label]
            for obj_instance in obj_label_dict:
                edges = obj_label_dict[obj_instance]
                for edge_name in edges:
                    furniture_type, room_type, edge_type = edge_name.split('-')
                    edge_prob = edges[edge_name]
                    if room_type not in room_nodes:
                        continue

                    if group_by_parent:
                        furniture_cat = furniture_type+'-'+room_type
                    else:
                        furniture_cat = furniture_type
                    if furniture_cat not in furniture_nodes:
                        furniture_nodes[furniture_cat] = PriorsNode(furniture_type, NodeType.FURNITURE, furniture_cat)

                    furniture_node = furniture_nodes[furniture_cat]

                    room_node = room_nodes[room_type]
                    if not furniture_node.has_edges_to(room_node):
                        Edge(room_node, furniture_node, EdgeType.CONTAINS, 0)
                    room_furniture_probs[room_type][furniture_type] = edge_prob

                    if group_by_parent:
                        obj_cat = room_type+'-'+obj_label
                    else:
                        obj_cat = obj_label
                    if obj_cat not in object_nodes:
                        object_nodes[obj_cat] = PriorsNode(obj_label, NodeType.OBJECT, obj_cat)
                    object_node = object_nodes[obj_cat]

                    if not object_node.has_edges_to(furniture_node):
                        edge_type_enum = RECIPROCAL_EDGE_TYPES[edge_enum_from_str(edge_type)]
                        Edge(furniture_node, object_node, edge_type_enum, edge_prob)
                    furniture_object_probs[furniture_type][obj_label] = edge_prob
                    room_furniture_object_probs[room_type][furniture_type][obj_label] = edge_prob
                # don't care about other instances lol
                break

    print('Loading procthor-10k...')
    dataset = prior.load_dataset("procthor-10k")['train']
    room_furniture_counts = defaultdict(lambda: defaultdict(lambda: 0))
    furniture_object_counts = defaultdict(lambda: defaultdict(lambda: 0))
    room_furniture_object_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    for i in range(len(dataset)):
        house = dataset[i]
        room_polygons = {}
        rooms_json = house['rooms']
        for i, room in enumerate(rooms_json):
            room_type = camel_to_snake(room['roomType'])
            polygon_indices = []
            for polygon_element in room['floorPolygon']:
                polygon_indices.append((polygon_element['x'],polygon_element['z']))
            polygon = Polygon(polygon_indices)
            room_polygons[room_type+'-'+str(i)] = polygon
            #x,y = polygon.exterior.xy
            #plt.plot(x,y)
            #plt.show()

        objects_json = house['objects']
        for obj_json in objects_json:
            obj_type = camel_to_snake(obj_json['id'].split('|')[0])
            obj_pos = obj_json['position']
            obj_pos = Point(obj_pos['x'], obj_pos['z'])
            is_furniture = 'children' in obj_json

            for numbered_room_type in room_polygons:
                if room_polygons[numbered_room_type].contains(obj_pos):
                    room_type = numbered_room_type.split('-')[0]
                    if is_furniture:
                        furniture_type = obj_type
                        room_furniture_counts[room_type][furniture_type]+=1
                    break

            if room_type not in room_names:
                continue

            if is_furniture:
                if group_by_parent:
                    furniture_cat = furniture_type+'-'+room_type
                else:
                    furniture_cat = furniture_type

                if furniture_cat not in furniture_nodes:
                    furniture_nodes[furniture_cat] = PriorsNode(furniture_type, NodeType.FURNITURE, furniture_cat)
                furniture_node = furniture_nodes[furniture_cat]

                room_node = room_nodes[room_type]
                if not furniture_node.has_edges_to(room_node):
                    Edge(room_node, furniture_node, EdgeType.CONTAINS, 0)

                object_counts = defaultdict(lambda: 0)
                for child_obj_json in obj_json['children']:
                    obj_pos = child_obj_json['position']
                    obj_pos = Point(obj_pos['x'], obj_pos['z'])
                    obj_type = camel_to_snake(child_obj_json['id'].split('|')[0])
                    object_counts[obj_type]+=1

                for obj_type,object_count in object_counts.items():
                    furniture_object_counts[furniture_type][obj_type]+=object_count
                    room_furniture_object_counts[room_type][furniture_type][obj_type]+=object_count
                    if group_by_parent:
                        obj_cat = room_type+'-'+obj_type
                    else:
                        obj_cat = obj_type

                    if obj_cat not in object_nodes:
                        object_nodes[obj_cat] = PriorsNode(obj_type, NodeType.OBJECT, obj_cat)
                    object_node = object_nodes[obj_cat]

                    if not object_node.has_edges_to(furniture_node):
                        Edge(furniture_node, object_node, EdgeType.UNDER, 0.0)
            else:
                pass

    for room_node in room_nodes.values():
        room_type = room_node.label
        furniture_counts = room_furniture_counts[room_type]
        furniture_probs = room_furniture_probs[room_type]
        furniture_max_count = float(max(list(furniture_counts.values())))
        for furniture_node in room_node.get_children_nodes():
            edge = room_node.get_edge_to(furniture_node)
            count = float(furniture_counts[furniture_node.label])
            ig_prob = furniture_probs[furniture_node.label]
            if count != 0:
                edge.prob = count/furniture_max_count
            else:
                edge.prob = ig_prob
            if group_by_parent:
                object_counts = room_furniture_object_counts[room_type][furniture_node.label]
                object_probs = room_furniture_object_counts[room_type][furniture_node.label]
                if len(object_counts.values()) != 0:
                    objects_max_count = float(max(list(object_counts.values())))
                else:
                    objects_max_count = 1
                for object_node in furniture_node.get_children_nodes():
                    ig_prob = object_probs[object_node.label]
                    count = float(object_counts[object_node.label])
                    edge = furniture_node.get_edge_to(object_node)
                    if count != 0:
                        edge.prob = count/objects_max_count
                    else:
                        edge.prob = ig_prob

    if not group_by_parent:
        for furniture_node in furniture_nodes.values():
            furniture_type = furniture_node.label
            if furniture_type not in furniture_object_counts:
                continue
            object_counts = furniture_object_counts[furniture_type]
            object_probs = furniture_object_probs[furniture_type]
            objects_max_count = float(max(list(object_counts.values())))
            for object_type, count in object_counts.items():
                object_node = object_nodes[object_type]
                edge = furniture_node.get_edge_to(object_node)
                ig_prob = object_probs[object_type]
                if count != 0:
                    edge.prob = float(count)/objects_max_count
                else:
                    edge.prob = ig_prob

    '''
    low_edge_object_nodes = []
    for key,node in object_nodes.items():
        if len(node.get_edges_to_me()) < 3:
            low_edge_object_nodes.append(key)
            for parent_node in node.get_parent_nodes():
                parent_node.remove_edges_to(node)

    for key in low_edge_object_nodes:
        del object_nodes[key]
    '''

    low_edge_furniture_nodes = []
    for key,node in furniture_nodes.items():
        if len(node.get_edges_from_me()) < 3:
            low_edge_furniture_nodes.append(key)
            for parent_node in node.get_parent_nodes():
                parent_node.remove_edges_to(node)
            for child_node in node.get_children_nodes():
                child_node.remove_edges_to(node)

    for key in low_edge_furniture_nodes:
        del furniture_nodes[key]
    graph = PriorsGraph(top_nodes + \
                        list(room_nodes.values()) + \
                        list(furniture_nodes.values()) + \
                        list(object_nodes.values()))

    #for node in graph.nodes:
    #    node.normalize_edges_from_me()

    return graph

def enrich_graph_with_metadata(graph):
    with open('priors/object_metadata.yaml','r') as f:
        metadata = yaml.safe_load(f)

    nodes_to_add = []
    nodes_to_remove = []
    for node in graph.nodes:
        if node.label not in metadata:
            nodes_to_remove.append(node)
            continue
        node_dict = metadata[node.label]
        node.unique_id = node_dict['unique_id']
        node.label = node_dict['label']
        node.max_count = node_dict['max_count']
        node.move_freq = float(node_dict['move_freq'])
        node.sample_prob = float(node_dict['sample_prob'])
        if 'spawn_prob' in node_dict:
            node.spawn_prob = node_dict['spawn_prob']
            node.remove_prob = node_dict['spawn_prob']
        else:
            node.spawn_prob = 0.0
            node.remove_prob = 0.0
        node.adjectives = node_dict['adjectives']
        if 'same_probs' not in node_dict:
            continue
        for new_label in node_dict['same_probs']:
            new_node = node.copy(with_unique_id=False, with_edge_copies=True, exclude_edges=True)
            new_node.label = new_label
            new_node.category = node.category.replace(node.label, new_label)
            new_node.unique_id = node.unique_id.replace(node.label, new_label)
            nodes_to_add.append(new_node)
    for node in nodes_to_remove:
        graph.remove_node(node)
    for node in nodes_to_add:
        graph.add_node(node)

if __name__ == "__main__":
    print('Generating coarse priors graph...')
    print('---------------------------------')
    graph = gen_priors_graph(group_by_parent=False)
    print('Coase priors graph has %d nodes and %d edges'%(len(graph.nodes), len(graph.get_edges())))
    enrich_graph_with_metadata(graph)
    print('Coarse priors graph with metadata has %d nodes and %d edges\n'%(len(graph.nodes), len(graph.get_edges())))
    save_graph(graph,'priors/coarse_prior_graph.pickle')
    graph.save_png('priors/coarse_priors_graph.png')

    print('Generating detailed priors graph...')
    print('---------------------------------')
    graph = gen_priors_graph(group_by_parent=True)
    print('Detailed priors graph has %d nodes and %d edges'%(len(graph.nodes), len(graph.get_edges())))
    enrich_graph_with_metadata(graph)
    print('Detailed priors graph with metadata has %d nodes and %d edges'%(len(graph.nodes), len(graph.get_edges())))
    save_graph(graph,'priors/detailed_prior_graph.pickle')
    graph.save_png('priors/detailed_priors_graph.png')
