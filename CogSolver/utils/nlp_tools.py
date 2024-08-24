# -*- coding:utf-8 -*-

def parse_sub_dependency_tree(depend2items, node, relation_node2parent):
    # depend->(left, right) => (node, relation_node2parent, left_children, right_children)
    if node in depend2items.keys():
        left_children, right_children = depend2items[node]
        left_children = [parse_sub_dependency_tree(depend2items, child, relation_child2node) for child, relation_child2node in left_children]
        right_children = [parse_sub_dependency_tree(depend2items, child, relation_child2node) for child, relation_child2node in right_children]
    else:
        left_children = []
        right_children = []
    return (node, relation_node2parent, left_children, right_children)

def parse_dependency_tree(dependency):
    # start from 1 => start from 0
    dependency = [(relation, depend-1, item-1) for (relation, depend, item) in dependency]
    root = -1

    # [(relation, depend, item)] => depend->(left, right)
    # depend->item->relation
    depend2items = dict()
    for (relation, depend, item) in dependency:
        if depend not in depend2items.keys():
            depend2items[depend] = dict()
        if item not in depend2items[depend].keys():
            depend2items[depend][item] = relation
    # depend->[(item, relation)]
    depend2items = {depend: sorted(items.items(), key=lambda kv: kv[0]) for depend, items in depend2items.items()}
    # depend->(left, right)
    # left/right: [(item, relation)]
    for depend, items in depend2items.items():
        for i, (item, _) in enumerate(items):
            if item > depend:
                left = items[:i]
                right = items[i:]
                break
        else:
            left = items
            right = []
        depend2items[depend] = (left, right)
    
    # (node, relation_node2parent, left_children, right_children)
    # children: [(node, relation_node2parent, left_children, right_children), ...]
    dependency_tree = None
    # only one token depend on ROOT and after ROOT
    if root in depend2items.keys() and len(depend2items[root][0]) == 0 and len(depend2items[root][1]) == 1:
        root, root_relation = depend2items[root][1][0]
        dependency_tree = parse_sub_dependency_tree(depend2items, root, root_relation)
    else:
        dependency_tree = None
    return dependency_tree
