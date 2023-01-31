def read_class_name(class_file_name):
    names = {}
    with open(class_file_name, 'r') as f:
        for id, name in enumerate(f):
            names[id] = name.strip('\n')
    return names