from flatten_dict import flatten, unflatten

def update_dict(old_dict, new_dict):
    # flatten, update, unflatten settings dict
    flat_old_dict = flatten(old_dict, reducer='dot')
    flat_new_dict = flatten(new_dict, reducer='dot')
    flat_old_dict.update(flat_new_dict)
    updated_old_dict = unflatten(flat_old_dict, splitter='dot')

    return updated_old_dict