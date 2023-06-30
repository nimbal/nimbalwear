from flatten_dict import flatten, unflatten

def update_settings(old_settings, new_settings):
    # flatten, update, unflatten settings dict
    flat_old_settings = flatten(old_settings, reducer='dot')
    flat_new_settings = flatten(new_settings, reducer='dot')
    flat_old_settings.update(flat_new_settings)
    settings = unflatten(flat_old_settings, splitter='dot')

    return settings