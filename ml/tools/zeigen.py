import os


def class_name_getter(mother_trace):
    class_name_list = [
        name
        for name in sorted(os.listdir(mother_trace))
        if os.path.isdir(os.path.join(mother_trace, name))
    ]

    print("Class names loaded:")
    for index, class_name in enumerate(class_name_list):
        print(f"{index}: {class_name}")

    return class_name_list, len(class_name_list)
