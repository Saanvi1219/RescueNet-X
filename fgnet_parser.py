
import os

def parse_fgnet(filename):

    name = filename.split(".")[0]

    person_id = name[:3]

    age = int(name[4:])

    return person_id, age
